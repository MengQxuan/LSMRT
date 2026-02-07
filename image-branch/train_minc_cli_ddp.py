#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""train_minc_cli_ddp.py (robust txt parser + DDP + mixup correct loss)

Fixes in this revision:
- Robust txt parsing: supports both "path label" and "label path" per line, and ignores blank/comment lines.
- num_classes inferred from parsed labels (robust).
- mixup/cutmix -> SoftTargetCrossEntropy.
- rank0-only save/load best.pt with barrier to avoid corruption.
- AMP uses torch.amp.autocast('cuda').

NCCL timeout/卡死排查建议（可选）：
  export NCCL_ASYNC_ERROR_HANDLING=1
  export NCCL_BLOCKING_WAIT=1
  export TORCH_DISTRIBUTED_DEBUG=DETAIL
  export NCCL_DEBUG=INFO

DDP launch (2 GPUs):
  CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=29501 \
    train_minc_cli_ddp.py --ddp --minc_root /root/mqx/LSMRT/data/minc-2500 --split 1 \
    --arch tf_efficientnet_b1 --weights_path /root/mqx/LSMRT/model.pth \
    --epochs 80 --batch 64 --lr 8e-4 --wd 2e-2 --strong_aug \
    --mixup 0.2 --cutmix 0.5 --mixup_prob 0.5 --ema --ema_decay 0.9999 --amp \
    --seed 42 --num_workers 8
"""

import argparse, os, sys, time, random
from typing import List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler

import timm
from timm.data import create_transform
from timm.data.mixup import Mixup
from timm.loss import SoftTargetCrossEntropy
from timm.utils import ModelEmaV2


# -------------------------
# DDP helpers
# -------------------------
def is_dist() -> bool:
    return dist.is_available() and dist.is_initialized()

def get_rank() -> int:
    return dist.get_rank() if is_dist() else 0

def get_world() -> int:
    return dist.get_world_size() if is_dist() else 1

def is_main() -> bool:
    return get_rank() == 0

def barrier():
    if is_dist():
        dist.barrier()

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def setup_ddp(args):
    if not args.ddp:
        return 0, 1, 0
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        raise RuntimeError('DDP requested but torchrun env vars not found. Use torchrun to launch.')
    rank = int(os.environ['RANK'])
    world = int(os.environ['WORLD_SIZE'])
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    return rank, world, local_rank

def cleanup_ddp():
    if is_dist():
        dist.destroy_process_group()

def now_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


# -------------------------
# Robust txt parsing
# -------------------------
def parse_txt_line(line: str) -> Optional[Tuple[str, int]]:
    """Return (rel_path, label) or None to skip.
    Supports:
      - "images/xxx.jpg 0"
      - "0 images/xxx.jpg"
    Ignores:
      - blank
      - comment lines starting with # or //
    """
    s = line.strip()
    if not s:
        return None
    if s.startswith('#') or s.startswith('//'):
        return None
    parts = s.split()
    if len(parts) < 2:
        return None

    a, b = parts[0], parts[1]

    # try path label
    try:
        y = int(b)
        rel = a
        return rel, y
    except Exception:
        pass

    # try label path
    try:
        y = int(a)
        rel = b
        return rel, y
    except Exception:
        return None


# -------------------------
# Dataset
# -------------------------
class MincTxtDataset(Dataset):
    def __init__(self, minc_root: str, txt_path: str, transform=None, debug_first: bool = False):
        self.minc_root = minc_root
        self.txt_path = txt_path
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []

        with open(txt_path, 'r') as f:
            for line in f:
                parsed = parse_txt_line(line)
                if parsed is None:
                    continue
                self.samples.append(parsed)

        if len(self.samples) == 0:
            raise RuntimeError(f'num_samples=0 from txt: {txt_path} (check content format)')

        if debug_first and is_main():
            print(f'[DBG] First 3 samples from {os.path.basename(txt_path)}:')
            for s in self.samples[:3]:
                print(' ', s)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        rel, y = self.samples[idx]
        img_path = os.path.join(self.minc_root, rel)
        if not os.path.exists(img_path):
            raise FileNotFoundError(f'Image not found: {img_path}')
        img = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img, y


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def ddp_gather_concat(t: torch.Tensor) -> torch.Tensor:
    if not is_dist():
        return t
    out = [torch.zeros_like(t) for _ in range(get_world())]
    dist.all_gather(out, t)
    return torch.cat(out, dim=0)

@torch.no_grad()
def evaluate(model, loader, device, amp: bool, num_classes: int):
    model.eval()
    correct, total = 0, 0
    preds_all, tgts_all = [], []

    autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=amp)

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with autocast_ctx:
            logits = model(x)
        p = logits.argmax(dim=1)
        correct += (p == y).sum().item()
        total += y.numel()
        preds_all.append(p.detach().cpu())
        tgts_all.append(y.detach().cpu())

    pack = torch.tensor([correct, total], device=device, dtype=torch.long)
    if is_dist():
        dist.all_reduce(pack, op=dist.ReduceOp.SUM)
    correct_g, total_g = pack.tolist()
    acc = correct_g / max(1, total_g)

    preds = torch.cat(preds_all, dim=0).to(device)
    tgts = torch.cat(tgts_all, dim=0).to(device)
    preds = ddp_gather_concat(preds)
    tgts = ddp_gather_concat(tgts)

    f1s = []
    for c in range(num_classes):
        tp = ((preds == c) & (tgts == c)).sum().item()
        fp = ((preds == c) & (tgts != c)).sum().item()
        fn = ((preds != c) & (tgts == c)).sum().item()
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    f1 = float(np.mean(f1s)) if f1s else 0.0
    return acc, f1


# -------------------------
# Checkpoint loading: drop mismatched head
# -------------------------
def load_ckpt_drop_head(model: nn.Module, ckpt_path: str):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if isinstance(ckpt, dict) and 'state_dict' in ckpt:
        state = ckpt['state_dict']
    elif isinstance(ckpt, dict) and 'model' in ckpt:
        state = ckpt['model']
    else:
        state = ckpt

    state2 = {}
    for k, v in state.items():
        if k.startswith('module.'):
            k = k[7:]
        state2[k] = v

    ms = model.state_dict()
    filtered, dropped = {}, []
    for k, v in state2.items():
        if k in ms and tuple(v.shape) == tuple(ms[k].shape):
            filtered[k] = v
        else:
            if any(h in k for h in ['classifier', 'head', 'fc']):
                dropped.append(k)

    if dropped and is_main():
        print(f"[INFO] Dropped {len(dropped)} mismatched head keys from checkpoint (e.g. {dropped[:2]})")

    msg = model.load_state_dict(filtered, strict=False)
    if is_main() and hasattr(msg, 'missing_keys'):
        print(f"[INFO] Missing keys when loading (ok for finetune): {msg.missing_keys[:10]}{'...' if len(msg.missing_keys)>10 else ''}")


# -------------------------
# Training
# -------------------------
def train_one_epoch(model, loader, optimizer, device, scaler, criterion, amp: bool, mixup_fn: Optional[Mixup]):
    model.train()
    autocast_ctx = torch.amp.autocast(device_type='cuda', enabled=amp)

    total_loss, n = 0.0, 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if mixup_fn is not None:
            x, y = mixup_fn(x, y)

        optimizer.zero_grad(set_to_none=True)
        with autocast_ctx:
            logits = model(x)
            loss = criterion(logits, y)

        if amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += float(loss.detach().item())
        n += 1

    pack = torch.tensor([total_loss, n], device=device, dtype=torch.float32)
    if is_dist():
        dist.all_reduce(pack, op=dist.ReduceOp.SUM)
    total_loss_g, n_g = pack.tolist()
    return total_loss_g / max(1.0, n_g)


def build_transform(img_size: int, strong_aug: bool, is_train: bool):
    if is_train:
        if strong_aug:
            return create_transform(
                input_size=img_size,
                is_training=True,
                auto_augment='rand-m9-mstd0.5-inc1',
                interpolation='bicubic',
                re_prob=0.25,
                re_mode='pixel',
                re_count=1,
            )
        return create_transform(input_size=img_size, is_training=True, interpolation='bicubic', re_prob=0.0)
    return create_transform(input_size=img_size, is_training=False, interpolation='bicubic')


def infer_num_classes(txt_paths: List[str]) -> int:
    max_y = -1
    bad = 0
    for p in txt_paths:
        with open(p, 'r') as f:
            for line in f:
                parsed = parse_txt_line(line)
                if parsed is None:
                    continue
                _, y = parsed
                max_y = max(max_y, y)
    if max_y < 0:
        raise RuntimeError('Failed to infer num_classes (no valid labels parsed).')
    return max_y + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--minc_root', type=str, required=True)
    ap.add_argument('--split', type=int, default=1)
    ap.add_argument('--arch', type=str, default='tf_efficientnet_b1')
    ap.add_argument('--pretrained', type=int, default=0)
    ap.add_argument('--weights_path', type=str, default='')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--epochs', type=int, default=10)
    ap.add_argument('--batch', type=int, default=64, help='Per-GPU batch in DDP.')
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--wd', type=float, default=1e-2)
    ap.add_argument('--num_workers', type=int, default=8)
    ap.add_argument('--seed', type=int, default=42)

    ap.add_argument('--strong_aug', action='store_true')
    ap.add_argument('--amp', action='store_true')
    ap.add_argument('--ema', action='store_true')
    ap.add_argument('--ema_decay', type=float, default=0.9999)

    ap.add_argument('--mixup', type=float, default=0.0)
    ap.add_argument('--cutmix', type=float, default=0.0)
    ap.add_argument('--mixup_prob', type=float, default=0.0)
    ap.add_argument('--mixup_switch_prob', type=float, default=0.5)
    ap.add_argument('--mixup_mode', type=str, default='batch', choices=['batch', 'pair', 'elem'])

    ap.add_argument('--ddp', action='store_true')
    ap.add_argument('--debug_first', action='store_true')

    ap.add_argument('--out_dir', type=str, default='runs_minc')
    ap.add_argument('--save_name', type=str, default='')
    ap.add_argument("--dl_timeout", type=int, default=0, help="DataLoader timeout seconds (0=disabled). Helps detect hangs.")
    ap.add_argument("--persistent_workers", type=int, default=0, help="1 to keep DataLoader workers alive between epochs; set 0 to avoid rare hangs.")
    ap.add_argument("--prefetch_factor", type=int, default=2, help="DataLoader prefetch_factor (only if num_workers>0).")

    args = ap.parse_args()

    rank, world, local_rank = setup_ddp(args)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if is_main():
        print(f'device: {device.type}')
        print(f'visible_gpus: {torch.cuda.device_count() if device.type == "cuda" else 0}')
        if args.ddp:
            print(f'[INFO] DDP rank={rank} world_size={world} local_rank={local_rank}')

    seed_all(args.seed + rank)

    labels_dir = os.path.join(args.minc_root, 'labels')
    train_txt = os.path.join(labels_dir, f'train{args.split}.txt')
    val_txt = os.path.join(labels_dir, f'validate{args.split}.txt')
    test_txt = os.path.join(labels_dir, f'test{args.split}.txt')

    if is_main():
        print(f'Train list: {train_txt}')
        print(f'Val list  : {val_txt}')
        print(f'Test list : {test_txt}')

    num_classes = infer_num_classes([train_txt, val_txt, test_txt])
    if is_main():
        print(f'Num classes: {num_classes}')

    train_tf = build_transform(args.img_size, args.strong_aug, is_train=True)
    eval_tf = build_transform(args.img_size, False, is_train=False)

    ds_tr = MincTxtDataset(args.minc_root, train_txt, transform=train_tf, debug_first=args.debug_first)
    ds_va = MincTxtDataset(args.minc_root, val_txt, transform=eval_tf, debug_first=args.debug_first)
    ds_te = MincTxtDataset(args.minc_root, test_txt, transform=eval_tf, debug_first=args.debug_first)

    if args.ddp:
        samp_tr = DistributedSampler(ds_tr, num_replicas=world, rank=rank, shuffle=True, drop_last=True)
        samp_va = DistributedSampler(ds_va, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
        samp_te = DistributedSampler(ds_te, num_replicas=world, rank=rank, shuffle=False, drop_last=False)
    else:
        samp_tr = None
        samp_va = None
        samp_te = None

    loader_tr = DataLoader(ds_tr, batch_size=args.batch, sampler=samp_tr, shuffle=(samp_tr is None),
                           num_workers=args.num_workers, pin_memory=True, drop_last=True,
                           persistent_workers=(args.persistent_workers == 1 and args.num_workers > 0),
                           timeout=args.dl_timeout,
                           prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None))
    loader_va = DataLoader(ds_va, batch_size=args.batch, sampler=samp_va, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False,
                           persistent_workers=(args.persistent_workers == 1 and args.num_workers > 0),
                           timeout=args.dl_timeout,
                           prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None))
    loader_te = DataLoader(ds_te, batch_size=args.batch, sampler=samp_te, shuffle=False,
                           num_workers=args.num_workers, pin_memory=True, drop_last=False,
                           persistent_workers=(args.persistent_workers == 1 and args.num_workers > 0),
                           timeout=args.dl_timeout,
                           prefetch_factor=(args.prefetch_factor if args.num_workers > 0 else None))

    model = timm.create_model(args.arch, pretrained=bool(args.pretrained), num_classes=num_classes).to(device)
    if args.weights_path:
        load_ckpt_drop_head(model, args.weights_path)

    # Mixup/Cutmix
    mixup_fn = None
    if (args.mixup > 0.0 or args.cutmix > 0.0) and args.mixup_prob > 0.0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup,
            cutmix_alpha=args.cutmix if args.cutmix > 0 else 0.0,
            prob=args.mixup_prob,
            switch_prob=args.mixup_switch_prob,
            mode=args.mixup_mode,
            label_smoothing=0.0,
            num_classes=num_classes,
        )

    # correct criterion
    criterion = SoftTargetCrossEntropy() if mixup_fn is not None else nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    model_ema = ModelEmaV2(model, decay=args.ema_decay, device=device) if args.ema else None

    if args.ddp:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)
        if is_main():
            print('[INFO] Using DistributedDataParallel (DDP)')

    run_name = args.save_name.strip() or f'{args.arch}_split{args.split}_{now_str()}'
    out_dir = os.path.join(args.out_dir, run_name)
    if is_main():
        os.makedirs(out_dir, exist_ok=True)

    best_path = os.path.join(out_dir, 'best.pt')
    last_path = os.path.join(out_dir, 'last.pt')
    best_acc = -1.0

    try:
        for epoch in range(1, args.epochs + 1):
            t0 = time.time()
            if args.ddp:
                samp_tr.set_epoch(epoch)

            train_loss = train_one_epoch(model, loader_tr, optimizer, device, scaler, criterion, args.amp, mixup_fn)

            if model_ema is not None:
                m = model.module if hasattr(model, 'module') else model
                model_ema.update(m)

            eval_model = model_ema.module if model_ema is not None else (model.module if hasattr(model, 'module') else model)
            val_acc, val_f1 = evaluate(eval_model, loader_va, device, amp=args.amp, num_classes=num_classes)

            if is_main():
                print(f'[{epoch}] loss={train_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={time.time()-t0:.1f}s', flush=True)

            if is_main():
                state = {
                    'epoch': epoch,
                    'model': eval_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scaler': scaler.state_dict() if args.amp else None,
                    'args': vars(args),
                    'best_acc': best_acc,
                }
                torch.save(state, last_path)
                if val_acc > best_acc:
                    best_acc = val_acc
                    state['best_acc'] = best_acc
                    torch.save(state, best_path)

            barrier()

        # rank0 final test
        if is_main():
            ck = torch.load(best_path, map_location='cpu')
            m = timm.create_model(args.arch, pretrained=False, num_classes=num_classes)
            m.load_state_dict(ck['model'], strict=True)
            m.to(device)
            test_acc, test_f1 = evaluate(m, loader_te, device, amp=args.amp, num_classes=num_classes)
            print(f'[TEST] acc={test_acc:.4f} f1={test_f1:.4f} (best_acc={ck.get("best_acc", -1):.4f})')

        barrier()
    finally:
        cleanup_ddp()


if __name__ == '__main__':
    main()
