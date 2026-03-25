import argparse
import json
import math
import os
import random
import re
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from scipy.signal import resample_poly
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.models import resnet18
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("Robot_impact_Data audio trainer (DDP)")
    p.add_argument(
        "--data_root",
        type=str,
        default="/root/mqx/LSMRT/data/YCB-impact-sounds/Robot_impact_Data",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/root/mqx/LSMRT/audio-branch/runs/robot_impact_material_ddp",
    )
    p.add_argument(
        "--subset",
        type=str,
        default="horizontal",
        choices=["horizontal", "vertical", "both"],
        help=(
            "horizontal: Horizontal_Pokes material classes; "
            "vertical: Vertical_Pokes material_id classes; "
            "both: merge both sets with prefixed labels."
        ),
    )
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_sec", type=float, default=1.0)
    p.add_argument("--target_sr", type=int, default=24000)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument(
        "--pretrained_ckpt",
        type=str,
        default="",
        help="Optional checkpoint path for transfer learning (e.g., YCB proxy best.pt).",
    )
    p.add_argument(
        "--pretrained_scope",
        type=str,
        default="backbone_embedding",
        choices=["none", "backbone", "backbone_embedding", "all"],
        help="Which parts to initialize from --pretrained_ckpt.",
    )
    p.add_argument(
        "--class_weight_mode",
        type=str,
        default="inv_sqrt",
        choices=["none", "inv_sqrt", "inv_freq"],
        help="Class weight strategy for CrossEntropyLoss.",
    )
    p.add_argument(
        "--eval_num_crops",
        type=int,
        default=1,
        help="Evenly-spaced temporal crops per clip during val/test; logits are averaged.",
    )
    p.add_argument(
        "--run_tag",
        type=str,
        default="",
        help="Optional suffix appended to run folder name.",
    )
    return p.parse_args()


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank, world_size, local_rank = 0, 1, 0
    return rank, world_size, local_rank


def cleanup_ddp():
    if dist.is_available() and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main(rank: int):
    return rank == 0


def reduce_sum_scalar(v: float, device: torch.device):
    t = torch.tensor([v], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_class_weights(train_records, class_to_idx, mode: str):
    if mode == "none":
        return None, {}

    counts = Counter([r.label_name for r in train_records])
    w = torch.ones(len(class_to_idx), dtype=torch.float32)
    for name, idx in class_to_idx.items():
        c = float(max(1, counts.get(name, 0)))
        if mode == "inv_freq":
            w[idx] = 1.0 / c
        elif mode == "inv_sqrt":
            w[idx] = 1.0 / math.sqrt(c)
        else:
            raise ValueError(f"Unknown class_weight_mode={mode}")

    w = w * (float(len(w)) / float(w.sum().clamp_min(1e-12).item()))
    label_counts = {name: int(counts.get(name, 0)) for name in sorted(class_to_idx.keys())}
    return w, label_counts


def load_pretrained_weights(model: nn.Module, ckpt_path: str, scope: str, rank: int):
    if (not ckpt_path) or scope == "none":
        return

    ckpt_p = Path(ckpt_path)
    if not ckpt_p.exists():
        raise FileNotFoundError(f"pretrained_ckpt not found: {ckpt_p}")

    raw = torch.load(str(ckpt_p), map_location="cpu")
    if isinstance(raw, dict) and "model" in raw:
        src_state = raw["model"]
    elif isinstance(raw, dict):
        src_state = raw
    else:
        raise RuntimeError(f"Unsupported checkpoint format: {type(raw)}")

    def allow_key(k: str):
        if scope == "all":
            return True
        if scope == "backbone":
            return k.startswith("mel.") or k.startswith("backbone.")
        if scope == "backbone_embedding":
            return (
                k.startswith("mel.")
                or k.startswith("backbone.")
                or k.startswith("embedding.")
            )
        raise ValueError(f"Unknown pretrained_scope={scope}")

    dst_state = model.state_dict()
    filtered = {}
    skipped_shape = []
    for k, v in src_state.items():
        if (k not in dst_state) or (not allow_key(k)):
            continue
        if dst_state[k].shape != v.shape:
            skipped_shape.append(k)
            continue
        filtered[k] = v

    missing, unexpected = model.load_state_dict(filtered, strict=False)
    if is_main(rank):
        print(
            f"[pretrain] loaded={len(filtered)} "
            f"skipped_shape={len(skipped_shape)} "
            f"missing_after_load={len(missing)} unexpected={len(unexpected)} "
            f"from={ckpt_p}",
            flush=True,
        )


@dataclass
class ClipRecord:
    path: str
    source: str
    split_group: str
    object_name: str
    label_name: str


def parse_material_id(object_name: str):
    m = re.search(r"_(-?\d+)$", object_name)
    if m is None:
        raise ValueError(f"Cannot parse material id from object_name={object_name}")
    return int(m.group(1))


def collect_vertical_records(root: Path):
    recs = []
    base_root = root / "Vertical_Pokes"
    for split_group in ["Known_Objects", "Unknown_Objects"]:
        base = base_root / split_group
        if not base.exists():
            continue
        for object_dir in sorted([x for x in base.iterdir() if x.is_dir()]):
            mid = parse_material_id(object_dir.name)
            label_name = f"V{mid}"
            clips = sorted(
                list(object_dir.glob("*.ogg"))
                + list(object_dir.glob("*.ogx"))
                + list(object_dir.glob("*.wav"))
                + list(object_dir.glob("*.flac"))
            )
            for c in clips:
                recs.append(
                    ClipRecord(
                        path=str(c),
                        source="vertical",
                        split_group=("known" if split_group == "Known_Objects" else "unknown"),
                        object_name=object_dir.name,
                        label_name=label_name,
                    )
                )
    return recs


def collect_horizontal_records(root: Path):
    recs = []
    base_root = root / "Horizontal_Pokes"
    if not base_root.exists():
        return recs

    for material_dir in sorted([x for x in base_root.iterdir() if x.is_dir()]):
        material_name = material_dir.name
        label_name = f"H_{material_name}"
        for split_group in ["train", "test"]:
            split_root = material_dir / split_group
            if not split_root.exists():
                continue
            for object_dir in sorted([x for x in split_root.iterdir() if x.is_dir()]):
                clips = sorted(
                    list(object_dir.glob("*.ogg"))
                    + list(object_dir.glob("*.ogx"))
                    + list(object_dir.glob("*.wav"))
                    + list(object_dir.glob("*.flac"))
                )
                for c in clips:
                    recs.append(
                        ClipRecord(
                            path=str(c),
                            source="horizontal",
                            split_group=split_group,
                            object_name=f"{material_name}/{object_dir.name}",
                            label_name=label_name,
                        )
                    )
    return recs


def collect_robot_impact_records(data_root: str, subset: str):
    root = Path(data_root)
    if not root.exists():
        raise FileNotFoundError(f"data_root not found: {data_root}")

    recs = []
    if subset in ("horizontal", "both"):
        recs.extend(collect_horizontal_records(root))
    if subset in ("vertical", "both"):
        recs.extend(collect_vertical_records(root))

    if len(recs) == 0:
        raise RuntimeError(
            f"No audio clips (.ogg/.ogx/.wav/.flac) found under {data_root} for subset={subset}"
        )
    return recs


def split_train_val(train_records, val_ratio: float, seed: int):
    by_label = {}
    for r in train_records:
        by_label.setdefault(r.label_name, []).append(r)

    rng = random.Random(seed)
    train, val = [], []
    for _, lst in by_label.items():
        lst = sorted(lst, key=lambda x: x.path)
        rng.shuffle(lst)
        if len(lst) < 2:
            train.extend(lst)
            continue
        n_val = int(round(len(lst) * val_ratio))
        n_val = max(1, min(n_val, len(lst) - 1))
        val.extend(lst[:n_val])
        train.extend(lst[n_val:])
    return train, val


def build_splits(records, subset: str, val_ratio: float, seed: int):
    if subset == "horizontal":
        train_pool = [r for r in records if r.source == "horizontal" and r.split_group == "train"]
        test_records = [r for r in records if r.source == "horizontal" and r.split_group == "test"]
        split_note = "train/val from Horizontal_Pokes/train, test from Horizontal_Pokes/test"
    elif subset == "vertical":
        train_pool = [r for r in records if r.source == "vertical" and r.split_group == "known"]
        test_records = [r for r in records if r.source == "vertical" and r.split_group == "unknown"]
        split_note = "train/val from Vertical_Pokes/Known_Objects, test from Vertical_Pokes/Unknown_Objects"
    else:
        horizontal_train = [
            r for r in records if r.source == "horizontal" and r.split_group == "train"
        ]
        vertical_known = [r for r in records if r.source == "vertical" and r.split_group == "known"]
        train_pool = horizontal_train + vertical_known
        horizontal_test = [r for r in records if r.source == "horizontal" and r.split_group == "test"]
        vertical_unknown = [r for r in records if r.source == "vertical" and r.split_group == "unknown"]
        test_records = horizontal_test + vertical_unknown
        split_note = (
            "train/val from (Horizontal_Pokes/train + Vertical_Pokes/Known_Objects), "
            "test from (Horizontal_Pokes/test + Vertical_Pokes/Unknown_Objects)"
        )

    train_records, val_records = split_train_val(
        train_records=train_pool, val_ratio=val_ratio, seed=seed
    )
    if len(train_records) == 0:
        raise RuntimeError("Empty train split.")
    if len(val_records) == 0:
        raise RuntimeError("Empty val split.")
    if len(test_records) == 0:
        raise RuntimeError("Empty test split.")
    return train_records, val_records, test_records, split_note


class RobotImpactDataset(Dataset):
    def __init__(
        self,
        records,
        class_to_idx,
        target_sr: int,
        clip_sec: float,
        mode: str,
        eval_num_crops: int = 1,
    ):
        self.records = records
        self.class_to_idx = class_to_idx
        self.target_sr = int(target_sr)
        self.clip_frames = int(round(float(clip_sec) * self.target_sr))
        self.mode = mode
        self.eval_num_crops = max(1, int(eval_num_crops))
        self._sf = None

    def __len__(self):
        return len(self.records)

    def _load_audio(self, path: str):
        if self._sf is None:
            try:
                import soundfile as sf  # type: ignore
            except Exception as e:
                raise RuntimeError(
                    "soundfile is required to decode .ogg/.ogx/.wav/.flac in Robot_impact_Data. "
                    "Please install it in lsmrt_py311."
                ) from e
            self._sf = sf

        wav, sr = self._sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)  # mono
        if sr != self.target_sr:
            wav = resample_poly(wav, up=self.target_sr, down=sr).astype(np.float32)
            sr = self.target_sr
        return wav, sr

    def __getitem__(self, idx):
        r = self.records[idx]
        y = self.class_to_idx[r.label_name]

        try:
            wav_np, _ = self._load_audio(r.path)
        except Exception as e:
            raise RuntimeError(f"Failed to decode audio: {r.path}") from e

        wav = torch.from_numpy(wav_np).float()
        n = wav.numel()
        t = self.clip_frames

        if n < t:
            wav = F.pad(wav, (0, t - n))
            n = wav.numel()

        if self.mode == "train":
            if n > t:
                s = random.randint(0, n - t)
                wav = wav[s : s + t]
            gain_db = random.uniform(-6.0, 6.0)
            wav = wav * float(10.0 ** (gain_db / 20.0))
            if random.random() < 0.5:
                wav = -wav
            noise_std = random.uniform(0.0, 0.005)
            if noise_std > 0:
                wav = wav + torch.randn_like(wav) * noise_std
            shift = random.randint(-int(0.01 * t), int(0.01 * t))
            if shift != 0:
                wav = torch.roll(wav, shifts=shift, dims=0)
            wav = wav.clamp(-1.0, 1.0)
            return wav, int(y)

        if self.eval_num_crops <= 1:
            if n > t:
                s = (n - t) // 2
                wav = wav[s : s + t]
            return wav, int(y)

        if n <= t:
            crops = [wav[:t] for _ in range(self.eval_num_crops)]
        else:
            starts = np.linspace(0, n - t, num=self.eval_num_crops, dtype=np.int64).tolist()
            crops = [wav[int(s) : int(s) + t] for s in starts]
        wav = torch.stack(crops, dim=0)
        return wav, int(y)


class AudioBranchNet(nn.Module):
    def __init__(self, num_classes: int, target_sr: int, n_fft: int, hop_length: int, n_mels: int):
        super().__init__()
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            center=True,
            power=2.0,
        )
        self.to_db = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80.0)

        backbone = resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(feat_dim, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, wav):
        x = wav.unsqueeze(1)
        x = self.mel(x).clamp_min(1e-8)
        x = self.to_db(x)
        x = (x + 80.0) / 80.0
        feat = self.backbone(x)
        emb = F.normalize(self.embedding(feat), dim=1)
        logits = self.classifier(emb)
        return logits, emb


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool, num_classes: int = 0):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0.0
    total_num = 0.0
    conf = None
    if num_classes > 0:
        conf = torch.zeros((num_classes, num_classes), dtype=torch.float64, device=device)
    for wav, y in loader:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        if wav.ndim == 3:
            bsz, num_crops, num_frames = wav.shape
            wav_in = wav.reshape(bsz * num_crops, num_frames)
        else:
            bsz = y.size(0)
            num_crops = 1
            wav_in = wav

        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(amp_enabled and device.type == "cuda")
        ):
            logits, _ = model(wav_in)
            if num_crops > 1:
                logits = logits.reshape(bsz, num_crops, -1).mean(dim=1)
            loss = loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * y.size(0)
        total_correct += float((pred == y).sum().item())
        total_num += float(y.size(0))
        if conf is not None:
            idx = (y.to(torch.int64) * num_classes + pred.to(torch.int64)).view(-1)
            bins = torch.bincount(idx, minlength=num_classes * num_classes).to(torch.float64)
            conf += bins.reshape(num_classes, num_classes)
    total_loss = reduce_sum_scalar(total_loss, device)
    total_correct = reduce_sum_scalar(total_correct, device)
    total_num = reduce_sum_scalar(total_num, device)
    if conf is not None and dist.is_available() and dist.is_initialized():
        dist.all_reduce(conf, op=dist.ReduceOp.SUM)
    loss_avg = total_loss / max(1.0, total_num)
    acc_avg = total_correct / max(1.0, total_num)
    if conf is None:
        return loss_avg, acc_avg
    return loss_avg, acc_avg, conf


def main():
    args = parse_args()
    if args.eval_num_crops < 1:
        raise ValueError("--eval_num_crops must be >= 1")
    rank, world_size, local_rank = setup_ddp()
    seed_everything(args.seed + rank)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    records = collect_robot_impact_records(args.data_root, subset=args.subset)
    train_records, val_records, test_records, split_note = build_splits(
        records, subset=args.subset, val_ratio=args.val_ratio, seed=args.seed
    )

    labels = sorted({r.label_name for r in train_records})
    class_to_idx = {name: i for i, name in enumerate(labels)}
    idx_to_label = {i: name for name, i in class_to_idx.items()}
    val_only = sorted({r.label_name for r in val_records} - set(labels))
    test_only = sorted({r.label_name for r in test_records} - set(labels))
    if val_only or test_only:
        raise RuntimeError(
            f"Found labels not present in train split. val_only={val_only}, test_only={test_only}"
        )
    num_classes = len(class_to_idx)

    train_ds = RobotImpactDataset(
        train_records,
        class_to_idx,
        args.target_sr,
        args.clip_sec,
        mode="train",
        eval_num_crops=1,
    )
    val_ds = RobotImpactDataset(
        val_records,
        class_to_idx,
        args.target_sr,
        args.clip_sec,
        mode="val",
        eval_num_crops=args.eval_num_crops,
    )
    test_ds = RobotImpactDataset(
        test_records,
        class_to_idx,
        args.target_sr,
        args.clip_sec,
        mode="test",
        eval_num_crops=args.eval_num_crops,
    )

    train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False)
    val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)
    test_sampler = DistributedSampler(test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False)

    loader_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        loader_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, sampler=train_sampler, **loader_kwargs)
    val_loader = DataLoader(val_ds, sampler=val_sampler, **loader_kwargs)
    test_loader = DataLoader(test_ds, sampler=test_sampler, **loader_kwargs)

    model = AudioBranchNet(
        num_classes=num_classes,
        target_sr=args.target_sr,
        n_fft=args.n_fft,
        hop_length=args.hop_length,
        n_mels=args.n_mels,
    ).to(device)
    load_pretrained_weights(
        model=model,
        ckpt_path=args.pretrained_ckpt,
        scope=args.pretrained_scope,
        rank=rank,
    )
    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    class_weights_cpu, label_counts = build_class_weights(
        train_records=train_records, class_to_idx=class_to_idx, mode=args.class_weight_mode
    )
    if is_main(rank):
        print(f"[split] label_counts={json.dumps(label_counts, ensure_ascii=False)}", flush=True)
        if class_weights_cpu is not None:
            w_map = {name: float(class_weights_cpu[idx].item()) for name, idx in class_to_idx.items()}
            print(f"[loss] class_weights={json.dumps(w_map, ensure_ascii=False)}", flush=True)
    class_weights = class_weights_cpu.to(device) if class_weights_cpu is not None else None
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing, weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    amp_enabled = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_name = f"robot_impact_sr{args.target_sr}_clip{args.clip_sec}_b{args.batch}x{world_size}_e{args.epochs}"
    run_name = f"{args.subset}_{run_name}"
    if args.run_tag:
        run_name = f"{run_name}_{args.run_tag}"
    out_dir = Path(args.out_dir) / run_name
    if is_main(rank):
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            str(out_dir / "config.json"),
            {
                **vars(args),
                "world_size": world_size,
                "num_classes": num_classes,
                "num_total": len(records),
                "num_train": len(train_records),
                "num_val": len(val_records),
                "num_test": len(test_records),
            },
        )
        save_json(str(out_dir / "class_index.json"), {str(i): name for i, name in idx_to_label.items()})

    best_val = -1.0
    best_ckpt = out_dir / "best.pt"
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()
        t0 = time.time()
        epoch_loss, epoch_correct, epoch_num = 0.0, 0.0, 0.0

        itr = train_loader
        if is_main(rank):
            itr = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)

        for wav, y in itr:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            if not torch.isfinite(wav).all():
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=amp_enabled):
                logits, _ = model(wav)
                loss = criterion(logits, y)
            if not torch.isfinite(loss):
                continue

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            pred = torch.argmax(logits, dim=1)
            bs = y.size(0)
            epoch_loss += float(loss.item()) * bs
            epoch_correct += float((pred == y).sum().item())
            epoch_num += float(bs)

        scheduler.step()

        sum_loss = reduce_sum_scalar(epoch_loss, device)
        sum_corr = reduce_sum_scalar(epoch_correct, device)
        sum_num = reduce_sum_scalar(epoch_num, device)
        train_loss = sum_loss / max(1.0, sum_num)
        train_acc = sum_corr / max(1.0, sum_num)
        val_loss, val_acc = evaluate(
            model, val_loader, device, amp_enabled=amp_enabled, num_classes=0
        )
        dt = time.time() - t0

        if is_main(rank):
            lr_now = float(optimizer.param_groups[0]["lr"])
            print(
                f"[{epoch:03d}] train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr_now:.2e} time={dt:.1f}s",
                flush=True,
            )
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "epoch": epoch,
                            "train_loss": train_loss,
                            "train_acc": train_acc,
                            "val_loss": val_loss,
                            "val_acc": val_acc,
                            "lr": lr_now,
                            "time_sec": dt,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

            if val_acc > best_val:
                best_val = val_acc
                state_dict = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save(
                    {
                        "model": state_dict,
                        "best_val_acc": best_val,
                        "class_to_idx": class_to_idx,
                        "idx_to_label": idx_to_label,
                        "target_sr": args.target_sr,
                        "clip_sec": args.clip_sec,
                        "n_fft": args.n_fft,
                        "hop_length": args.hop_length,
                        "n_mels": args.n_mels,
                    },
                    best_ckpt,
                )

    if is_main(rank):
        print(f"Best ckpt: {best_ckpt}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if best_ckpt.exists():
        ckpt = torch.load(best_ckpt, map_location=device)
        state_dict = ckpt["model"]
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    test_loss, test_acc, test_conf = evaluate(
        model, test_loader, device, amp_enabled=amp_enabled, num_classes=num_classes
    )

    if is_main(rank):
        per_class_acc = {}
        for i in range(num_classes):
            label_name = idx_to_label[i]
            row = test_conf[i]
            row_sum = float(row.sum().item())
            diag = float(row[i].item())
            per_class_acc[label_name] = (diag / row_sum) if row_sum > 0 else None
        valid_class_acc = [x for x in per_class_acc.values() if x is not None]
        test_macro_acc = float(sum(valid_class_acc) / max(1, len(valid_class_acc)))

        metrics = {
            "best_val_acc": best_val,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_acc": test_macro_acc,
            "test_per_class_acc": per_class_acc,
            "eval_num_crops": args.eval_num_crops,
            "class_weight_mode": args.class_weight_mode,
            "pretrained_ckpt": args.pretrained_ckpt,
            "pretrained_scope": args.pretrained_scope,
            "split_note": split_note,
            "class_note": (
                "horizontal labels are material names prefixed with H_; "
                "vertical labels are material_id parsed from object folder suffix and prefixed with V."
            ),
        }
        save_json(str(out_dir / "metrics.json"), metrics)
        print("==== DONE ====", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
