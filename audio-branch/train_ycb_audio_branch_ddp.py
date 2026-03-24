import argparse
import json
import os
import random
import time
import wave
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler
from torchvision.models import resnet18
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser("YCB-impact sounds audio-branch trainer (DDP)")
    p.add_argument(
        "--wav_root",
        type=str,
        default="/root/mqx/LSMRT/data/YCB-impact-sounds/raw_audio/extracted/Audio",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="/root/mqx/LSMRT/audio-branch/runs/ycb_audio_proxy_ddp",
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch", type=int, default=32, help="Per-GPU batch size")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--clip_sec", type=float, default=1.0)
    p.add_argument("--target_sr", type=int, default=24000)
    p.add_argument("--n_fft", type=int, default=1024)
    p.add_argument("--hop_length", type=int, default=256)
    p.add_argument("--n_mels", type=int, default=128)
    p.add_argument("--train_clips_per_file", type=int, default=24)
    p.add_argument("--eval_clips_per_file", type=int, default=8)
    p.add_argument("--split_train", type=float, default=0.70)
    p.add_argument("--split_val", type=float, default=0.15)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--grad_clip", type=float, default=2.0)
    p.add_argument("--no_amp", action="store_true", help="Disable AMP")
    return p.parse_args()


def setup_ddp():
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl", init_method="env://")
    else:
        rank = 0
        world_size = 1
        local_rank = 0
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


def is_main_process(rank: int):
    return rank == 0


def list_wavs(wav_root: str):
    root = Path(wav_root)
    if not root.exists():
        raise FileNotFoundError(f"WAV root not found: {wav_root}")
    wavs = sorted(root.glob("*.WAV"))
    if len(wavs) == 0:
        wavs = sorted(root.glob("*.wav"))
    if len(wavs) == 0:
        raise RuntimeError(f"No WAV files found in {wav_root}")
    return [str(p) for p in wavs]


def read_wav_meta(path: str):
    with wave.open(path, "rb") as w:
        return {
            "sample_rate": int(w.getframerate()),
            "num_frames": int(w.getnframes()),
            "num_channels": int(w.getnchannels()),
            "sampwidth": int(w.getsampwidth()),
        }


def pcm24_to_int32(raw_bytes: bytes):
    x = np.frombuffer(raw_bytes, dtype=np.uint8)
    if x.size == 0:
        return np.zeros((0,), dtype=np.int32)
    usable = (x.size // 3) * 3
    x = x[:usable].reshape(-1, 3)
    y = (
        x[:, 0].astype(np.int32)
        | (x[:, 1].astype(np.int32) << 8)
        | (x[:, 2].astype(np.int32) << 16)
    )
    sign = y & 0x800000
    y = y - (sign << 1)
    return y.astype(np.int32)


def load_wav_segment(path: str, start_frame: int, num_frames: int):
    with wave.open(path, "rb") as w:
        sr = int(w.getframerate())
        n_ch = int(w.getnchannels())
        sampwidth = int(w.getsampwidth())
        total = int(w.getnframes())

        if total <= 0:
            return np.zeros((num_frames,), dtype=np.float32), sr

        max_start = max(0, total - num_frames)
        start_frame = int(max(0, min(start_frame, max_start)))
        w.setpos(start_frame)
        raw = w.readframes(num_frames)

    if sampwidth == 2:
        pcm = np.frombuffer(raw, dtype="<i2").astype(np.int32)
        denom = 32768.0
    elif sampwidth == 3:
        pcm = pcm24_to_int32(raw)
        denom = 8388608.0
    elif sampwidth == 4:
        pcm = np.frombuffer(raw, dtype="<i4").astype(np.int32)
        denom = 2147483648.0
    else:
        raise RuntimeError(f"Unsupported sample width={sampwidth} for {path}")

    if n_ch > 1:
        usable = (pcm.size // n_ch) * n_ch
        pcm = pcm[:usable].reshape(-1, n_ch).mean(axis=1)
    else:
        pcm = pcm.reshape(-1)

    wav = pcm.astype(np.float32) / float(denom)

    if wav.shape[0] < num_frames:
        wav = np.pad(wav, (0, num_frames - wav.shape[0]), mode="constant")
    elif wav.shape[0] > num_frames:
        wav = wav[:num_frames]
    return wav, sr


class YCBProxyDataset(Dataset):
    def __init__(
        self,
        wav_paths,
        mode: str,
        clip_sec: float,
        target_sr: int,
        split_train: float,
        split_val: float,
        clips_per_file: int,
    ):
        assert mode in {"train", "val", "test"}
        self.wav_paths = wav_paths
        self.mode = mode
        self.clip_sec = float(clip_sec)
        self.target_sr = int(target_sr)
        self.clips_per_file = int(clips_per_file)
        self.split_train = float(split_train)
        self.split_val = float(split_val)

        self.class_names = [Path(p).stem for p in self.wav_paths]
        self.class_to_idx = {c: i for i, c in enumerate(self.class_names)}
        self.metas = [read_wav_meta(p) for p in self.wav_paths]

        self.frame_ranges = []
        for m in self.metas:
            total = m["num_frames"]
            train_end = int(total * self.split_train)
            val_end = int(total * (self.split_train + self.split_val))
            if self.mode == "train":
                lo, hi = 0, max(1, train_end)
            elif self.mode == "val":
                lo, hi = train_end, max(train_end + 1, val_end)
            else:
                lo, hi = val_end, max(val_end + 1, total)
            self.frame_ranges.append((int(lo), int(hi)))

    def __len__(self):
        return len(self.wav_paths) * self.clips_per_file

    def _choose_start(self, file_idx: int, clip_idx_for_file: int):
        meta = self.metas[file_idx]
        lo, hi = self.frame_ranges[file_idx]
        clip_frames = max(1, int(round(self.clip_sec * meta["sample_rate"])))

        if hi <= lo:
            lo, hi = 0, max(1, meta["num_frames"])

        if (hi - lo) <= clip_frames:
            return lo, clip_frames

        max_start = hi - clip_frames
        if self.mode == "train":
            start = random.randint(lo, max_start)
        else:
            if self.clips_per_file <= 1:
                start = lo
            else:
                alpha = clip_idx_for_file / float(self.clips_per_file - 1)
                start = lo + int(round((max_start - lo) * alpha))
        return int(start), int(clip_frames)

    def __getitem__(self, idx):
        file_idx = idx % len(self.wav_paths)
        clip_idx_for_file = idx // len(self.wav_paths)

        path = self.wav_paths[file_idx]
        y = file_idx
        start, clip_frames = self._choose_start(file_idx, clip_idx_for_file)
        wav_np, orig_sr = load_wav_segment(path, start, clip_frames)

        wav = torch.from_numpy(wav_np).float().unsqueeze(0)
        if orig_sr != self.target_sr:
            wav = torchaudio.functional.resample(
                wav, orig_freq=orig_sr, new_freq=self.target_sr
            )

        target_frames = int(round(self.clip_sec * self.target_sr))
        if wav.shape[-1] < target_frames:
            wav = F.pad(wav, (0, target_frames - wav.shape[-1]))
        elif wav.shape[-1] > target_frames:
            wav = wav[:, :target_frames]

        if self.mode == "train":
            gain_db = random.uniform(-6.0, 6.0)
            wav = wav * float(10.0 ** (gain_db / 20.0))
            if random.random() < 0.5:
                wav = -wav
            noise_std = random.uniform(0.0, 0.005)
            if noise_std > 0:
                wav = wav + torch.randn_like(wav) * noise_std
            wav = wav.clamp(-1.0, 1.0)

        return wav.squeeze(0), int(y)


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
        backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.embedding = nn.Linear(feat_dim, 256)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, wav):
        # wav: (B, T)
        x = wav.unsqueeze(1)  # (B, 1, T)
        x = self.mel(x).clamp_min(1e-8)  # (B, 1, M, TT)
        x = self.to_db(x)
        x = (x + 80.0) / 80.0
        feat = self.backbone(x)
        emb = F.normalize(self.embedding(feat), dim=1)
        logits = self.classifier(emb)
        return logits, emb


def reduce_sum_scalar(value: float, device: torch.device):
    t = torch.tensor([value], dtype=torch.float64, device=device)
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
    return float(t.item())


@torch.no_grad()
def evaluate(model, loader, device, amp_enabled: bool):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    total_correct = 0.0
    total_num = 0.0

    for wav, y in loader:
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(
            device_type="cuda", dtype=torch.float16, enabled=(amp_enabled and device.type == "cuda")
        ):
            logits, _ = model(wav)
            loss = loss_fn(logits, y)
        pred = torch.argmax(logits, dim=1)
        total_loss += float(loss.item()) * y.size(0)
        total_correct += float((pred == y).sum().item())
        total_num += float(y.size(0))

    total_loss = reduce_sum_scalar(total_loss, device)
    total_correct = reduce_sum_scalar(total_correct, device)
    total_num = reduce_sum_scalar(total_num, device)
    return total_loss / max(1.0, total_num), total_correct / max(1.0, total_num)


def save_json(path: str, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    rank, world_size, local_rank = setup_ddp()
    is_main = is_main_process(rank)

    seed_everything(args.seed + rank)

    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")

    wav_paths = list_wavs(args.wav_root)
    num_classes = len(wav_paths)

    train_ds = YCBProxyDataset(
        wav_paths=wav_paths,
        mode="train",
        clip_sec=args.clip_sec,
        target_sr=args.target_sr,
        split_train=args.split_train,
        split_val=args.split_val,
        clips_per_file=args.train_clips_per_file,
    )
    val_ds = YCBProxyDataset(
        wav_paths=wav_paths,
        mode="val",
        clip_sec=args.clip_sec,
        target_sr=args.target_sr,
        split_train=args.split_train,
        split_val=args.split_val,
        clips_per_file=args.eval_clips_per_file,
    )
    test_ds = YCBProxyDataset(
        wav_paths=wav_paths,
        mode="test",
        clip_sec=args.clip_sec,
        target_sr=args.target_sr,
        split_train=args.split_train,
        split_val=args.split_val,
        clips_per_file=args.eval_clips_per_file,
    )

    train_sampler = DistributedSampler(
        train_ds, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )
    test_sampler = DistributedSampler(
        test_ds, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

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

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], output_device=local_rank)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_enabled = (not args.no_amp) and (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    run_name = (
        f"ycb_proxy_sr{args.target_sr}_clip{args.clip_sec}_"
        f"b{args.batch}x{world_size}_e{args.epochs}"
    )
    out_dir = Path(args.out_dir) / run_name
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
        save_json(
            str(out_dir / "config.json"),
            {
                **vars(args),
                "world_size": world_size,
                "num_classes": num_classes,
                "num_wavs": len(wav_paths),
            },
        )
        save_json(
            str(out_dir / "class_index.json"),
            {i: Path(p).stem for i, p in enumerate(wav_paths)},
        )

    best_val = -1.0
    best_ckpt_path = out_dir / "best.pt"
    log_path = out_dir / "train_log.jsonl"

    for epoch in range(1, args.epochs + 1):
        train_sampler.set_epoch(epoch)
        model.train()

        epoch_loss = 0.0
        epoch_correct = 0.0
        epoch_num = 0.0

        t0 = time.time()
        iterable = train_loader
        if is_main:
            iterable = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", ncols=100)

        for wav, y in iterable:
            wav = wav.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

             # Skip abnormal batches to keep long training stable.
            if not torch.isfinite(wav).all():
                continue

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=amp_enabled,
            ):
                logits, _ = model(wav)
                loss = criterion(logits, y)

            if not torch.isfinite(loss):
                if is_main:
                    print(f"[warn] non-finite loss at epoch={epoch}, skip batch", flush=True)
                optimizer.zero_grad(set_to_none=True)
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

        train_loss = reduce_sum_scalar(epoch_loss, device) / max(1.0, reduce_sum_scalar(epoch_num, device))
        train_acc = reduce_sum_scalar(epoch_correct, device) / max(1.0, reduce_sum_scalar(epoch_num, device))
        val_loss, val_acc = evaluate(model, val_loader, device, amp_enabled=amp_enabled)
        dt = time.time() - t0

        if is_main:
            lr_now = float(optimizer.param_groups[0]["lr"])
            msg = (
                f"[{epoch:03d}] "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} "
                f"lr={lr_now:.2e} time={dt:.1f}s"
            )
            print(msg, flush=True)
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
                        "num_classes": num_classes,
                        "target_sr": args.target_sr,
                        "clip_sec": args.clip_sec,
                        "n_fft": args.n_fft,
                        "hop_length": args.hop_length,
                        "n_mels": args.n_mels,
                        "class_names": [Path(p).stem for p in wav_paths],
                    },
                    best_ckpt_path,
                )

    if is_main:
        print(f"Best ckpt: {best_ckpt_path}", flush=True)

    if dist.is_available() and dist.is_initialized():
        dist.barrier()

    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location=device)
        state_dict = ckpt["model"]
        if isinstance(model, DDP):
            model.module.load_state_dict(state_dict)
        else:
            model.load_state_dict(state_dict)

    # All ranks must participate in distributed reduction inside evaluate().
    test_loss, test_acc = evaluate(model, test_loader, device, amp_enabled=amp_enabled)

    if is_main:
        metrics = {
            "best_val_acc": best_val,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "note": (
                "Proxy pretraining on YCB recording IDs (ZOOM files). "
                "Use saved backbone embedding for downstream material adaptation."
            ),
        }
        save_json(str(out_dir / "metrics.json"), metrics)
        print("==== DONE ====", flush=True)
        print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)

    cleanup_ddp()


if __name__ == "__main__":
    main()
