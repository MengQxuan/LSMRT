#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import resample_poly
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from torchvision.models import efficientnet_b0, resnet18


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_path(p: str, root: Path) -> str:
    pp = Path(p)
    if pp.is_absolute():
        return str(pp)
    return str((root / pp).resolve())


def center_crop_or_pad(wav: torch.Tensor, target_len: int) -> torch.Tensor:
    n = int(wav.numel())
    if n < target_len:
        wav = F.pad(wav, (0, target_len - n))
        return wav
    if n == target_len:
        return wav
    start = (n - target_len) // 2
    return wav[start : start + target_len]


def parse_label_list(s: str):
    if not s:
        return []
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def unique_params(params):
    out = []
    seen = set()
    for p in params:
        if id(p) in seen:
            continue
        out.append(p)
        seen.add(id(p))
    return out


class PairedImageAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, target_sr: int, clip_sec: float):
        self.df = df.reset_index(drop=True)
        self.target_sr = int(target_sr)
        self.clip_frames = int(round(float(clip_sec) * float(self.target_sr)))
        self._sf = None

        self.img_tf = transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.df)

    def _load_audio(self, path: str):
        if self._sf is None:
            try:
                import soundfile as sf  # type: ignore
            except Exception as e:
                raise RuntimeError("soundfile is required for reading .wav files.") from e
            self._sf = sf

        wav, sr = self._sf.read(path, dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
        if sr != self.target_sr:
            wav = resample_poly(wav, up=self.target_sr, down=sr).astype(np.float32)
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        x_img = self.img_tf(img)

        wav_np = self._load_audio(row["audio_path"])
        wav = torch.from_numpy(wav_np).float()
        wav = center_crop_or_pad(wav, self.clip_frames)

        y = int(row["label_id"])
        sid = row["sample_id"]
        return x_img, wav, y, sid, int(idx)


class ImageEffB0Backbone(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

        if "classifier.1.weight" not in sd:
            raise RuntimeError("Unexpected image ckpt format: missing classifier.1.weight")

        num_classes = int(sd["classifier.1.weight"].shape[0])
        in_features = int(sd["classifier.1.weight"].shape[1])

        m = efficientnet_b0(weights=None)
        if m.classifier[-1].in_features != in_features:
            raise RuntimeError("Image checkpoint classifier input dim mismatch")

        m.classifier[-1] = nn.Linear(in_features, num_classes)
        m.load_state_dict(sd, strict=True)

        self.features = m.features
        self.avgpool = m.avgpool
        self.out_dim = in_features

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


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

    def forward_embedding(self, wav):
        x = wav.unsqueeze(1)
        x = self.mel(x).clamp_min(1e-8)
        x = self.to_db(x)
        x = (x + 80.0) / 80.0
        feat = self.backbone(x)
        emb = F.normalize(self.embedding(feat), dim=1)
        return emb

    def forward(self, wav):
        emb = self.forward_embedding(wav)
        logits = self.classifier(emb)
        return logits, emb


class AudioBackboneFromCkpt(nn.Module):
    def __init__(
        self,
        ckpt_path: str,
        target_sr_override: int = 0,
        n_fft_override: int = 0,
        hop_length_override: int = 0,
        n_mels_override: int = 0,
    ):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

        if "embedding.weight" not in sd or "classifier.weight" not in sd:
            raise RuntimeError("Unexpected audio ckpt format")

        num_classes = int(sd["classifier.weight"].shape[0])
        target_sr = int(target_sr_override or (ck.get("target_sr", 24000) if isinstance(ck, dict) else 24000))
        n_fft = int(n_fft_override or (ck.get("n_fft", 1024) if isinstance(ck, dict) else 1024))
        hop_length = int(
            hop_length_override or (ck.get("hop_length", 256) if isinstance(ck, dict) else 256)
        )
        n_mels = int(n_mels_override or (ck.get("n_mels", 128) if isinstance(ck, dict) else 128))

        self.model = AudioBranchNet(
            num_classes=num_classes,
            target_sr=target_sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
        )
        self.model.load_state_dict(sd, strict=True)
        self.out_dim = int(sd["embedding.weight"].shape[0])

    def forward(self, wav):
        return self.model.forward_embedding(wav)


class FusionHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int, hidden_dim: int = 512, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class FusionModel(nn.Module):
    def __init__(self, img_backbone: nn.Module, audio_backbone: nn.Module, num_classes: int, hidden_dim: int, dropout: float):
        super().__init__()
        self.img_backbone = img_backbone
        self.audio_backbone = audio_backbone
        self.in_dim = int(img_backbone.out_dim + audio_backbone.out_dim)
        self.head = FusionHead(self.in_dim, num_classes=num_classes, hidden_dim=hidden_dim, dropout=dropout)

    def forward(self, x_img, x_audio):
        f_img = self.img_backbone(x_img)
        f_audio = self.audio_backbone(x_audio)
        f_img = F.normalize(f_img, p=2, dim=1)
        f_audio = F.normalize(f_audio, p=2, dim=1)
        feat = torch.cat([f_img, f_audio], dim=1)
        logits = self.head(feat)
        return logits, feat


def eval_from_logits(y_true: np.ndarray, logits: torch.Tensor):
    prob = torch.softmax(logits, dim=1).numpy()
    pred = np.argmax(prob, axis=1)
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred, average="macro"))
    cm = confusion_matrix(y_true, pred)
    conf = prob.max(axis=1)
    return acc, f1, cm, pred, conf


@torch.no_grad()
def infer_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    logits_all, y_all, sid_all, idx_all = [], [], [], []
    for x_img, x_audio, y, sid, idx in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_audio = x_audio.to(device, non_blocking=True)
        logits, _ = model(x_img, x_audio)
        logits_all.append(logits.cpu())
        y_all.append(torch.as_tensor(y, dtype=torch.long))
        sid_all.extend(list(sid))
        idx_all.append(torch.as_tensor(idx, dtype=torch.long))

    logits = torch.cat(logits_all, dim=0)
    y = torch.cat(y_all, dim=0)
    idx = torch.cat(idx_all, dim=0)
    return logits, y, sid_all, idx


@torch.no_grad()
def compute_sample_losses(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    crit = nn.CrossEntropyLoss(reduction="none")
    losses = np.zeros(len(loader.dataset), dtype=np.float32)

    for x_img, x_audio, y, sid, idx in loader:
        x_img = x_img.to(device, non_blocking=True)
        x_audio = x_audio.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits, _ = model(x_img, x_audio)
        loss = crit(logits, y)
        losses[idx.numpy()] = loss.detach().cpu().numpy()

    return losses


def build_class_weights(
    df_train: pd.DataFrame,
    num_classes: int,
    idx_to_label: dict,
    mode: str,
    focus_labels,
    focus_boost: float,
):
    counts = np.zeros(num_classes, dtype=np.float64)
    for y in df_train["label_id"].astype(int).tolist():
        counts[int(y)] += 1.0

    weights = np.ones(num_classes, dtype=np.float64)
    if mode == "inv_sqrt":
        weights = 1.0 / np.sqrt(np.maximum(counts, 1.0))
    elif mode == "inv_freq":
        weights = 1.0 / np.maximum(counts, 1.0)
    elif mode == "none":
        weights = np.ones(num_classes, dtype=np.float64)
    else:
        raise ValueError(f"Unknown class_weight_mode={mode}")

    label_to_idx = {str(v).lower(): int(k) for k, v in idx_to_label.items()}
    boosted = []
    for lb in focus_labels:
        if lb in label_to_idx:
            yi = label_to_idx[lb]
            weights[yi] *= float(focus_boost)
            boosted.append(lb)

    weights = weights * (float(len(weights)) / float(np.sum(weights)))
    weight_tensor = torch.tensor(weights, dtype=torch.float32)

    count_map = {idx_to_label[i]: int(counts[i]) for i in range(num_classes)}
    weight_map = {idx_to_label[i]: float(weights[i]) for i in range(num_classes)}
    return weight_tensor, count_map, weight_map, boosted


def configure_trainable_layers(model_core: FusionModel, finetune_last: bool):
    for p in model_core.parameters():
        p.requires_grad = False

    for p in model_core.head.parameters():
        p.requires_grad = True

    ft_params = []
    if finetune_last:
        # Image branch: last EfficientNet feature stage.
        for p in model_core.img_backbone.features[-1].parameters():
            p.requires_grad = True
            ft_params.append(p)

        # Audio branch: last ResNet stage + embedding layer.
        for p in model_core.audio_backbone.model.backbone.layer4.parameters():
            p.requires_grad = True
            ft_params.append(p)
        for p in model_core.audio_backbone.model.embedding.parameters():
            p.requires_grad = True
            ft_params.append(p)

    head_params = [p for p in model_core.head.parameters() if p.requires_grad]
    ft_params = [p for p in ft_params if p.requires_grad]

    head_params = unique_params(head_params)
    ft_params = unique_params(ft_params)

    total_trainable = sum(p.numel() for p in model_core.parameters() if p.requires_grad)
    return head_params, ft_params, total_trainable


def train_model(
    model: nn.Module,
    model_core: FusionModel,
    ds_train: PairedImageAudioDataset,
    ds_val: PairedImageAudioDataset,
    df_train: pd.DataFrame,
    num_classes: int,
    class_weights: torch.Tensor,
    head_params,
    ft_params,
    args,
    device: torch.device,
    use_cuda: bool,
):
    pin_memory = bool(use_cuda)
    persistent_workers = bool(args.num_workers > 0)

    dl_train_eval = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
    )

    param_groups = [{"params": head_params, "lr": args.lr}]
    if ft_params:
        param_groups.append({"params": ft_params, "lr": args.lr_ft})

    opt = torch.optim.AdamW(param_groups, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(
        weight=class_weights.to(device),
        label_smoothing=args.label_smoothing,
    )

    label_ids = df_train["label_id"].astype(int).to_numpy()
    class_w_np = class_weights.cpu().numpy()
    hard_multipliers = np.ones(len(ds_train), dtype=np.float32)

    use_amp = bool(args.amp and use_cuda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best = {"epoch": -1, "val_acc": -1.0, "val_f1": -1.0, "state": None}
    wait = 0
    history = []

    for ep in range(1, args.epochs + 1):
        sample_weights = class_w_np[label_ids].astype(np.float32)
        if args.hard_ratio > 0 and args.hard_boost > 1.0:
            sample_weights = sample_weights * hard_multipliers

        sampler = WeightedRandomSampler(
            weights=torch.from_numpy(sample_weights).double(),
            num_samples=len(sample_weights),
            replacement=True,
        )

        dl_train = DataLoader(
            ds_train,
            batch_size=args.batch,
            sampler=sampler,
            num_workers=args.num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
        )

        model.train()
        train_losses = []

        for x_img, x_audio, y, sid, idx in dl_train:
            x_img = x_img.to(device, non_blocking=True)
            x_audio = x_audio.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits, _ = model(x_img, x_audio)
                loss = crit(logits, y)

            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()

            train_losses.append(float(loss.item()))

        val_logits, val_y, _, _ = infer_loader(model, dl_val, device)
        val_acc, val_f1, _, _, _ = eval_from_logits(val_y.numpy(), val_logits)

        hist_row = {
            "epoch": ep,
            "train_loss": float(np.mean(train_losses)) if train_losses else 0.0,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }

        if args.hard_ratio > 0 and args.hard_boost > 1.0:
            sample_losses = compute_sample_losses(model, dl_train_eval, device)
            k = int(round(len(sample_losses) * args.hard_ratio))
            k = max(1, min(len(sample_losses), k))
            hard_idx = np.argpartition(sample_losses, -k)[-k:]
            hard_multipliers[:] = 1.0
            hard_multipliers[hard_idx] = float(args.hard_boost)
            hist_row["hard_k"] = int(k)
            hist_row["hard_loss_thr"] = float(np.min(sample_losses[hard_idx]))

        history.append(hist_row)

        improved = (val_f1 > best["val_f1"]) or (
            abs(val_f1 - best["val_f1"]) < 1e-9 and val_acc > best["val_acc"]
        )

        if improved:
            best["epoch"] = ep
            best["val_acc"] = val_acc
            best["val_f1"] = val_f1
            core = model.module if isinstance(model, nn.DataParallel) else model
            best["state"] = copy.deepcopy(core.state_dict())
            wait = 0
        else:
            wait += 1

        print(
            f"[{ep:03d}] loss={hist_row['train_loss']:.4f} val_acc={val_acc:.4f} "
            f"val_f1={val_f1:.4f}",
            flush=True,
        )

        if wait >= args.patience:
            print(f"Early stop at epoch {ep}, best epoch={best['epoch']}", flush=True)
            break

    if best["state"] is None:
        core = model.module if isinstance(model, nn.DataParallel) else model
        best["state"] = copy.deepcopy(core.state_dict())

    model_core.load_state_dict(best["state"])
    return best, history


def parse_args():
    p = argparse.ArgumentParser("Train image+audio fusion with weighted loss + hard oversampling + last-layer finetune")
    p.add_argument("--split_dir", type=Path, default=Path("img-audio-fusion/split_fusiondata_8_1_1"))
    p.add_argument(
        "--image_ckpt",
        type=Path,
        default=Path("image-branch/runs/effb0_s224_split1_resize/best.pt"),
    )
    p.add_argument(
        "--audio_ckpt",
        type=Path,
        default=Path(
            "audio-branch/runs/horizontal_robot_impact_sr24000_clip1.0_b32x2_e50_pretrain_invSqrt_c5/best.pt"
        ),
    )
    p.add_argument("--out_dir", type=Path, default=Path("runs/img_audio_fusion_head"))

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--target_sr", type=int, default=24000)
    p.add_argument("--clip_sec", type=float, default=1.0)
    p.add_argument("--n_fft", type=int, default=0, help="0 means reading from audio checkpoint")
    p.add_argument("--hop_length", type=int, default=0, help="0 means reading from audio checkpoint")
    p.add_argument("--n_mels", type=int, default=0, help="0 means reading from audio checkpoint")

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3, help="Fusion head learning rate")
    p.add_argument("--lr_ft", type=float, default=1e-5, help="Backbone last-layer finetune learning rate")
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--grad_clip", type=float, default=2.0)

    p.add_argument("--class_weight_mode", type=str, default="inv_sqrt", choices=["none", "inv_sqrt", "inv_freq"])
    p.add_argument("--focus_labels", type=str, default="ceramic,steel,woodgrain")
    p.add_argument("--focus_boost", type=float, default=2.0)

    p.add_argument("--hard_ratio", type=float, default=0.25)
    p.add_argument("--hard_boost", type=float, default=3.0)

    p.add_argument("--finetune_last", action="store_true")

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--amp", action="store_true")
    p.add_argument("--use_cuda", action="store_true")
    p.add_argument("--use_data_parallel", action="store_true")
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    set_seed(args.seed)

    root = Path.cwd()
    split_dir = (root / args.split_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    val_csv = split_dir / "val.csv"
    test_csv = split_dir / "test.csv"
    if not train_csv.exists() or not val_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"train.csv/val.csv/test.csv not found in {split_dir}")

    df_train = pd.read_csv(train_csv)
    df_val = pd.read_csv(val_csv)
    df_test = pd.read_csv(test_csv)

    req_cols = ["sample_id", "label", "label_id", "image_path", "audio_path"]
    for c in req_cols:
        if c not in df_train.columns or c not in df_val.columns or c not in df_test.columns:
            raise RuntimeError(f"Missing required column: {c}")

    for df in (df_train, df_val, df_test):
        df["image_path"] = df["image_path"].astype(str).apply(lambda x: resolve_path(x, root))
        df["audio_path"] = df["audio_path"].astype(str).apply(lambda x: resolve_path(x, root))

    tr_ids = set(df_train["sample_id"].tolist())
    va_ids = set(df_val["sample_id"].tolist())
    te_ids = set(df_test["sample_id"].tolist())
    assert tr_ids.isdisjoint(va_ids)
    assert tr_ids.isdisjoint(te_ids)
    assert va_ids.isdisjoint(te_ids)

    num_classes = int(max(df_train["label_id"].max(), df_val["label_id"].max(), df_test["label_id"].max()) + 1)
    label_pairs = pd.concat(
        [
            df_train[["label_id", "label"]],
            df_val[["label_id", "label"]],
            df_test[["label_id", "label"]],
        ],
        ignore_index=True,
    ).drop_duplicates()
    idx_to_label = {int(r.label_id): str(r.label) for r in label_pairs.itertuples(index=False)}

    use_cuda = bool(args.use_cuda and torch.cuda.is_available())
    device = torch.device("cuda:0" if use_cuda else "cpu")
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    num_gpu = torch.cuda.device_count() if use_cuda else 0
    use_dp = bool(args.use_data_parallel and use_cuda and num_gpu > 1)

    focus_labels = parse_label_list(args.focus_labels)
    class_weights, class_counts, class_weight_map, boosted_labels = build_class_weights(
        df_train=df_train,
        num_classes=num_classes,
        idx_to_label=idx_to_label,
        mode=args.class_weight_mode,
        focus_labels=focus_labels,
        focus_boost=args.focus_boost,
    )

    print("split_dir:", split_dir, flush=True)
    print("train/val/test:", len(df_train), len(df_val), len(df_test), flush=True)
    print("num_classes:", num_classes, flush=True)
    print("device:", str(device), "num_gpu:", num_gpu, "use_data_parallel:", use_dp, flush=True)
    print("class_weight_mode:", args.class_weight_mode, "focus_labels:", boosted_labels, "focus_boost:", args.focus_boost, flush=True)
    print("hard_ratio:", args.hard_ratio, "hard_boost:", args.hard_boost, flush=True)

    ds_train = PairedImageAudioDataset(df_train, args.img_size, args.target_sr, args.clip_sec)
    ds_val = PairedImageAudioDataset(df_val, args.img_size, args.target_sr, args.clip_sec)
    ds_test = PairedImageAudioDataset(df_test, args.img_size, args.target_sr, args.clip_sec)

    image_ckpt = str((root / args.image_ckpt).resolve())
    audio_ckpt = str((root / args.audio_ckpt).resolve())

    img_backbone = ImageEffB0Backbone(image_ckpt)
    audio_backbone = AudioBackboneFromCkpt(
        audio_ckpt,
        target_sr_override=args.target_sr,
        n_fft_override=args.n_fft,
        hop_length_override=args.hop_length,
        n_mels_override=args.n_mels,
    )

    model_core = FusionModel(
        img_backbone=img_backbone,
        audio_backbone=audio_backbone,
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    )

    head_params, ft_params, total_trainable = configure_trainable_layers(model_core, finetune_last=args.finetune_last)
    print("finetune_last:", args.finetune_last, "trainable_params:", total_trainable, flush=True)

    model_core = model_core.to(device)
    model = nn.DataParallel(model_core) if use_dp else model_core

    best, history = train_model(
        model=model,
        model_core=model_core,
        ds_train=ds_train,
        ds_val=ds_val,
        df_train=df_train,
        num_classes=num_classes,
        class_weights=class_weights,
        head_params=head_params,
        ft_params=ft_params,
        args=args,
        device=device,
        use_cuda=use_cuda,
    )

    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )

    test_logits, y_test, sid_test, _ = infer_loader(model, dl_test, device)
    test_acc, test_f1, test_cm, test_pred, test_conf = eval_from_logits(y_test.numpy(), test_logits)

    best_path = out_dir / "best_head.pt"
    torch.save(
        {
            "model": model_core.state_dict(),
            "input_dim": int(model_core.in_dim),
            "num_classes": num_classes,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "idx_to_label": idx_to_label,
            "image_ckpt": image_ckpt,
            "audio_ckpt": audio_ckpt,
            "target_sr": args.target_sr,
            "clip_sec": args.clip_sec,
            "n_fft": args.n_fft,
            "hop_length": args.hop_length,
            "n_mels": args.n_mels,
            "finetune_last": bool(args.finetune_last),
            "focus_labels": focus_labels,
        },
        best_path,
    )

    pred_rows = []
    df_test_sid = df_test.set_index("sample_id")
    for sid, yt, yp, cf in zip(sid_test, y_test.numpy().tolist(), test_pred.tolist(), test_conf.tolist()):
        r = df_test_sid.loc[sid]
        pred_rows.append(
            {
                "sample_id": sid,
                "label_true": idx_to_label[int(yt)],
                "label_pred": idx_to_label[int(yp)],
                "label_id_true": int(yt),
                "label_id_pred": int(yp),
                "confidence": float(cf),
                "image_path": r["image_path"],
                "audio_path": r["audio_path"],
            }
        )
    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8")

    df_train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    df_val.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    df_test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    metrics = {
        "split_dir": str(split_dir),
        "image_ckpt": image_ckpt,
        "audio_ckpt": audio_ckpt,
        "device": str(device),
        "num_gpu": int(num_gpu),
        "use_data_parallel": bool(use_dp),
        "seed": args.seed,
        "num_classes": num_classes,
        "num_train": int(len(df_train)),
        "num_val": int(len(df_val)),
        "num_test": int(len(df_test)),
        "feature_dim": int(model_core.in_dim),
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "best_val_f1": float(best["val_f1"]),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_confusion_matrix": test_cm.tolist(),
        "no_leakage": {
            "train_vs_val_overlap": int(len(tr_ids & va_ids)),
            "train_vs_test_overlap": int(len(tr_ids & te_ids)),
            "val_vs_test_overlap": int(len(va_ids & te_ids)),
        },
        "class_counts_train": class_counts,
        "class_weights": class_weight_map,
        "class_weight_mode": args.class_weight_mode,
        "focus_labels": focus_labels,
        "focus_labels_effective": boosted_labels,
        "focus_boost": float(args.focus_boost),
        "hard_ratio": float(args.hard_ratio),
        "hard_boost": float(args.hard_boost),
        "finetune_last": bool(args.finetune_last),
        "lr_head": float(args.lr),
        "lr_finetune": float(args.lr_ft),
        "history": history,
        "elapsed_sec": float(time.time() - t0),
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== DONE ===", flush=True)
    print("out_dir:", out_dir, flush=True)
    print("best_head:", best_path, flush=True)
    print(f"test_acc={test_acc:.4f}, test_macro_f1={test_f1:.4f}", flush=True)


if __name__ == "__main__":
    main()
