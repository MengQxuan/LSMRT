#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import random
import time
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
from torch.utils.data import DataLoader, Dataset, TensorDataset
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


def center_crop_or_pad_np(wav: np.ndarray, target_len: int) -> np.ndarray:
    n = int(wav.shape[0])
    if n < target_len:
        out = np.zeros((target_len,), dtype=np.float32)
        out[:n] = wav.astype(np.float32, copy=False)
        return out
    if n == target_len:
        return wav.astype(np.float32, copy=False)
    start = (n - target_len) // 2
    return wav[start : start + target_len].astype(np.float32, copy=False)


class PairedImageAudioDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, target_sr: int, clip_sec: float, cache_audio: bool = True):
        self.df = df.reset_index(drop=True)
        self.target_sr = int(target_sr)
        self.clip_frames = int(round(float(clip_sec) * float(self.target_sr)))
        self.cache_audio = bool(cache_audio)
        self.audio_cache = {}
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

        with self._sf.SoundFile(path, mode="r") as f:
            sr = int(f.samplerate)
            total_frames = int(len(f))

            src_need = int(round(self.clip_frames * float(sr) / float(self.target_sr)))
            src_need = max(1, src_need + 2)

            if total_frames > src_need:
                start = (total_frames - src_need) // 2
                f.seek(start)
                wav = f.read(frames=src_need, dtype="float32", always_2d=True)
            else:
                f.seek(0)
                wav = f.read(dtype="float32", always_2d=True)

        wav = wav.mean(axis=1)
        if sr != self.target_sr:
            wav = resample_poly(wav, up=self.target_sr, down=sr).astype(np.float32)
        return wav

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        x_img = self.img_tf(img)

        audio_path = str(row["audio_path"])
        if self.cache_audio and audio_path in self.audio_cache:
            wav_np = self.audio_cache[audio_path]
        else:
            wav_raw = self._load_audio(audio_path)
            wav_np = center_crop_or_pad_np(wav_raw, self.clip_frames)
            if self.cache_audio:
                self.audio_cache[audio_path] = wav_np

        wav = torch.from_numpy(wav_np).float()

        y = int(row["label_id"])
        sid = row["sample_id"]
        return x_img, wav, y, sid


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
        hop_length = int(hop_length_override or (ck.get("hop_length", 256) if isinstance(ck, dict) else 256))
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


@torch.no_grad()
def extract_features(img_backbone: nn.Module, audio_backbone: nn.Module, loader: DataLoader, device: torch.device, desc: str):
    img_backbone.eval()
    audio_backbone.eval()

    xs, ys, sids = [], [], []
    total_batches = len(loader)

    for bi, (x_img, x_audio, y, sid) in enumerate(loader, start=1):
        x_img = x_img.to(device, non_blocking=True)
        x_audio = x_audio.to(device, non_blocking=True)

        f_img = img_backbone(x_img)
        f_audio = audio_backbone(x_audio)

        f_img = F.normalize(f_img, p=2, dim=1)
        f_audio = F.normalize(f_audio, p=2, dim=1)

        feat = torch.cat([f_img, f_audio], dim=1)
        xs.append(feat.cpu())
        ys.append(torch.as_tensor(y, dtype=torch.long))
        sids.extend(list(sid))

        if bi % 5 == 0 or bi == total_batches:
            print(f"[extract:{desc}] batch {bi}/{total_batches}", flush=True)

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return X, Y, sids


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, device: torch.device, batch: int = 512):
    model.eval()
    outs = []
    n = X.shape[0]
    for i in range(0, n, batch):
        xb = X[i : i + batch].to(device)
        logits = model(xb)
        outs.append(logits.cpu())
    return torch.cat(outs, dim=0)


def eval_from_logits(y_true: np.ndarray, logits: torch.Tensor):
    prob = torch.softmax(logits, dim=1).numpy()
    pred = np.argmax(prob, axis=1)
    acc = float(accuracy_score(y_true, pred))
    f1 = float(f1_score(y_true, pred, average="macro"))
    cm = confusion_matrix(y_true, pred)
    conf = prob.max(axis=1)
    return acc, f1, cm, pred, conf


def train_head(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    num_classes: int,
    args,
    device: torch.device,
):
    model = FusionHead(
        in_dim=int(X_train.shape[1]),
        num_classes=num_classes,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)

    ds = TensorDataset(X_train, y_train)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=True, num_workers=0)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    best = {"epoch": -1, "val_acc": -1.0, "val_f1": -1.0, "state": None}
    wait = 0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)

            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward()
            opt.step()

            losses.append(float(loss.item()))

        val_logits = predict_logits(model, X_val, device=device, batch=max(args.batch, 256))
        val_acc, val_f1, _, _, _ = eval_from_logits(y_val.numpy(), val_logits)

        hist_row = {
            "epoch": ep,
            "train_loss": float(np.mean(losses)) if losses else 0.0,
            "val_acc": val_acc,
            "val_f1": val_f1,
        }
        history.append(hist_row)

        improved = (val_f1 > best["val_f1"]) or (
            abs(val_f1 - best["val_f1"]) < 1e-9 and val_acc > best["val_acc"]
        )

        if improved:
            best["epoch"] = ep
            best["val_acc"] = val_acc
            best["val_f1"] = val_f1
            best["state"] = copy.deepcopy(model.state_dict())
            wait = 0
        else:
            wait += 1

        print(
            f"[{ep:03d}] loss={hist_row['train_loss']:.4f} "
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}",
            flush=True,
        )

        if wait >= args.patience:
            print(f"Early stop at epoch {ep}, best epoch={best['epoch']}", flush=True)
            break

    if best["state"] is None:
        best["state"] = copy.deepcopy(model.state_dict())

    model.load_state_dict(best["state"])
    return model, best, history


def parse_args():
    p = argparse.ArgumentParser("Train image+audio fusion head with frozen branches (baseline)")
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
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
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

    print("split_dir:", split_dir, flush=True)
    print("train/val/test:", len(df_train), len(df_val), len(df_test), flush=True)
    print("num_classes:", num_classes, flush=True)
    print("device:", str(device), "num_gpu:", num_gpu, "use_data_parallel:", use_dp, flush=True)

    ds_train = PairedImageAudioDataset(df_train, args.img_size, args.target_sr, args.clip_sec)
    ds_val = PairedImageAudioDataset(df_val, args.img_size, args.target_sr, args.clip_sec)
    ds_test = PairedImageAudioDataset(df_test, args.img_size, args.target_sr, args.clip_sec)

    dl_train = DataLoader(
        ds_train,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        persistent_workers=(args.num_workers > 0),
    )

    image_ckpt = str((root / args.image_ckpt).resolve())
    audio_ckpt = str((root / args.audio_ckpt).resolve())

    img_backbone = ImageEffB0Backbone(image_ckpt).to(device)
    audio_backbone = AudioBackboneFromCkpt(
        audio_ckpt,
        target_sr_override=args.target_sr,
        n_fft_override=args.n_fft,
        hop_length_override=args.hop_length,
        n_mels_override=args.n_mels,
    ).to(device)

    if use_dp:
        img_backbone = nn.DataParallel(img_backbone)
        audio_backbone = nn.DataParallel(audio_backbone)

    for p in img_backbone.parameters():
        p.requires_grad = False
    for p in audio_backbone.parameters():
        p.requires_grad = False

    X_train, y_train, sid_train = extract_features(img_backbone, audio_backbone, dl_train, device, desc="train")
    X_val, y_val, sid_val = extract_features(img_backbone, audio_backbone, dl_val, device, desc="val")
    X_test, y_test, sid_test = extract_features(img_backbone, audio_backbone, dl_test, device, desc="test")

    print("feature_dim:", int(X_train.shape[1]), flush=True)
    print("features train/val/test:", int(X_train.shape[0]), int(X_val.shape[0]), int(X_test.shape[0]), flush=True)

    head, best, history = train_head(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=num_classes,
        args=args,
        device=device,
    )

    test_logits = predict_logits(head, X_test, device=device, batch=max(args.batch, 256))
    test_acc, test_f1, test_cm, test_pred, test_conf = eval_from_logits(y_test.numpy(), test_logits)

    best_path = out_dir / "best_head.pt"
    torch.save(
        {
            "model": head.state_dict(),
            "input_dim": int(X_train.shape[1]),
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
        "feature_dim": int(X_train.shape[1]),
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
