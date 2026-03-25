#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchvision import transforms
from torchvision.models import efficientnet_b0


IMU_COLS_9 = [
    "AccelerationX", "AccelerationY", "AccelerationZ",
    "GyroX", "GyroY", "GyroZ",
    "MagneticFieldX", "MagneticFieldY", "MagneticFieldZ",
]


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


def stratified_split_train_val(df_train: pd.DataFrame, val_ratio: float, seed: int):
    rng = random.Random(seed)
    tr_rows, va_rows = [], []

    for label in sorted(df_train["label"].unique().tolist()):
        sub = df_train[df_train["label"] == label].sort_values("sample_id")
        rows = sub.to_dict("records")
        rng.shuffle(rows)

        n = len(rows)
        if n <= 1:
            n_val = 0
        else:
            n_val = int(round(n * val_ratio))
            n_val = max(1, n_val)
            n_val = min(n - 1, n_val)

        va_rows.extend(rows[:n_val])
        tr_rows.extend(rows[n_val:])

    df_tr = pd.DataFrame(tr_rows).sort_values("sample_id").reset_index(drop=True)
    df_va = pd.DataFrame(va_rows).sort_values("sample_id").reset_index(drop=True)
    return df_tr, df_va


def resample_sequence(arr: np.ndarray, target_len: int = 256) -> np.ndarray:
    t, c = arr.shape
    if t == target_len:
        return arr.astype(np.float32)

    x_old = np.linspace(0.0, 1.0, t)
    x_new = np.linspace(0.0, 1.0, target_len)

    out = np.zeros((target_len, c), dtype=np.float32)
    for i in range(c):
        out[:, i] = np.interp(x_new, x_old, arr[:, i]).astype(np.float32)
    return out


def load_imu_csv(imu_path: str, target_len: int = 256) -> np.ndarray:
    df = pd.read_csv(imu_path)
    df.columns = [str(c).strip() for c in df.columns]

    present_sensor_cols = [c for c in IMU_COLS_9 if c in df.columns]
    if len(present_sensor_cols) >= 6:
        x = df[present_sensor_cols].copy()
        for miss in (c for c in IMU_COLS_9 if c not in present_sensor_cols):
            x[miss] = 0.0
        x = x[IMU_COLS_9]
    else:
        # Fallback: use first 9 columns that can be coerced to numeric.
        x = df.copy()
        keep = []
        for c in x.columns:
            v = pd.to_numeric(x[c], errors="coerce")
            if v.notna().sum() > 0:
                keep.append(c)
            if len(keep) >= 6:
                break
        if len(keep) < 6:
            raise ValueError(f"IMU columns not enough in {imu_path}")
        x = x[keep].copy()

    for c in x.columns:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna(how="all")
    if len(x) == 0:
        raise ValueError(f"IMU empty after cleaning: {imu_path}")

    x = x.ffill().bfill().fillna(0.0)
    arr = x.to_numpy(dtype=np.float32)

    if arr.shape[1] > 9:
        arr = arr[:, :9]
    elif arr.shape[1] < 9:
        pad = np.zeros((arr.shape[0], 9 - arr.shape[1]), dtype=np.float32)
        arr = np.concatenate([arr, pad], axis=1)

    arr = resample_sequence(arr, target_len=target_len)
    return arr


def compute_train_imu_norm(df_train_sub: pd.DataFrame, imu_len: int):
    s = np.zeros(9, dtype=np.float64)
    ss = np.zeros(9, dtype=np.float64)
    n = 0

    total_files = len(df_train_sub)
    for i, (_, row) in enumerate(df_train_sub.iterrows(), start=1):
        arr = load_imu_csv(row["imu_path"], target_len=imu_len)  # (T,9)
        s += arr.sum(axis=0)
        ss += (arr ** 2).sum(axis=0)
        n += arr.shape[0]
        if i % 50 == 0 or i == total_files:
            print(f"[imu_norm] {i}/{total_files}")

    if n == 0:
        raise RuntimeError("No IMU samples for normalization.")

    mean = s / n
    var = ss / n - mean ** 2
    var = np.maximum(var, 1e-8)
    std = np.sqrt(var)
    return mean.astype(np.float32), std.astype(np.float32)


class PairedAlignedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, imu_len: int, imu_mean: np.ndarray, imu_std: np.ndarray):
        self.df = df.reset_index(drop=True)
        self.imu_len = imu_len
        self.imu_mean = imu_mean
        self.imu_std = imu_std

        self.img_tf = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img = Image.open(row["image_path"]).convert("RGB")
        x_img = self.img_tf(img)

        imu = load_imu_csv(row["imu_path"], target_len=self.imu_len)
        imu = (imu - self.imu_mean[None, :]) / (self.imu_std[None, :] + 1e-6)
        x_imu = torch.from_numpy(imu).float().transpose(0, 1).contiguous()  # (9,T)

        y = int(row["label_id"])
        sid = row["sample_id"]
        return x_img, x_imu, y, sid


class IMU1DCNNV1(nn.Module):
    def __init__(self, in_ch=9, num_classes=12, width=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.Conv1d(width, width, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.Conv1d(width, width * 2, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(width * 2, width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 4),
            nn.ReLU(inplace=True),
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(width * 4, num_classes),
        )

    def forward(self, x):
        z = self.net(x)
        return self.head(z)

    def forward_features(self, x):
        z = self.net(x)
        return self.head[:-1](z)


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


class IMUBackboneFromCkpt(nn.Module):
    def __init__(self, ckpt_path: str):
        super().__init__()
        ck = torch.load(ckpt_path, map_location="cpu")
        sd = ck["model"] if isinstance(ck, dict) and "model" in ck else ck

        if "net.0.weight" not in sd or "head.3.weight" not in sd:
            raise RuntimeError("Unexpected IMU ckpt format")

        width = int(sd["net.0.weight"].shape[0])
        in_ch = int(sd["net.0.weight"].shape[1])
        num_classes = int(sd["head.3.weight"].shape[0])

        self.model = IMU1DCNNV1(in_ch=in_ch, num_classes=num_classes, width=width, dropout=0.2)
        self.model.load_state_dict(sd, strict=True)
        self.out_dim = int(sd["head.3.weight"].shape[1])

    def forward_features(self, x):
        return self.model.forward_features(x)


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
def extract_features(
    img_backbone: ImageEffB0Backbone,
    imu_backbone: IMUBackboneFromCkpt,
    loader: DataLoader,
    device: str,
    desc: str = "split",
):
    img_backbone.eval()
    imu_backbone.eval()

    xs, ys, sids = [], [], []

    total_batches = len(loader)
    for bi, (x_img, x_imu, y, sid) in enumerate(loader, start=1):
        x_img = x_img.to(device)
        x_imu = x_imu.to(device)

        f_img = img_backbone(x_img)
        f_imu = imu_backbone.forward_features(x_imu)

        # Normalize each modality feature to balance scales.
        f_img = F.normalize(f_img, p=2, dim=1)
        f_imu = F.normalize(f_imu, p=2, dim=1)

        feat = torch.cat([f_img, f_imu], dim=1)

        xs.append(feat.cpu())
        ys.append(torch.as_tensor(y, dtype=torch.long))
        sids.extend(list(sid))

        if bi % 5 == 0 or bi == total_batches:
            print(f"[extract:{desc}] batch {bi}/{total_batches}")

    X = torch.cat(xs, dim=0)
    Y = torch.cat(ys, dim=0)
    return X, Y, sids


@torch.no_grad()
def predict_logits(model: nn.Module, X: torch.Tensor, device: str, batch: int = 512):
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
    device: str,
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

    best = {
        "epoch": -1,
        "val_acc": -1.0,
        "val_f1": -1.0,
        "state": None,
    }

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
            f"val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if wait >= args.patience:
            print(f"Early stop at epoch {ep}, best epoch={best['epoch']}")
            break

    if best["state"] is None:
        best["state"] = copy.deepcopy(model.state_dict())

    model.load_state_dict(best["state"])
    return model, best, history


def parse_args():
    p = argparse.ArgumentParser("Train fusion head on aligned data with frozen image/IMU branches")
    p.add_argument("--split_dir", type=Path, default=Path("data/aligned-data/split_fusion_v1"))
    p.add_argument(
        "--image_ckpt",
        type=Path,
        default=Path("image-branch/runs/effb0_s224_split1_resize/best.pt"),
    )
    p.add_argument(
        "--imu_ckpt",
        type=Path,
        default=Path("imu-branch/runs/imu_trial_split_cnn/best.pt"),
    )
    p.add_argument("--out_dir", type=Path, default=Path("runs/fusion_head_aligned"))

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--imu_len", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=20)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--use_cuda", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    root = Path.cwd()
    split_dir = (root / args.split_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"train.csv/test.csv not found in {split_dir}")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    req_cols = ["sample_id", "label", "label_id", "image_path", "imu_path"]
    for c in req_cols:
        if c not in df_train.columns or c not in df_test.columns:
            raise RuntimeError(f"Missing required column: {c}")

    for df in (df_train, df_test):
        df["image_path"] = df["image_path"].astype(str).apply(lambda x: resolve_path(x, root))
        df["imu_path"] = df["imu_path"].astype(str).apply(lambda x: resolve_path(x, root))

    # Secondary split: train_sub / val from train only (no test leakage).
    df_train_sub, df_val = stratified_split_train_val(df_train, args.val_ratio, args.seed)

    tr_ids = set(df_train_sub["sample_id"].tolist())
    va_ids = set(df_val["sample_id"].tolist())
    te_ids = set(df_test["sample_id"].tolist())

    assert tr_ids.isdisjoint(va_ids)
    assert tr_ids.isdisjoint(te_ids)
    assert va_ids.isdisjoint(te_ids)

    num_classes = int(max(df_train["label_id"].max(), df_test["label_id"].max()) + 1)
    label_pairs = pd.concat(
        [df_train[["label_id", "label"]], df_test[["label_id", "label"]]],
        ignore_index=True,
    ).drop_duplicates()
    idx_to_label = {int(r.label_id): str(r.label) for r in label_pairs.itertuples(index=False)}

    print("split_dir:", split_dir)
    print("train/test:", len(df_train), len(df_test))
    print("train_sub/val:", len(df_train_sub), len(df_val))
    print("num_classes:", num_classes)

    imu_mean, imu_std = compute_train_imu_norm(df_train_sub, imu_len=args.imu_len)
    np.save(out_dir / "imu_mean.npy", imu_mean)
    np.save(out_dir / "imu_std.npy", imu_std)

    ds_train = PairedAlignedDataset(df_train_sub, args.img_size, args.imu_len, imu_mean, imu_std)
    ds_val = PairedAlignedDataset(df_val, args.img_size, args.imu_len, imu_mean, imu_std)
    ds_test = PairedAlignedDataset(df_test, args.img_size, args.imu_len, imu_mean, imu_std)

    dl_train = DataLoader(ds_train, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    dl_val = DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    dl_test = DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print("device:", device)

    img_backbone = ImageEffB0Backbone(str((root / args.image_ckpt).resolve())).to(device)
    imu_backbone = IMUBackboneFromCkpt(str((root / args.imu_ckpt).resolve())).to(device)

    for p in img_backbone.parameters():
        p.requires_grad = False
    for p in imu_backbone.parameters():
        p.requires_grad = False

    X_train, y_train, sid_train = extract_features(
        img_backbone, imu_backbone, dl_train, device, desc="train_sub"
    )
    X_val, y_val, sid_val = extract_features(
        img_backbone, imu_backbone, dl_val, device, desc="val"
    )
    X_test, y_test, sid_test = extract_features(
        img_backbone, imu_backbone, dl_test, device, desc="test"
    )

    print("feature dims:", X_train.shape[1])
    print("features train/val/test:", X_train.shape[0], X_val.shape[0], X_test.shape[0])

    head, best, history = train_head(
        X_train, y_train, X_val, y_val,
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
            "imu_mean_path": str((out_dir / "imu_mean.npy").resolve()),
            "imu_std_path": str((out_dir / "imu_std.npy").resolve()),
            "image_ckpt": str((root / args.image_ckpt).resolve()),
            "imu_ckpt": str((root / args.imu_ckpt).resolve()),
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
                "imu_path": r["imu_path"],
            }
        )

    pd.DataFrame(pred_rows).to_csv(out_dir / "test_predictions.csv", index=False, encoding="utf-8")

    df_train_sub.to_csv(out_dir / "train_sub.csv", index=False, encoding="utf-8")
    df_val.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    df_test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    metrics = {
        "split_dir": str(split_dir),
        "image_ckpt": str((root / args.image_ckpt).resolve()),
        "imu_ckpt": str((root / args.imu_ckpt).resolve()),
        "device": device,
        "seed": args.seed,
        "num_classes": num_classes,
        "num_train_total": int(len(df_train)),
        "num_test": int(len(df_test)),
        "num_train_sub": int(len(df_train_sub)),
        "num_val": int(len(df_val)),
        "feature_dim": int(X_train.shape[1]),
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "best_val_f1": float(best["val_f1"]),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "test_confusion_matrix": test_cm.tolist(),
        "history": history,
        "no_leakage": {
            "train_sub_vs_val_overlap": int(len(tr_ids & va_ids)),
            "train_sub_vs_test_overlap": int(len(tr_ids & te_ids)),
            "val_vs_test_overlap": int(len(va_ids & te_ids)),
        },
    }

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== DONE ===")
    print("out_dir:", out_dir)
    print("best_head:", best_path)
    print(f"test_acc={test_acc:.4f}, test_macro_f1={test_f1:.4f}")


if __name__ == "__main__":
    main()
