# import os, json, time, argparse
# import numpy as np
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# # ----------------------------
# # 1) Dataset
# # ----------------------------
# class IMUWindowNPZ(Dataset):
#     def __init__(self, X, y):
#         self.X = X  # (N, T, C)
#         self.y = y  # (N,)
#         assert self.X.ndim == 3 and self.y.ndim == 1
#         assert self.X.shape[0] == self.y.shape[0]

#     def __len__(self):
#         return self.X.shape[0]

#     def __getitem__(self, idx):
#         x = self.X[idx]  # (T, C)
#         # to (C, T) for Conv1d
#         x = torch.from_numpy(x).float().transpose(0, 1)  # (C, T)
#         y = int(self.y[idx])
#         return x, y

# # ----------------------------
# # 2) Simple 1D CNN baseline
# # ----------------------------
# class CNN1D(nn.Module):
#     def __init__(self, in_ch=9, num_classes=12, width=64, dropout=0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, width, kernel_size=9, padding=4),
#             nn.BatchNorm1d(width),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),

#             nn.Conv1d(width, width*2, kernel_size=7, padding=3),
#             nn.BatchNorm1d(width*2),
#             nn.ReLU(inplace=True),
#             nn.MaxPool1d(2),

#             nn.Conv1d(width*2, width*2, kernel_size=5, padding=2),
#             nn.BatchNorm1d(width*2),
#             nn.ReLU(inplace=True),

#             nn.AdaptiveAvgPool1d(1),  # -> (B, width*2, 1)
#             nn.Flatten(),             # -> (B, width*2)
#             nn.Dropout(dropout),
#             nn.Linear(width*2, num_classes)
#         )

#     def forward(self, x):
#         return self.net(x)

# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     ys, ps = [], []
#     for x, y in loader:
#         x = x.to(device, non_blocking=True)
#         logits = model(x)
#         pred = logits.argmax(1).cpu().numpy()
#         ys.append(np.array(y))
#         ps.append(pred)
#     y = np.concatenate(ys)
#     p = np.concatenate(ps)
#     acc = accuracy_score(y, p)
#     f1 = f1_score(y, p, average="macro")
#     cm = confusion_matrix(y, p)
#     return float(acc), float(f1), cm

# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--npz", type=str, required=True)
#     ap.add_argument("--out_dir", type=str, default="runs/imu_cnn1d_npz")
#     ap.add_argument("--epochs", type=int, default=30)
#     ap.add_argument("--batch", type=int, default=256)
#     ap.add_argument("--lr", type=float, default=1e-3)
#     ap.add_argument("--wd", type=float, default=1e-4)
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--num_workers", type=int, default=2)
#     ap.add_argument("--use_cuda", action="store_true")
#     args = ap.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     np.random.seed(args.seed)
#     torch.manual_seed(args.seed)
#     torch.cuda.manual_seed_all(args.seed)

#     device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
#     print("device:", device)

#     data = np.load(args.npz)
#     X = data["X"]  # (N, T, C)
#     y = data["y"]  # (N,)

#     num_classes = int(y.max() + 1)
#     T, C = X.shape[1], X.shape[2]
#     print("Loaded:", args.npz)
#     print("X:", X.shape, "y:", y.shape, "num_classes:", num_classes, "T:", T, "C:", C)

#     # ---------
#     # IMPORTANT:
#     # This split is window-level (may leak). Use for quick sanity run only.
#     # ---------
#     idx = np.arange(len(y))
#     tr_idx, te_idx = train_test_split(idx, test_size=0.1, random_state=args.seed, stratify=y)
#     tr_idx, va_idx = train_test_split(tr_idx, test_size=0.1, random_state=args.seed, stratify=y[tr_idx])

#     X_tr, y_tr = X[tr_idx], y[tr_idx]
#     X_va, y_va = X[va_idx], y[va_idx]
#     X_te, y_te = X[te_idx], y[te_idx]

#     # Normalize per-channel using train stats
#     # X: (N, T, C)
#     mean = X_tr.reshape(-1, C).mean(axis=0)
#     std  = X_tr.reshape(-1, C).std(axis=0) + 1e-6
#     X_tr = (X_tr - mean) / std
#     X_va = (X_va - mean) / std
#     X_te = (X_te - mean) / std

#     train_ds = IMUWindowNPZ(X_tr, y_tr)
#     val_ds   = IMUWindowNPZ(X_va, y_va)
#     test_ds  = IMUWindowNPZ(X_te, y_te)

#     train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
#                               num_workers=args.num_workers, pin_memory=(device=="cuda"))
#     val_loader   = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
#                               num_workers=args.num_workers, pin_memory=(device=="cuda"))
#     test_loader  = DataLoader(test_ds, batch_size=args.batch, shuffle=False,
#                               num_workers=args.num_workers, pin_memory=(device=="cuda"))

#     model = CNN1D(in_ch=C, num_classes=num_classes, width=64, dropout=0.2).to(device)
#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     best_val = -1.0
#     best_path = os.path.join(args.out_dir, "best.pt")

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         t0 = time.time()
#         losses = []

#         for x, yb in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
#             x = x.to(device, non_blocking=True)
#             yb = yb.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             logits = model(x)
#             loss = criterion(logits, yb)
#             loss.backward()
#             optimizer.step()

#             losses.append(float(loss.item()))

#         scheduler.step()
#         val_acc, val_f1, _ = evaluate(model, val_loader, device)
#         dt = time.time() - t0
#         print(f"[{epoch}] loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={dt:.1f}s")

#         if val_acc > best_val:
#             best_val = val_acc
#             torch.save({
#                 "model": model.state_dict(),
#                 "mean": mean.tolist(),
#                 "std": std.tolist(),
#                 "num_classes": num_classes,
#                 "T": T,
#                 "C": C,
#             }, best_path)

#     ckpt = torch.load(best_path, map_location=device)
#     model.load_state_dict(ckpt["model"])
#     test_acc, test_f1, cm = evaluate(model, test_loader, device)

#     metrics = {
#         "npz": args.npz,
#         "epochs": args.epochs,
#         "batch": args.batch,
#         "lr": args.lr,
#         "wd": args.wd,
#         "test_acc": test_acc,
#         "test_macro_f1": test_f1,
#         "confusion_matrix": cm.tolist(),
#         "note": "window-level split (may leak). For final result, split by original trial/pkl.",
#     }
#     with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics, f, ensure_ascii=False, indent=2)

#     print("=== DONE ===")
#     print("best:", best_path)
#     print("metrics:", os.path.join(args.out_dir, "metrics.json"))
#     print("test_acc:", test_acc, "macro_f1:", test_f1)

# if __name__ == "__main__":
#     main()


# import os
# import json
# import time
# import argparse
# import numpy as np
# from dataclasses import dataclass

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader

# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# # -------------------------
# # Model: simple 1D CNN baseline
# # Input: (T, C) -> transpose -> (C, T)
# # -------------------------
# class IMU1DCNN(nn.Module):
#     def __init__(self, in_ch: int, num_classes: int, width: int = 128, dropout: float = 0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, width, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm1d(width),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
#             nn.BatchNorm1d(width),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm1d(width * 2),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width * 2, width * 2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(width * 2),
#             nn.ReLU(inplace=True),

#             nn.AdaptiveAvgPool1d(1),  # (B, 2W, 1)
#             nn.Flatten(),             # (B, 2W)
#             nn.Dropout(dropout),
#             nn.Linear(width * 2, num_classes),
#         )

#     def forward(self, x):
#         # x: (B, T, C) -> (B, C, T)
#         x = x.transpose(1, 2).contiguous()
#         return self.net(x)


# # -------------------------
# # Dataset with lazy normalization (no huge array allocation)
# # -------------------------
# class IMUNPZDataset(Dataset):
#     def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
#                  mean: np.ndarray, std: np.ndarray):
#         """
#         X: (N, T, C) float32 numpy array (loaded from npz)
#         indices: indices into X/y for this split
#         mean/std: shape (C,) float32
#         """
#         self.X = X
#         self.y = y
#         self.idx = indices.astype(np.int64, copy=False)
#         self.mean = mean.astype(np.float32, copy=False)
#         self.std = std.astype(np.float32, copy=False)

#     def __len__(self):
#         return self.idx.shape[0]

#     def __getitem__(self, i):
#         j = self.idx[i]
#         x = self.X[j]  # (T, C) view
#         # per-sample normalize (broadcast on T)
#         x = (x - self.mean) / self.std
#         return torch.from_numpy(x).float(), int(self.y[j])


# # -------------------------
# # Utils
# # -------------------------
# @torch.no_grad()
# def evaluate(model, loader, device):
#     model.eval()
#     ys, ps = [], []
#     for xb, yb in loader:
#         xb = xb.to(device, non_blocking=True)
#         logits = model(xb)
#         pred = logits.argmax(1).cpu().numpy()
#         ps.append(pred)
#         ys.append(np.asarray(yb))
#     y = np.concatenate(ys)
#     p = np.concatenate(ps)
#     acc = accuracy_score(y, p)
#     f1 = f1_score(y, p, average="macro")
#     cm = confusion_matrix(y, p)
#     return float(acc), float(f1), cm


# def estimate_mean_std(X: np.ndarray, indices: np.ndarray, max_samples: int = 20000, seed: int = 42):
#     """
#     Estimate per-channel mean/std from a random subset of training indices.
#     This avoids loading/processing the whole training tensor at once.

#     X: (N, T, C) float32
#     returns mean,std: (C,)
#     """
#     rng = np.random.default_rng(seed)
#     if indices.shape[0] > max_samples:
#         pick = rng.choice(indices, size=max_samples, replace=False)
#     else:
#         pick = indices

#     # shape: (K, T, C) -> compute over K*T
#     # Important: use float64 accumulator for numerical stability, but not huge memory
#     xs = X[pick]  # this is at most (20000, 256, 9) ~ 184MB float32, OK
#     mean = xs.mean(axis=(0, 1), dtype=np.float64)
#     var = xs.var(axis=(0, 1), dtype=np.float64)
#     std = np.sqrt(var + 1e-8)
#     return mean.astype(np.float32), std.astype(np.float32)


# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--npz", type=str, required=True)
#     ap.add_argument("--use_cuda", action="store_true")
#     ap.add_argument("--batch", type=int, default=256)
#     ap.add_argument("--epochs", type=int, default=30)
#     ap.add_argument("--lr", type=float, default=3e-4)
#     ap.add_argument("--wd", type=float, default=1e-2)
#     ap.add_argument("--num_workers", type=int, default=2)
#     ap.add_argument("--pin_memory", action="store_true")
#     ap.add_argument("--seed", type=int, default=42)
#     ap.add_argument("--max_norm_samples", type=int, default=20000,
#                     help="how many training samples to estimate mean/std")
#     ap.add_argument("--out_dir", type=str, default="runs/imu_npz_cnn_w256")
#     args = ap.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
#     print("device:", device)

#     # Load npz
#     data = np.load(args.npz)
#     X = data["X"]  # (N, T, C)
#     y = data["y"]  # (N,)
#     print("Loaded:", args.npz)
#     print("X:", X.shape, "y:", y.shape, "num_classes:", int(y.max() + 1), "T:", X.shape[1], "C:", X.shape[2])

#     # Ensure dtype float32
#     if X.dtype != np.float32:
#         X = X.astype(np.float32, copy=False)

#     # Split: simple random split on windows (baseline only)
#     # NOTE: Later we will do "trial-level split" to avoid leakage.
#     rng = np.random.default_rng(args.seed)
#     N = y.shape[0]
#     perm = rng.permutation(N)

#     n_train = int(0.8 * N)
#     n_val = int(0.1 * N)
#     tr_idx = perm[:n_train]
#     va_idx = perm[n_train:n_train + n_val]
#     te_idx = perm[n_train + n_val:]

#     # Estimate mean/std from subset of training
#     mean, std = estimate_mean_std(X, tr_idx, max_samples=args.max_norm_samples, seed=args.seed)
#     # avoid zero std
#     std = np.maximum(std, 1e-6).astype(np.float32)

#     print("mean:", mean)
#     print("std :", std)

#     # Datasets / loaders
#     train_ds = IMUNPZDataset(X, y, tr_idx, mean, std)
#     val_ds   = IMUNPZDataset(X, y, va_idx, mean, std)
#     test_ds  = IMUNPZDataset(X, y, te_idx, mean, std)

#     train_loader = DataLoader(
#         train_ds, batch_size=args.batch, shuffle=True,
#         num_workers=args.num_workers, pin_memory=args.pin_memory
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=args.batch, shuffle=False,
#         num_workers=args.num_workers, pin_memory=args.pin_memory
#     )
#     test_loader = DataLoader(
#         test_ds, batch_size=args.batch, shuffle=False,
#         num_workers=args.num_workers, pin_memory=args.pin_memory
#     )

#     num_classes = int(y.max() + 1)
#     in_ch = X.shape[2]

#     model = IMU1DCNN(in_ch=in_ch, num_classes=num_classes, width=128, dropout=0.2).to(device)

#     criterion = nn.CrossEntropyLoss()
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

#     scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))

#     best_val = -1.0
#     best_path = os.path.join(args.out_dir, "best.pt")

#     for epoch in range(1, args.epochs + 1):
#         model.train()
#         t0 = time.time()
#         losses = []

#         for xb, yb in train_loader:
#             xb = xb.to(device, non_blocking=True)
#             yb = yb.to(device, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)
#             with torch.cuda.amp.autocast(enabled=(device == "cuda")):
#                 logits = model(xb)
#                 loss = criterion(logits, yb)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             losses.append(float(loss.item()))

#         scheduler.step()

#         val_acc, val_f1, _ = evaluate(model, val_loader, device)
#         dt = time.time() - t0
#         print(f"[{epoch}] loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={dt:.1f}s")

#         if val_acc > best_val:
#             best_val = val_acc
#             torch.save({
#                 "model": model.state_dict(),
#                 "mean": mean,
#                 "std": std,
#                 "num_classes": num_classes,
#                 "in_ch": in_ch,
#                 "T": int(X.shape[1]),
#             }, best_path)

#     # Test best
#     ckpt = torch.load(best_path, map_location=device, weights_only=False)
#     model.load_state_dict(ckpt["model"])
#     test_acc, test_f1, cm = evaluate(model, test_loader, device)

#     metrics = {
#         "npz": args.npz,
#         "device": device,
#         "batch": args.batch,
#         "epochs": args.epochs,
#         "lr": args.lr,
#         "wd": args.wd,
#         "max_norm_samples": args.max_norm_samples,
#         "test_acc": test_acc,
#         "test_macro_f1": test_f1,
#         "confusion_matrix": cm.tolist(),
#         "best_path": best_path,
#     }
#     with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics, f, ensure_ascii=False, indent=2)

#     print("=== DONE ===")
#     print("best:", best_path)
#     print("test_acc:", test_acc, "macro_f1:", test_f1)
#     print("metrics:", os.path.join(args.out_dir, "metrics.json"))


# if __name__ == "__main__":
#     main()



import os
import json
import time
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# -------------------------
# Model: simple 1D CNN baseline
# Input: (T, C) -> transpose -> (C, T)
# -------------------------
class IMU1DCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, width: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.Conv1d(width, width, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(inplace=True),

            nn.Conv1d(width, width * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),

            nn.Conv1d(width * 2, width * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes),
        )

    def forward(self, x):
        x = x.transpose(1, 2).contiguous()  # (B,T,C)->(B,C,T)
        return self.net(x)


# -------------------------
# Dataset with lazy normalization (no huge array allocation)
# -------------------------
class IMUNPZDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray, indices: np.ndarray,
                 mean: np.ndarray, std: np.ndarray):
        self.X = X
        self.y = y
        self.idx = indices.astype(np.int64, copy=False)
        self.mean = mean.astype(np.float32, copy=False)
        self.std = std.astype(np.float32, copy=False)

    def __len__(self):
        return int(self.idx.shape[0])

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j]  # (T, C) view
        x = (x - self.mean) / self.std
        return torch.from_numpy(x).float(), int(self.y[j])


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        pred = logits.argmax(1).cpu().numpy()
        ps.append(pred)
        ys.append(np.asarray(yb))
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p)
    return float(acc), float(f1), cm


def estimate_mean_std(X: np.ndarray, indices: np.ndarray, max_samples: int = 20000, seed: int = 42):
    """
    Estimate per-channel mean/std from a random subset of training indices.
    Avoids allocating (X_tr - mean)/std for the full training set.
    """
    rng = np.random.default_rng(seed)
    if indices.shape[0] > max_samples:
        pick = rng.choice(indices, size=max_samples, replace=False)
    else:
        pick = indices

    xs = X[pick]  # (K,T,C)
    mean = xs.mean(axis=(0, 1), dtype=np.float64)
    var = xs.var(axis=(0, 1), dtype=np.float64)
    std = np.sqrt(var + 1e-8)
    return mean.astype(np.float32), std.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", type=str, required=True)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=1e-2)
    ap.add_argument("--num_workers", type=int, default=0,
                    help="Windows 上建议先用 0；稳定后再尝试 2/4")
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_norm_samples", type=int, default=20000)
    ap.add_argument("--out_dir", type=str, default="runs/imu_npz_cnn_w256")
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    device = "cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu"
    print("device:", device)

    # Load npz (optionally mmap to reduce RAM pressure)
    # Note: compressed npz cannot be memory-mapped. If you later save to .npy, you can mmap.
    data = np.load(args.npz)
    X = data["X"]
    y = data["y"]

    print("Loaded:", args.npz)
    print("X:", X.shape, "y:", y.shape, "num_classes:", int(y.max() + 1), "T:", X.shape[1], "C:", X.shape[2])

    if X.dtype != np.float32:
        X = X.astype(np.float32, copy=False)

    rng = np.random.default_rng(args.seed)
    N = y.shape[0]
    perm = rng.permutation(N)

    n_train = int(0.8 * N)
    n_val = int(0.1 * N)
    tr_idx = perm[:n_train]
    va_idx = perm[n_train:n_train + n_val]
    te_idx = perm[n_train + n_val:]

    mean, std = estimate_mean_std(X, tr_idx, max_samples=args.max_norm_samples, seed=args.seed)
    std = np.maximum(std, 1e-6).astype(np.float32)

    print("mean:", mean)
    print("std :", std)

    train_ds = IMUNPZDataset(X, y, tr_idx, mean, std)
    val_ds = IMUNPZDataset(X, y, va_idx, mean, std)
    test_ds = IMUNPZDataset(X, y, te_idx, mean, std)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    num_classes = int(y.max() + 1)
    in_ch = int(X.shape[2])
    T = int(X.shape[1])

    model = IMU1DCNN(in_ch=in_ch, num_classes=num_classes, width=args.width, dropout=args.dropout).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # New AMP API (avoids FutureWarning)
    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    best_val = -1.0
    best_path = os.path.join(args.out_dir, "best.pt")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for xb, yb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = criterion(logits, yb)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            losses.append(float(loss.item()))

        scheduler.step()

        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        dt = time.time() - t0
        print(f"[{epoch}] loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={dt:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "mean": mean,
                "std": std,
                "num_classes": num_classes,
                "in_ch": in_ch,
                "T": T,
                "args": vars(args),
            }, best_path)

    # Test best
    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc, test_f1, cm = evaluate(model, test_loader, device)

    metrics = {
        "npz": args.npz,
        "device": device,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "width": args.width,
        "dropout": args.dropout,
        "max_norm_samples": args.max_norm_samples,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "confusion_matrix": cm.tolist(),
        "best_path": best_path,
    }
    with open(os.path.join(args.out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== DONE ===")
    print("best:", best_path)
    print("test_acc:", test_acc, "macro_f1:", test_f1)
    print("metrics:", os.path.join(args.out_dir, "metrics.json"))


if __name__ == "__main__":
    # Windows 多进程 DataLoader 时建议保留 main guard
    main()
