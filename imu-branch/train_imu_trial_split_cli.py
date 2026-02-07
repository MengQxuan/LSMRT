# import os
# import json
# import time
# import argparse
# import numpy as np

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader


# # -------------------------
# # Dataset (memmap + index)
# # -------------------------
# class MemmapWindowDataset(Dataset):
#     """
#     X_path: (N, T, C) float32 npy (memmap)
#     y_path: (N,) int64 npy (memmap)
#     idx:    (M,) int64 indices selecting subset windows
#     """
#     def __init__(self, X_path, y_path, idx_path):
#         self.X = np.load(X_path, mmap_mode="r")  # memmap
#         self.y = np.load(y_path, mmap_mode="r")  # memmap
#         self.idx = np.load(idx_path).astype(np.int64)

#         # infer shapes
#         _, self.T, self.C = self.X.shape

#         # num_classes from y in subset (safe)
#         y_sub = self.y[self.idx[: min(len(self.idx), 200000)]]
#         self.num_classes = int(np.max(y_sub)) + 1

#     def __len__(self):
#         return int(self.idx.shape[0])

#     def __getitem__(self, i):
#         j = int(self.idx[i])
#         x = self.X[j]  # (T,C) float32
#         y = int(self.y[j])
#         # to torch: (C,T) for 1D conv
#         # x = torch.from_numpy(np.asarray(x)).float().transpose(0, 1).contiguous()
#         x = torch.from_numpy(np.asarray(x)).float().transpose(0, 1).contiguous()
#         y = torch.tensor(y, dtype=torch.long)
#         return x, y


# # -------------------------
# # Model: simple 1D CNN
# # -------------------------
# class IMU1DCNN(nn.Module):
#     def __init__(self, in_ch=9, num_classes=12, width=128, dropout=0.2):
#         super().__init__()
#         self.net = nn.Sequential(
#             nn.Conv1d(in_ch, width, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm1d(width),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width, width, kernel_size=5, stride=1, padding=2, bias=False),
#             nn.BatchNorm1d(width),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width, width * 2, kernel_size=5, stride=2, padding=2, bias=False),
#             nn.BatchNorm1d(width * 2),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width * 2, width * 2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm1d(width * 2),
#             nn.ReLU(inplace=True),

#             nn.Conv1d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False),
#             nn.BatchNorm1d(width * 4),
#             nn.ReLU(inplace=True),
#         )
#         self.head = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Flatten(),
#             nn.Dropout(p=dropout),
#             nn.Linear(width * 4, num_classes),
#         )

#     def forward(self, x):  # x: (B,C,T)
#         x = self.net(x)
#         x = self.head(x)
#         return x


# # -------------------------
# # Metrics
# # -------------------------
# @torch.no_grad()
# def eval_epoch(model, loader, device):
#     model.eval()
#     total = 0
#     correct = 0

#     # macro-F1 without sklearn: accumulate confusion
#     num_classes = model.head[-1].out_features
#     conf = torch.zeros((num_classes, num_classes), dtype=torch.int64)

#     for xb, yb in loader:
#         xb = xb.to(device, non_blocking=True)
#         yb = yb.to(device, non_blocking=True)
#         logits = model(xb)
#         pred = torch.argmax(logits, dim=1)

#         total += int(yb.numel())
#         correct += int((pred == yb).sum().item())

#         for t, p in zip(yb.view(-1), pred.view(-1)):
#             conf[int(t.item()), int(p.item())] += 1

#     acc = correct / max(total, 1)

#     # macro f1 from confusion
#     tp = conf.diag().to(torch.float32)
#     fp = conf.sum(0).to(torch.float32) - tp
#     fn = conf.sum(1).to(torch.float32) - tp

#     prec = tp / torch.clamp(tp + fp, min=1.0)
#     rec = tp / torch.clamp(tp + fn, min=1.0)
#     f1 = 2 * prec * rec / torch.clamp(prec + rec, min=1e-8)
#     macro_f1 = float(torch.mean(f1).item())

#     return acc, macro_f1


# def train_one_epoch(model, loader, optimizer, scaler, device, use_amp=True):
#     model.train()
#     total_loss = 0.0
#     n = 0

#     for xb, yb in loader:
#         xb = xb.to(device, non_blocking=True)
#         yb = yb.to(device, non_blocking=True)

#         optimizer.zero_grad(set_to_none=True)

#         if use_amp and device.type == "cuda":
#             with torch.amp.autocast("cuda"):
#                 logits = model(xb)
#                 loss = F.cross_entropy(logits, yb)
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             logits = model(xb)
#             loss = F.cross_entropy(logits, yb)
#             loss.backward()
#             optimizer.step()

#         bs = int(yb.size(0))
#         total_loss += float(loss.item()) * bs
#         n += bs

#     return total_loss / max(n, 1)


# # -------------------------
# # Main
# # -------------------------
# def main():
#     ap = argparse.ArgumentParser()
#     ap.add_argument("--data_dir", type=str, required=True,
#                     help="processed_trial_split directory (contains X.npy,y.npy,train_idx.npy,...)")
#     ap.add_argument("--use_cuda", action="store_true")
#     ap.add_argument("--batch", type=int, default=256)
#     ap.add_argument("--epochs", type=int, default=30)
#     ap.add_argument("--lr", type=float, default=1e-3)
#     ap.add_argument("--wd", type=float, default=1e-4)
#     ap.add_argument("--num_workers", type=int, default=0)
#     ap.add_argument("--pin_memory", action="store_true")
#     ap.add_argument("--width", type=int, default=128)
#     ap.add_argument("--dropout", type=float, default=0.2)
#     ap.add_argument("--out_dir", type=str, default="runs/imu_trial_split_cnn")
#     args = ap.parse_args()

#     os.makedirs(args.out_dir, exist_ok=True)

#     device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
#     print("device:", device)

#     X_path = os.path.join(args.data_dir, "X.npy")
#     y_path = os.path.join(args.data_dir, "y.npy")
#     tr_idx = os.path.join(args.data_dir, "train_idx.npy")
#     va_idx = os.path.join(args.data_dir, "val_idx.npy")
#     te_idx = os.path.join(args.data_dir, "test_idx.npy")

#     ds_tr = MemmapWindowDataset(X_path, y_path, tr_idx)
#     ds_va = MemmapWindowDataset(X_path, y_path, va_idx)
#     ds_te = MemmapWindowDataset(X_path, y_path, te_idx)

#     num_classes = max(ds_tr.num_classes, ds_va.num_classes, ds_te.num_classes)
#     print("Train/Val/Test windows:", len(ds_tr), len(ds_va), len(ds_te))
#     print("T:", ds_tr.T, "C:", ds_tr.C, "num_classes:", num_classes)

#     tr_loader = DataLoader(
#         ds_tr, batch_size=args.batch, shuffle=True,
#         num_workers=args.num_workers, pin_memory=args.pin_memory, drop_last=True
#     )
#     va_loader = DataLoader(
#         ds_va, batch_size=args.batch, shuffle=False,
#         num_workers=args.num_workers, pin_memory=args.pin_memory
#     )
#     te_loader = DataLoader(
#         ds_te, batch_size=args.batch, shuffle=False,
#         num_workers=args.num_workers, pin_memory=args.pin_memory
#     )

#     model = IMU1DCNN(in_ch=ds_tr.C, num_classes=num_classes, width=args.width, dropout=args.dropout).to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
#     scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

#     best_val = -1.0
#     best_path = os.path.join(args.out_dir, "best.pt")
#     metrics_path = os.path.join(args.out_dir, "metrics.json")
#     hist = []

#     for ep in range(1, args.epochs + 1):
#         t0 = time.time()
#         loss = train_one_epoch(model, tr_loader, optimizer, scaler, device, use_amp=True)
#         val_acc, val_f1 = eval_epoch(model, va_loader, device)
#         dt = time.time() - t0

#         line = {"epoch": ep, "loss": loss, "val_acc": val_acc, "val_f1": val_f1, "sec": dt}
#         hist.append(line)
#         print("[{}] loss={:.4f} val_acc={:.4f} val_f1={:.4f} time={:.1f}s".format(
#             ep, loss, val_acc, val_f1, dt
#         ))

#         score = val_f1  # prioritize macro-f1
#         if score > best_val:
#             best_val = score
#             torch.save({"model": model.state_dict(), "epoch": ep, "val_f1": val_f1}, best_path)

#         with open(metrics_path, "w", encoding="utf-8") as f:
#             json.dump({"best_val_f1": best_val, "history": hist}, f, indent=2)

#     # test best
#     ckpt = torch.load(best_path, map_location=device, weights_only=True)
#     model.load_state_dict(ckpt["model"])
#     test_acc, test_f1 = eval_epoch(model, te_loader, device)
#     print("=== DONE ===")
#     print("best:", best_path)
#     print("test_acc:", test_acc, "macro_f1:", test_f1)
#     with open(metrics_path, "r", encoding="utf-8") as f:
#         m = json.load(f)
#     m["test_acc"] = test_acc
#     m["test_f1"] = test_f1
#     with open(metrics_path, "w", encoding="utf-8") as f:
#         json.dump(m, f, indent=2)
#     print("metrics:", metrics_path)


# if __name__ == "__main__":
#     main()





import os
import json
import time
import argparse
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import accuracy_score, f1_score, confusion_matrix


# =========================
# Utils
# =========================
def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def load_json(p):
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, p):
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def find_file(root, names):
    for n in names:
        p = os.path.join(root, n)
        if os.path.exists(p):
            return p
    return None


# =========================
# Dataset
# =========================
class IMUDataset(Dataset):
    """
    Works for:
    - np.memmap (X.mmap)
    - np.ndarray loaded via np.load(..., mmap_mode="r") from X.npy
    """
    def __init__(self, X, y, indices, mean=None, std=None):
        self.X = X
        self.y = y
        self.idx = indices.astype(np.int64)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        j = int(self.idx[i])
        x = self.X[j]          # (T, C), float32
        y = int(self.y[j])

        # avoid non-writable warning
        x = np.array(x, copy=True)

        if self.mean is not None:
            x = (x - self.mean[None, :]) / (self.std[None, :] + 1e-6)

        # Conv1d expects (C, T)
        x = torch.from_numpy(x).float().transpose(0, 1).contiguous()
        return x, y


# =========================
# Model
# =========================
class IMU1DCNN(nn.Module):
    def __init__(self, in_ch, num_classes, width=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),

            nn.Conv1d(width, width, 5, stride=2, padding=2, bias=False),
            nn.BatchNorm1d(width),
            nn.ReLU(),

            nn.Conv1d(width, width * 2, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(width * 2),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(width * 2, num_classes)
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Eval
# =========================
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for xb, yb in loader:
        xb = xb.to(device)
        logits = model(xb)
        pred = logits.argmax(dim=1).cpu().numpy()
        ys.append(np.asarray(yb))
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    return (
        accuracy_score(y, p),
        f1_score(y, p, average="macro"),
        confusion_matrix(y, p)
    )


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--out_dir", default="runs/imu_ood")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=5e-2)
    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--width", type=int, default=128)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--pin_memory", action="store_true")
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--no_val", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()
    set_seed(args.seed)

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print("device:", device)

    root = args.data_dir

    # ---------- load X ----------
    X_path = find_file(root, ["X.mmap", "X.dat", "X.npy"])
    if X_path is None:
        raise FileNotFoundError("X.mmap / X.dat / X.npy not found")

    if X_path.endswith(".npy"):
        X = np.load(X_path, mmap_mode="r")
    else:
        # infer shape from y
        y_tmp = np.load(os.path.join(root, "y.npy"))
        N = y_tmp.shape[0]
        X = np.memmap(X_path, dtype=np.float32, mode="r", shape=(N, 256, 9))

    y = np.load(os.path.join(root, "y.npy"))

    idx_train = np.load(os.path.join(root, "train_idx.npy"))
    idx_test = np.load(os.path.join(root, "test_idx.npy"))

    idx_val = None
    if not args.no_val:
        idx_val = np.load(os.path.join(root, "val_idx.npy"))

    print("Train / Val / Test:",
          len(idx_train),
          0 if idx_val is None else len(idx_val),
          len(idx_test))

    T, C = X.shape[1], X.shape[2]
    num_classes = int(y.max() + 1)
    print("T:", T, "C:", C, "num_classes:", num_classes)

    # ---------- mean / std ----------
    mean_path = os.path.join(root, "mean.npy")
    std_path = os.path.join(root, "std.npy")

    if os.path.exists(mean_path):
        mean = np.load(mean_path)
        std = np.load(std_path)
    else:
        # compute on train only
        xs = []
        for i in idx_train[:100000]:
            xs.append(X[int(i)])
        xs = np.concatenate(xs, axis=0)
        mean = xs.mean(axis=0)
        std = xs.std(axis=0) + 1e-6
        np.save(mean_path, mean)
        np.save(std_path, std)

    # ---------- loaders ----------
    train_ds = IMUDataset(X, y, idx_train, mean, std)
    test_ds = IMUDataset(X, y, idx_test, mean, std)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch, shuffle=True,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch, shuffle=False,
        num_workers=args.num_workers, pin_memory=args.pin_memory
    )

    # ---------- model ----------
    model = IMU1DCNN(C, num_classes, args.width, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    ensure_dir(args.out_dir)
    best_path = os.path.join(args.out_dir, "best.pt")

    best_loss = 1e9

    # ---------- train ----------
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses = []

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(xb)
                loss = crit(logits, yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())

        loss_ep = float(np.mean(losses))
        dt = time.time() - t0
        print(f"[{ep}] loss={loss_ep:.4f} time={dt:.1f}s")

        if loss_ep < best_loss:
            best_loss = loss_ep
            torch.save({"model": model.state_dict()}, best_path)

    # ---------- test ----------
    model.load_state_dict(torch.load(best_path)["model"])
    acc, f1, cm = evaluate(model, test_loader, device)

    print("=== DONE ===")
    print("test_acc:", acc, "macro_f1:", f1)

    save_json({
        "test_acc": acc,
        "test_macro_f1": f1,
        "confusion_matrix": cm.tolist(),
        "no_val": True
    }, os.path.join(args.out_dir, "metrics.json"))


if __name__ == "__main__":
    main()
