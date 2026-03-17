import os
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import timm
from torchvision import transforms


# =========================
# 1. IMU model (与 train_imu_trial_split_cli.py 对齐)
# =========================
class IMU1DCNN_V2(nn.Module):
    """
    你当前 train_imu_trial_split_cli.py 里真正启用的版本
    """
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


class IMU1DCNN_V1(nn.Module):
    """
    旧版更深的 CNN：和你 checkpoint 形状更像
    """
    def __init__(self, in_ch, num_classes, width=128, dropout=0.2):
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
        x = self.net(x)
        x = self.head(x)
        return x


# =========================
# 2. utils
# =========================
IMU_COLS_9 = [
    "AccelerationX", "AccelerationY", "AccelerationZ",
    "GyroX", "GyroY", "GyroZ",
    "MagneticFieldX", "MagneticFieldY", "MagneticFieldZ"
]


def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def softmax_np(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / np.sum(e)


def load_ckpt_state_dict(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    if isinstance(ckpt, dict):
        if "model" in ckpt:
            return ckpt["model"]
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
    return ckpt


def strip_module_prefix(state_dict):
    new_sd = {}
    for k, v in state_dict.items():
        if k.startswith("module."):
            new_sd[k[len("module."):]] = v
        else:
            new_sd[k] = v
    return new_sd


def infer_num_classes_from_state_dict(state_dict):
    # 尝试从最后分类层的 shape 推断类别数
    for k, v in reversed(list(state_dict.items())):
        if k.endswith("weight") and len(v.shape) == 2:
            return v.shape[0]
    raise RuntimeError("无法从 state_dict 中推断 num_classes")


def build_image_transform(img_size=224):
    # 这里先用 ImageNet 标准归一化；如果你 image-branch 训练时有特殊归一化，再改这里
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def read_image(image_path, transform):
    img = Image.open(image_path).convert("RGB")
    x = transform(img)  # (3,H,W)
    return x


def resample_sequence(arr, target_len=256):
    """
    arr: (T, C)
    线性重采样到固定长度
    """
    T, C = arr.shape
    if T == target_len:
        return arr.astype(np.float32)

    x_old = np.linspace(0, 1, T)
    x_new = np.linspace(0, 1, target_len)

    out = np.zeros((target_len, C), dtype=np.float32)
    for c in range(C):
        out[:, c] = np.interp(x_new, x_old, arr[:, c]).astype(np.float32)
    return out


def load_imu_csv(csv_path, expect_cols=IMU_COLS_9, target_len=256):
    df = pd.read_csv(csv_path)

    miss = [c for c in expect_cols if c not in df.columns]
    if miss:
        raise ValueError(f"IMU缺少列: {miss}, file={csv_path}")

    x = df[expect_cols].copy()

    # 数值化 + 去 NaN
    for c in expect_cols:
        x[c] = pd.to_numeric(x[c], errors="coerce")

    x = x.dropna()
    if len(x) == 0:
        raise ValueError(f"IMU文件清洗后为空: {csv_path}")

    arr = x.values.astype(np.float32)   # (T, 9)
    arr = resample_sequence(arr, target_len=target_len)
    return arr


def normalize_imu(arr, mean=None, std=None):
    """
    arr: (T, C)
    """
    if mean is not None and std is not None:
        return (arr - mean[None, :]) / (std[None, :] + 1e-6)

    # 若没有训练集 mean/std，则退化成按样本标准化
    mu = arr.mean(axis=0, keepdims=True)
    sigma = arr.std(axis=0, keepdims=True) + 1e-6
    return (arr - mu) / sigma


def topk_info(prob, idx_to_label, k=3):
    order = np.argsort(-prob)[:k]
    return [(idx_to_label[int(i)], float(prob[int(i)])) for i in order]


# =========================
# 3. dataset
# =========================
class SmallFusionDebugDataset(Dataset):
    def __init__(self, pairs_csv, label_to_idx, img_tf, imu_mean=None, imu_std=None, imu_len=256):
        self.df = pd.read_csv(pairs_csv)

        # 只保留可用样本
        if "img_ok" in self.df.columns:
            self.df = self.df[self.df["img_ok"] == True]
        if "imu_ok" in self.df.columns:
            self.df = self.df[self.df["imu_ok"] == True]

        self.df = self.df.reset_index(drop=True)

        self.label_to_idx = label_to_idx
        self.img_tf = img_tf
        self.imu_mean = imu_mean
        self.imu_std = imu_std
        self.imu_len = imu_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]

        label_name = row["label"]
        y = self.label_to_idx[label_name]

        img = read_image(row["image_path"], self.img_tf)   # (3,H,W)

        imu = load_imu_csv(row["imu_path"], target_len=self.imu_len)   # (T,9)
        imu = normalize_imu(imu, self.imu_mean, self.imu_std)
        imu = torch.from_numpy(imu).float().transpose(0, 1).contiguous()  # (9,T)

        meta = {
            "sample_id": row["sample_id"],
            "label": label_name,
            "image_path": row["image_path"],
            "imu_path": row["imu_path"],
        }
        return img, imu, y, meta


def collate_fn(batch):
    imgs, imus, ys, metas = zip(*batch)
    imgs = torch.stack(imgs, dim=0)
    imus = torch.stack(imus, dim=0)
    ys = torch.tensor(ys, dtype=torch.long)
    return imgs, imus, ys, list(metas)


# =========================
# 4. build models
# =========================
def build_image_model(img_arch, num_classes, ckpt_path, device):
    model = timm.create_model(img_arch, pretrained=False, num_classes=num_classes)
    sd = load_ckpt_state_dict(ckpt_path, device=device)
    sd = strip_module_prefix(sd)

    msg = model.load_state_dict(sd, strict=False)
    print("[Image] load_state_dict:", msg)
    model.eval().to(device)
    return model


def build_imu_model(num_classes, ckpt_path, device, width=128, dropout=0.2, in_ch=9):
    sd = load_ckpt_state_dict(ckpt_path, device=device)
    sd = strip_module_prefix(sd)

    # 根据 checkpoint 形状自动判断结构
    # 旧版 5 层 CNN 的关键特征：
    #   net.6.weight -> [256, 128, 5]
    #   head.3.weight / 或最后线性层输入维度是 512
    use_v1 = False

    if "net.6.weight" in sd:
        w = sd["net.6.weight"]
        if len(w.shape) == 3 and w.shape[0] == width * 2 and w.shape[1] == width and w.shape[2] == 5:
            use_v1 = True

    if use_v1:
        print("[IMU] detected old 5-layer CNN checkpoint")
        model = IMU1DCNN_V1(in_ch=in_ch, num_classes=num_classes, width=width, dropout=dropout)
    else:
        print("[IMU] detected current 3-layer CNN checkpoint")
        model = IMU1DCNN_V2(in_ch=in_ch, num_classes=num_classes, width=width, dropout=dropout)

    msg = model.load_state_dict(sd, strict=False)
    print("[IMU] load_state_dict:", msg)
    model.eval().to(device)
    return model


# =========================
# 5. main infer
# =========================
@torch.no_grad()
def run_debug(args):
    set_seed(args.seed)
    ensure_dir(args.out_dir)

    device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
    print("device:", device)

    df = pd.read_csv(args.pairs_csv)

    # 类别映射：直接用 pairs.csv 中实际出现的标签
    labels = sorted(df["label"].unique().tolist())
    label_to_idx = {x: i for i, x in enumerate(labels)}
    idx_to_label = {i: x for x, i in label_to_idx.items()}

    print("labels:", labels)
    print("num_classes:", len(labels))

    # ---------- IMU mean/std ----------
    imu_mean, imu_std = None, None
    if args.imu_mean and args.imu_std and os.path.exists(args.imu_mean) and os.path.exists(args.imu_std):
        imu_mean = np.load(args.imu_mean)
        imu_std = np.load(args.imu_std)
        print("Loaded IMU mean/std:", imu_mean.shape, imu_std.shape)
    else:
        print("No external IMU mean/std found, fallback to per-sample normalization.")

    # ---------- data ----------
    img_tf = build_image_transform(args.img_size)
    ds = SmallFusionDebugDataset(
        pairs_csv=args.pairs_csv,
        label_to_idx=label_to_idx,
        img_tf=img_tf,
        imu_mean=imu_mean,
        imu_std=imu_std,
        imu_len=args.imu_len
    )
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_fn)

    # ---------- model ----------
    # 图像类别数如果和当前小样本类别数不一致，你可以手工指定 label_map_json 做统一
    img_model = build_image_model(
        img_arch=args.img_arch,
        num_classes=len(labels),
        ckpt_path=args.img_ckpt,
        device=device
    )

    imu_model = build_imu_model(
        num_classes=len(labels),
        ckpt_path=args.imu_ckpt,
        device=device,
        width=args.imu_width,
        dropout=args.imu_dropout,
        in_ch=9
    )

    # ---------- infer ----------
    rows = []
    img_correct = 0
    imu_correct = 0
    fuse_correct = 0
    total = 0

    for imgs, imus, ys, metas in loader:
        imgs = imgs.to(device)
        imus = imus.to(device)
        ys = ys.to(device)

        img_logits = img_model(imgs)
        imu_logits = imu_model(imus)

        img_prob = F.softmax(img_logits, dim=1).cpu().numpy()
        imu_prob = F.softmax(imu_logits, dim=1).cpu().numpy()

        fuse_prob = args.alpha * img_prob + (1.0 - args.alpha) * imu_prob

        img_pred = np.argmax(img_prob, axis=1)
        imu_pred = np.argmax(imu_prob, axis=1)
        fuse_pred = np.argmax(fuse_prob, axis=1)
        ys_np = ys.cpu().numpy()

        for i in range(len(metas)):
            gt_idx = int(ys_np[i])
            gt_name = idx_to_label[gt_idx]

            img_name = idx_to_label[int(img_pred[i])]
            imu_name = idx_to_label[int(imu_pred[i])]
            fuse_name = idx_to_label[int(fuse_pred[i])]

            img_ok = (img_pred[i] == gt_idx)
            imu_ok = (imu_pred[i] == gt_idx)
            fuse_ok = (fuse_pred[i] == gt_idx)

            img_correct += int(img_ok)
            imu_correct += int(imu_ok)
            fuse_correct += int(fuse_ok)
            total += 1

            rows.append({
                "sample_id": metas[i]["sample_id"],
                "gt_label": gt_name,

                "img_pred": img_name,
                "img_conf": float(np.max(img_prob[i])),
                "img_ok": bool(img_ok),
                "img_top3": json.dumps(topk_info(img_prob[i], idx_to_label, 3), ensure_ascii=False),

                "imu_pred": imu_name,
                "imu_conf": float(np.max(imu_prob[i])),
                "imu_ok": bool(imu_ok),
                "imu_top3": json.dumps(topk_info(imu_prob[i], idx_to_label, 3), ensure_ascii=False),

                "fuse_pred": fuse_name,
                "fuse_conf": float(np.max(fuse_prob[i])),
                "fuse_ok": bool(fuse_ok),

                "img_imu_same": bool(img_name == imu_name),
                "image_path": metas[i]["image_path"],
                "imu_path": metas[i]["imu_path"],
            })

    out_csv = os.path.join(args.out_dir, "debug_results.csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

    summary = {
        "num_samples": total,
        "labels": labels,
        "alpha_image": args.alpha,
        "image_acc": img_correct / max(total, 1),
        "imu_acc": imu_correct / max(total, 1),
        "fusion_acc": fuse_correct / max(total, 1),
        "num_img_imu_agree": int(sum(r["img_imu_same"] for r in rows)),
        "num_img_imu_disagree": int(sum((not r["img_imu_same"]) for r in rows)),
        "out_csv": out_csv
    }

    with open(os.path.join(args.out_dir, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print("\n===== SUMMARY =====")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"detail csv saved to: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pairs_csv", type=str, required=True)

    ap.add_argument("--img_ckpt", type=str, required=True)
    ap.add_argument("--imu_ckpt", type=str, required=True)

    ap.add_argument("--img_arch", type=str, default="tf_efficientnet_b0")
    ap.add_argument("--img_size", type=int, default=224)

    ap.add_argument("--imu_len", type=int, default=256)
    ap.add_argument("--imu_width", type=int, default=128)
    ap.add_argument("--imu_dropout", type=float, default=0.2)

    ap.add_argument("--imu_mean", type=str, default="")
    ap.add_argument("--imu_std", type=str, default="")

    ap.add_argument("--alpha", type=float, default=0.7,
                    help="fusion prob = alpha*img + (1-alpha)*imu")
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out_dir", type=str, default="/root/mqx/LSMRT/debug_fusion_smallset_out")
    args = ap.parse_args()

    run_debug(args)


if __name__ == "__main__":
    main()