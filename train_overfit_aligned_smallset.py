#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Overfit sanity-check on aligned small set (img + imu).

目标：
1) 用 aligned-data 的小样本(如 8 对)做过拟合训练；
2) 检查数据读取、配对、标签映射、模型与预处理是否可学习；
3) 支持 imu / img / fusion 三种模式。

说明：
- 标签由文件名前缀推断，如 carton01 -> carton
- 该脚本默认从头训练轻量网络，目的是“是否能记住训练集”
- 如果你想用预训练初始化，可在后续再加加载 ckpt 的逻辑
"""

import os
import re
import glob
import json
import random
import argparse
from dataclasses import dataclass
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# -------------------------
# Utils
# -------------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def infer_label_from_key(key: str) -> str:
    m = re.match(r"([A-Za-z_]+)\d+$", key)
    return m.group(1) if m else key


def stem_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


@dataclass
class PairItem:
    key: str
    img_path: str
    imu_path: str
    label_name: str
    label_id: int


def discover_pairs(aligned_root: str) -> Tuple[List[Tuple[str, str, str]], Dict]:
    img_dir = os.path.join(aligned_root, "img")
    imu_dir = os.path.join(aligned_root, "imu")

    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    imu_files = sorted(glob.glob(os.path.join(imu_dir, "*.csv")))

    img_map = {stem_key(p): p for p in img_files}
    imu_map = {stem_key(p): p for p in imu_files}

    img_keys = set(img_map.keys())
    imu_keys = set(imu_map.keys())

    both = sorted(list(img_keys & imu_keys))
    only_img = sorted(list(img_keys - imu_keys))
    only_imu = sorted(list(imu_keys - img_keys))

    pairs = [(k, img_map[k], imu_map[k]) for k in both]
    summary = {
        "n_img": len(img_files),
        "n_imu": len(imu_files),
        "n_pairs": len(pairs),
        "only_img": only_img,
        "only_imu": only_imu,
    }
    return pairs, summary


# -------------------------
# Dataset
# -------------------------
class AlignedSmallsetDataset(Dataset):
    def __init__(
        self,
        items: List[PairItem],
        image_size: int = 224,
        imu_norm: bool = True,
        imu_mean: np.ndarray = None,
        imu_std: np.ndarray = None,
    ):
        self.items = items
        self.image_size = image_size
        self.imu_norm = imu_norm
        self.imu_mean = imu_mean
        self.imu_std = imu_std

        self.img_tfm = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.items)

    def _load_img(self, p: str):
        img = Image.open(p).convert("RGB")
        return self.img_tfm(img)

    def _load_imu(self, p: str):
        # 读取数值列，期望 (T, C)
        df = pd.read_csv(p)
        vals = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        if vals.size == 0:
            raise ValueError(f"IMU 无数值列: {p}")

        # 对齐到 C=9（不足补0，超出截断）
        C_expect = 9
        if vals.shape[1] > C_expect:
            vals = vals[:, :C_expect]
        elif vals.shape[1] < C_expect:
            pad = np.zeros((vals.shape[0], C_expect - vals.shape[1]), dtype=np.float32)
            vals = np.concatenate([vals, pad], axis=1)

        if self.imu_norm and (self.imu_mean is not None) and (self.imu_std is not None):
            vals = (vals - self.imu_mean[None, :]) / (self.imu_std[None, :] + 1e-6)

        # Conv1d input: (C, T)
        x = torch.from_numpy(vals).float().transpose(0, 1).contiguous()
        return x

    def __getitem__(self, i):
        it = self.items[i]
        x_img = self._load_img(it.img_path)         # [3,H,W]
        x_imu = self._load_imu(it.imu_path)         # [9,T]
        y = torch.tensor(it.label_id, dtype=torch.long)
        return x_img, x_imu, y, it.key

def collate_pad_imu(batch):
    """
    batch: [(x_img[3,H,W], x_imu[9,T], y, key), ...]
    把 x_imu 在时间维右侧补零到同一长度
    """
    x_imgs, x_imus, ys, keys = zip(*batch)

    x_imgs = torch.stack(x_imgs, dim=0)  # [B,3,H,W]
    ys = torch.stack(ys, dim=0)          # [B]

    c = x_imus[0].shape[0]
    t_max = max(x.shape[1] for x in x_imus)

    x_imu_pad = torch.zeros((len(x_imus), c, t_max), dtype=x_imus[0].dtype)
    for i, x in enumerate(x_imus):
        t = x.shape[1]
        x_imu_pad[i, :, :t] = x

    return x_imgs, x_imu_pad, ys, list(keys)

# -------------------------
# Models
# -------------------------
class TinyImageCNN(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, num_classes)

    def forward(self, x):
        z = self.features(x).flatten(1)
        return self.head(z)


class IMU1DCNN(nn.Module):
    """与你仓库旧版 train_imu_trial_split_cli.py 兼容风格"""
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
        return self.head(self.net(x))


class FusionModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.img = TinyImageCNN(num_classes=num_classes)
        self.imu = IMU1DCNN(in_ch=9, num_classes=num_classes, width=64, dropout=0.1)

    def forward(self, x_img, x_imu):
        li = self.img(x_img)
        lm = self.imu(x_imu)
        return 0.5 * li + 0.5 * lm, li, lm


# -------------------------
# Train / Eval
# -------------------------
@torch.no_grad()
def eval_trainset(mode, model, loader, device):
    model.eval()
    ys, ps = [], []
    rows = []
    for x_img, x_imu, y, key in loader:
        x_img = x_img.to(device)
        x_imu = x_imu.to(device)
        y = y.to(device)

        if mode == "img":
            logits = model(x_img)
        elif mode == "imu":
            logits = model(x_imu)
        else:
            logits, li, lm = model(x_img, x_imu)

        pred = torch.argmax(logits, dim=1)
        ys.extend(y.cpu().tolist())
        ps.extend(pred.cpu().tolist())
        for i in range(len(key)):
            rows.append({
                "key": key[i],
                "y_true": int(y[i].item()),
                "y_pred": int(pred[i].item()),
                "correct": bool(pred[i].item() == y[i].item())
            })

    ys = np.array(ys)
    ps = np.array(ps)
    acc = float((ys == ps).mean()) if len(ys) > 0 else 0.0
    return acc, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_root", type=str, default="/root/mqx/LSMRT/data/aligned-data")
    ap.add_argument("--out_dir", type=str, default="/root/mqx/LSMRT/data/aligned-data/check_results/overfit")
    ap.add_argument("--mode", type=str, default="fusion", choices=["img", "imu", "fusion"])
    ap.add_argument("--epochs", type=int, default=400)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--image_size", type=int, default=224)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    set_seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    pairs_raw, pair_summary = discover_pairs(args.aligned_root)
    if len(pairs_raw) == 0:
        raise RuntimeError("没有找到 img/csv 配对样本")

    class_names = sorted(list({infer_label_from_key(k) for k, _, _ in pairs_raw}))
    cls2id = {c: i for i, c in enumerate(class_names)}

    items = []
    for k, ip, mp in pairs_raw:
        cname = infer_label_from_key(k)
        items.append(PairItem(
            key=k,
            img_path=ip,
            imu_path=mp,
            label_name=cname,
            label_id=cls2id[cname]
        ))

    # 用当前小样本计算 imu mean/std（用于更稳定过拟合）
    all_imu = []
    for it in items:
        df = pd.read_csv(it.imu_path)
        vals = df.select_dtypes(include=[np.number]).values.astype(np.float32)
        if vals.shape[1] > 9:
            vals = vals[:, :9]
        elif vals.shape[1] < 9:
            vals = np.concatenate([vals, np.zeros((vals.shape[0], 9 - vals.shape[1]), dtype=np.float32)], axis=1)
        all_imu.append(vals)
    cat = np.concatenate(all_imu, axis=0)  # [sumT, 9]
    imu_mean = cat.mean(axis=0)
    imu_std = cat.std(axis=0) + 1e-6

    ds = AlignedSmallsetDataset(
        items=items,
        image_size=args.image_size,
        imu_norm=True,
        imu_mean=imu_mean,
        imu_std=imu_std,
    )
    loader = DataLoader(
        ds,
        batch_size=args.batch,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_pad_imu
    )

    device = args.device
    num_classes = len(class_names)

    if args.mode == "img":
        model = TinyImageCNN(num_classes).to(device)
    elif args.mode == "imu":
        model = IMU1DCNN(in_ch=9, num_classes=num_classes, width=64, dropout=0.1).to(device)
    else:
        model = FusionModel(num_classes).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    crit = nn.CrossEntropyLoss()

    best_acc = -1.0
    history = []

    for ep in range(1, args.epochs + 1):
        model.train()
        losses = []

        for x_img, x_imu, y, key in loader:
            x_img = x_img.to(device)
            x_imu = x_imu.to(device)
            y = y.to(device)

            opt.zero_grad(set_to_none=True)

            if args.mode == "img":
                logits = model(x_img)
            elif args.mode == "imu":
                logits = model(x_imu)
            else:
                logits, li, lm = model(x_img, x_imu)

            loss = crit(logits, y)
            loss.backward()
            opt.step()
            losses.append(float(loss.item()))

        train_acc, rows = eval_trainset(args.mode, model, loader, device)
        loss_ep = float(np.mean(losses)) if losses else 0.0
        best_acc = max(best_acc, train_acc)

        history.append({
            "epoch": ep,
            "loss": loss_ep,
            "train_acc": train_acc,
            "best_acc": best_acc
        })

        if ep % 20 == 0 or ep == 1:
            print(f"[{ep:03d}] loss={loss_ep:.4f} train_acc={train_acc:.4f} best={best_acc:.4f}")

        # 提前停止：完全记住
        if train_acc >= 0.999:
            print(f"[EARLY STOP] epoch={ep}, train_acc={train_acc:.4f}")
            break

    # final eval details
    final_acc, final_rows = eval_trainset(args.mode, model, loader, device)

    # save
    torch.save({
        "model": model.state_dict(),
        "mode": args.mode,
        "class_names": class_names
    }, os.path.join(args.out_dir, f"overfit_{args.mode}.pt"))

    with open(os.path.join(args.out_dir, f"overfit_{args.mode}_history.json"), "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    pd.DataFrame(final_rows).to_csv(
        os.path.join(args.out_dir, f"overfit_{args.mode}_predictions.csv"),
        index=False, encoding="utf-8"
    )

    summary = {
        "mode": args.mode,
        "pair_summary": pair_summary,
        "num_samples": len(items),
        "class_names": class_names,
        "final_train_acc": final_acc,
        "best_train_acc": best_acc,
        "epochs_ran": len(history),
        "criterion": "能否在训练集(小样本)达到接近100%准确率"
    }
    with open(os.path.join(args.out_dir, f"overfit_{args.mode}_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== DONE ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"outputs -> {args.out_dir}")


if __name__ == "__main__":
    main()