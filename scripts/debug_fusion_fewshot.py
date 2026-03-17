#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import glob
import argparse
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


@dataclass
class PairItem:
    key: str
    img_path: str
    imu_path: str


def stem_key(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def discover_pairs(aligned_root: str) -> Tuple[List[PairItem], Dict]:
    img_dir = os.path.join(aligned_root, "img")
    imu_dir = os.path.join(aligned_root, "imu")
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    imu_files = sorted(glob.glob(os.path.join(imu_dir, "*.csv")))

    img_map = {stem_key(p): p for p in img_files}
    imu_map = {stem_key(p): p for p in imu_files}

    both = sorted(list(set(img_map) & set(imu_map)))
    only_img = sorted(list(set(img_map) - set(imu_map)))
    only_imu = sorted(list(set(imu_map) - set(img_map)))

    pairs = [PairItem(k, img_map[k], imu_map[k]) for k in both]
    summary = {
        "n_img": len(img_files),
        "n_imu": len(imu_files),
        "n_pairs": len(pairs),
        "only_img": only_img,
        "only_imu": only_imu,
    }
    return pairs, summary


def infer_label_from_key(key: str) -> str:
    m = re.match(r"([A-Za-z_]+)\d+$", key)
    return m.group(1) if m else key


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - np.max(x))
    return e / (e.sum() + 1e-12)


def load_ckpt_raw(ckpt_path: str, device: str):
    return torch.load(ckpt_path, map_location=device)


def extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            sd = ckpt_obj["state_dict"]
        elif "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            sd = ckpt_obj["model"]
        else:
            sd = ckpt_obj
    else:
        sd = ckpt_obj

    new_sd = {}
    for k, v in sd.items():
        nk = k[7:] if k.startswith("module.") else k
        new_sd[nk] = v
    return new_sd


# ========= IMU: 与 train_imu_trial_split_cli.py 一致 =========
class IMU1DCNN(nn.Module):
    """
    与 imu-branch/train_imu_trial_split_cli.py 旧版结构匹配：
    net: conv7 -> conv5 -> conv5 -> conv3 -> conv3
    head: GAP -> Flatten -> Dropout -> Linear
    """
    def __init__(self, in_ch=9, num_classes=12, width=128, dropout=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, width, kernel_size=7, stride=2, padding=3, bias=False),   # net.0
            nn.BatchNorm1d(width),                                                      # net.1
            nn.ReLU(inplace=True),                                                      # net.2

            nn.Conv1d(width, width, kernel_size=5, stride=1, padding=2, bias=False),   # net.3
            nn.BatchNorm1d(width),                                                      # net.4
            nn.ReLU(inplace=True),                                                      # net.5

            nn.Conv1d(width, width * 2, kernel_size=5, stride=2, padding=2, bias=False), # net.6
            nn.BatchNorm1d(width * 2),                                                    # net.7
            nn.ReLU(inplace=True),                                                        # net.8

            nn.Conv1d(width * 2, width * 2, kernel_size=3, stride=1, padding=1, bias=False), # net.9
            nn.BatchNorm1d(width * 2),                                                        # net.10
            nn.ReLU(inplace=True),                                                            # net.11

            nn.Conv1d(width * 2, width * 4, kernel_size=3, stride=2, padding=1, bias=False), # net.12
            nn.BatchNorm1d(width * 4),                                                        # net.13
            nn.ReLU(inplace=True),                                                            # net.14
        )
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # head.0
            nn.Flatten(),             # head.1
            nn.Dropout(p=dropout),    # head.2
            nn.Linear(width * 4, num_classes),  # head.3
        )

    def forward(self, x):
        x = self.net(x)
        x = self.head(x)
        return x


def infer_imu_hparams_from_sd(sd: Dict[str, torch.Tensor]):
    conv_w = sd.get("net.0.weight", None)
    if conv_w is None or conv_w.ndim != 3:
        raise KeyError("未找到 net.0.weight，无法推断 in_ch/width")

    in_ch = int(conv_w.shape[1])
    width = int(conv_w.shape[0])

    if "head.3.weight" in sd and sd["head.3.weight"].ndim == 2:
        num_classes = int(sd["head.3.weight"].shape[0])
        fc_key = "head.3.weight"
    else:
        # 回退：最后一个2D权重
        fc_candidates = [(k, v) for k, v in sd.items() if isinstance(v, torch.Tensor) and v.ndim == 2]
        if not fc_candidates:
            raise KeyError("未找到分类层2D权重")
        fc_candidates.sort(key=lambda x: x[0])
        fc_key, fc_w = fc_candidates[-1]
        num_classes = int(fc_w.shape[0])

    return in_ch, width, num_classes, "net.0.weight", fc_key

def build_imu_model_from_ckpt(imu_ckpt: str, device: str):
    raw = load_ckpt_raw(imu_ckpt, device)
    sd = extract_state_dict(raw)

    in_ch, width, num_classes, conv_k, fc_k = infer_imu_hparams_from_sd(sd)
    model = IMU1DCNN(in_ch=in_ch, num_classes=num_classes, width=width, dropout=0.2).to(device).eval()

    # 用 strict=False，避免命名细节差异导致硬失败；同时把关键信息打印出来
    missing, unexpected = model.load_state_dict(sd, strict=False)

    return model, {
        "in_ch": in_ch,
        "width": width,
        "num_classes": num_classes,
        "conv_key_used": conv_k,
        "fc_key_used": fc_k,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def load_imu_series(imu_csv_path: str, expect_c: int) -> np.ndarray:
    df = pd.read_csv(imu_csv_path)
    vals = df.select_dtypes(include=[np.number]).values.astype(np.float32)  # [T,C]
    if vals.size == 0:
        raise ValueError(f"IMU 文件无数值列: {imu_csv_path}")

    c = vals.shape[1]
    if c > expect_c:
        vals = vals[:, :expect_c]
    elif c < expect_c:
        pad = np.zeros((vals.shape[0], expect_c - c), dtype=np.float32)
        vals = np.concatenate([vals, pad], axis=1)

    return vals  # [T, C]


# ========= IMAGE: 尽量适配 effb0 =========
class FallbackImageModel(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.net(x)


def build_image_model_from_ckpt(img_ckpt: str, device: str):
    raw = load_ckpt_raw(img_ckpt, device)
    sd = extract_state_dict(raw)

    num_classes = None
    for k in ["classifier.weight", "fc.weight", "head.fc.weight", "head.weight"]:
        if k in sd:
            num_classes = int(sd[k].shape[0])
            break
    if num_classes is None:
        # 回退为4类（你的小样本目前是 carton/desk/mousepad/tile）
        num_classes = 4

    model = None
    err = None

    try:
        import timm
        model = timm.create_model("efficientnet_b0", pretrained=False, num_classes=num_classes)
    except Exception as e:
        err = str(e)
        model = FallbackImageModel(num_classes)

    missing, unexpected = model.load_state_dict(sd, strict=False)
    model = model.to(device).eval()

    return model, {
        "num_classes": num_classes,
        "build_error": err,
        "missing": list(missing),
        "unexpected": list(unexpected),
    }


def load_img_tensor(img_path: str, image_size: int = 224) -> torch.Tensor:
    tfm = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    img = Image.open(img_path).convert("RGB")
    return tfm(img).unsqueeze(0)


@torch.no_grad()
def run_image_infer(model: nn.Module, img_path: str, device: str):
    x = load_img_tensor(img_path).to(device)
    logits = model(x).squeeze(0).detach().cpu().numpy()
    prob = softmax_np(logits)
    pred_idx = int(np.argmax(prob))
    return pred_idx, float(prob[pred_idx]), prob.tolist()


@torch.no_grad()
def run_imu_infer(model: nn.Module, imu_path: str, device: str, expect_c: int, mean=None, std=None):
    x_tc = load_imu_series(imu_path, expect_c=expect_c)  # [T,C]
    if mean is not None and std is not None:
        x_tc = (x_tc - mean[None, :]) / (std[None, :] + 1e-6)
    x_ct = x_tc.T  # [C,T]
    x = torch.from_numpy(x_ct).unsqueeze(0).to(device)  # [1,C,T]
    logits = model(x).squeeze(0).detach().cpu().numpy()
    prob = softmax_np(logits)
    pred_idx = int(np.argmax(prob))
    return pred_idx, float(prob[pred_idx]), prob.tolist(), int(x_tc.shape[0]), int(x_tc.shape[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--aligned_root", type=str, default="/root/mqx/LSMRT/data/aligned-data")
    ap.add_argument("--imu_ckpt", type=str, default="/root/mqx/LSMRT/imu-branch/runs/imu_trial_split_cnn/best.pt")
    ap.add_argument("--img_ckpt", type=str, default="/root/mqx/LSMRT/image-branch/runs/effb0_s224_split1_resize/best.pt")
    ap.add_argument("--imu_norm_dir", type=str, default="", help="可选：含 mean.npy/std.npy 的目录")
    ap.add_argument("--out_dir", type=str, default="/root/mqx/LSMRT/data/aligned-data/check_results/fusion_debug")
    ap.add_argument("--w_img", type=float, default=0.5)
    ap.add_argument("--w_imu", type=float, default=0.5)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    pairs, pair_summary = discover_pairs(args.aligned_root)
    if len(pairs) == 0:
        raise RuntimeError(f"未发现 img/csv 配对样本: {args.aligned_root}")

    class_names = sorted(list({infer_label_from_key(p.key) for p in pairs}))
    cls2id = {c: i for i, c in enumerate(class_names)}

    device = args.device

    imu_model, imu_info = build_imu_model_from_ckpt(args.imu_ckpt, device)
    img_model, img_info = build_image_model_from_ckpt(args.img_ckpt, device)

    mean = std = None
    if args.imu_norm_dir:
        mp = os.path.join(args.imu_norm_dir, "mean.npy")
        sp = os.path.join(args.imu_norm_dir, "std.npy")
        if os.path.exists(mp) and os.path.exists(sp):
            mean = np.load(mp).astype(np.float32)
            std = np.load(sp).astype(np.float32)

    results, ok_fusion = [], 0
    for p in pairs:
        gt_name = infer_label_from_key(p.key)
        gt_idx = cls2id[gt_name]

        img_idx, img_conf, img_prob = run_image_infer(img_model, p.img_path, device)
        imu_idx, imu_conf, imu_prob, t_len, c_dim = run_imu_infer(
            imu_model, p.imu_path, device,
            expect_c=imu_info["in_ch"], mean=mean, std=std
        )

        # 若模型类别数与当前小样本类别数不一致，做安全截断对齐
        k = min(len(img_prob), len(imu_prob), len(class_names))
        fused = args.w_img * np.array(img_prob[:k]) + args.w_imu * np.array(imu_prob[:k])
        fused_idx = int(np.argmax(fused))
        fused_conf = float(fused[fused_idx])

        fusion_correct = (fused_idx == gt_idx)
        ok_fusion += int(fusion_correct)

        results.append({
            "key": p.key,
            "gt": gt_name,
            "img_pred_idx": img_idx,
            "img_conf": img_conf,
            "imu_pred_idx": imu_idx,
            "imu_conf": imu_conf,
            "fusion_pred": class_names[fused_idx],
            "fusion_conf": fused_conf,
            "fusion_correct": fusion_correct,
            "imu_T": t_len,
            "imu_C": c_dim,
            "img_path": p.img_path,
            "imu_path": p.imu_path
        })

    df = pd.DataFrame(results)
    csv_path = os.path.join(args.out_dir, "fusion_debug_results.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8")

    summary = {
        "pair_summary": pair_summary,
        "class_names_from_filename": class_names,
        "num_samples": len(results),
        "fusion_correct": ok_fusion,
        "fusion_correct_ratio": ok_fusion / max(len(results), 1),
        "weights": {"w_img": args.w_img, "w_imu": args.w_imu},
        "imu_model_info": imu_info,
        "img_model_info": img_info,
        "output_csv": csv_path
    }

    summary_path = os.path.join(args.out_dir, "fusion_debug_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"[OK] results -> {csv_path}")
    print(f"[OK] summary -> {summary_path}")


if __name__ == "__main__":
    main()