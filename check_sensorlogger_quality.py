#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from dataclasses import dataclass, asdict
from typing import List, Dict

import numpy as np
import pandas as pd

CANONICAL = ["SamplingTime", "AccelerationX", "AccelerationY", "AccelerationZ", "GyroX", "GyroY", "GyroZ"]

ALIASES: Dict[str, str] = {
    # time
    "samplingtime": "SamplingTime",
    "time": "SamplingTime",
    "timestamp": "SamplingTime",
    # acc
    "accelerationx": "AccelerationX",
    "accelerationy": "AccelerationY",
    "accelerationz": "AccelerationZ",
    "ax": "AccelerationX",
    "ay": "AccelerationY",
    "az": "AccelerationZ",
    # gyro
    "gyrox": "GyroX",
    "gyroy": "GyroY",
    "gyroz": "GyroZ",
    "wx": "GyroX",
    "wy": "GyroY",
    "wz": "GyroZ",
    "gyrx": "GyroX",
    "gyry": "GyroY",
    "gyrz": "GyroZ",
}

@dataclass
class Report:
    file: str
    valid_rows: int
    duration_s: float
    fs_mean: float
    missing_ratio: float
    has_motion: bool
    motion_ratio: float
    is_valid: bool
    reasons: List[str]

def norm_col(c: str) -> str:
    return c.strip().replace(" ", "").replace("(", "").replace(")", "").replace("/", "").replace("_", "").lower()

def unify_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {}
    for c in df.columns:
        key = norm_col(str(c))
        if key in ALIASES:
            rename_map[c] = ALIASES[key]
    df = df.rename(columns=rename_map)
    return df

def moving_average(x, w):
    if w <= 1:
        return x.copy()
    return np.convolve(x, np.ones(w)/w, mode="same")

def check_file(path: str) -> Report:
    reasons = []
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    df = pd.read_csv(path)
    df = unify_columns(df)

    missing = [c for c in CANONICAL if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}\n当前列: {list(df.columns)}")

    # 转数值
    for c in CANONICAL:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # 去掉全空和关键空
    df = df.dropna(how="all")
    total = len(df)
    df = df.dropna(subset=CANONICAL).copy()

    valid_rows = len(df)
    if valid_rows < 2:
        return Report(path, valid_rows, 0, 0, 1.0, False, 0.0, False, ["有效行过少"])

    t = df["SamplingTime"].values.astype(float)
    ax = df["AccelerationX"].values.astype(float)
    ay = df["AccelerationY"].values.astype(float)
    az = df["AccelerationZ"].values.astype(float)
    gx = df["GyroX"].values.astype(float)
    gy = df["GyroY"].values.astype(float)
    gz = df["GyroZ"].values.astype(float)

    dt = np.diff(t)
    dt_pos = dt[dt > 0]
    duration_s = float(t[-1] - t[0]) if len(t) >= 2 else 0.0
    fs_mean = float(1.0 / np.mean(dt_pos)) if len(dt_pos) else 0.0

    missing_ratio = float((total - valid_rows) / max(total, 1))

    acc_norm = np.sqrt(ax**2 + ay**2 + az**2)
    dt_ref = float(np.median(dt_pos)) if len(dt_pos) else 0.01
    w = max(3, int(round(0.25 / max(dt_ref, 1e-4))))
    baseline = moving_average(acc_norm, w)
    dyn = np.abs(acc_norm - baseline)
    motion_mask = dyn > 0.12
    motion_ratio = float(np.mean(motion_mask))
    has_motion = motion_ratio > 0.03

    is_valid = True
    if valid_rows < 100:
        is_valid = False
        reasons.append(f"有效行太少: {valid_rows}")
    if fs_mean < 80:
        is_valid = False
        reasons.append(f"采样率偏低: {fs_mean:.2f} Hz")
    if duration_s < 1.5:
        is_valid = False
        reasons.append(f"时长太短: {duration_s:.3f} s")
    if missing_ratio > 0.01:
        is_valid = False
        reasons.append(f"缺失比例过高: {missing_ratio:.2%}")
    if not has_motion:
        is_valid = False
        reasons.append("未检测到明显运动段")

    if not reasons:
        reasons.append("通过所有检查")

    return Report(path, valid_rows, duration_s, fs_mean, missing_ratio, has_motion, motion_ratio, is_valid, reasons)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    r = check_file(args.input)
    print(f"File: {r.file}")
    print(f"Valid rows: {r.valid_rows}")
    print(f"Duration: {r.duration_s:.3f}s")
    print(f"Fs(mean): {r.fs_mean:.2f}Hz")
    print(f"Missing ratio: {r.missing_ratio:.2%}")
    print(f"Has motion: {r.has_motion}, motion_ratio={r.motion_ratio:.2%}")
    print(f"Result: {'VALID ✅' if r.is_valid else 'INVALID ❌'}")
    for i, msg in enumerate(r.reasons, 1):
        print(f"{i}. {msg}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(asdict(r), f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()