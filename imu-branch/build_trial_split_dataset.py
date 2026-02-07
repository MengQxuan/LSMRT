# imu-branch/build_trial_split_dataset.py
import os
import re
import csv
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd  # legacy env: pandas 0.20.3

# -----------------------
# Config
# -----------------------
ROOT = r"/root/mqx/LSMRT/data/multimodal-tactile-texture-dataset"
OUT_DIR = os.path.join(ROOT, "processed_trial_split")
if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)

WIN = 256
STRIDE = 128
SEED = 42

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
assert abs(TRAIN_RATIO + VAL_RATIO + TEST_RATIO - 1.0) < 1e-9

# -----------------------
# Helpers
# -----------------------
def parse_velocity_from_path(p):
    m = re.search(r"pickles_(\d+)", p.replace("\\", "/"))
    if not m:
        raise ValueError("Cannot parse velocity from path: {}".format(p))
    return int(m.group(1))

def parse_texture_episode_from_name(fname):
    m = re.search(r"imu_?(\d{2})(\d{2})", fname)
    if not m:
        raise ValueError("Cannot parse texture/episode from file name: {}".format(fname))
    tex = int(m.group(1))   # 1..12
    epi = int(m.group(2))   # 1..100
    return tex, epi

def iter_full_imu_pkls(root):
    for p in Path(root).rglob("full_imu/*.pkl"):
        yield str(p)

def df_to_imu_array(df):
    """
    pandas 0.20.3 compatible:
    - use .values instead of .to_numpy
    - prefer named 9 IMU channels if present, otherwise fallback to numeric columns
    """
    cols = ["imu_ax","imu_ay","imu_az","imu_gx","imu_gy","imu_gz","imu_mx","imu_my","imu_mz"]
    has_all = True
    for c in cols:
        if c not in df.columns:
            has_all = False
            break

    if has_all:
        arr = df[cols].values
        return np.asarray(arr, dtype=np.float32)

    # fallback: take numeric columns except timestamp-like
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] != 9:
        raise ValueError("Unexpected numeric columns (need 9), got {}: {}".format(
            num_df.shape[1], list(num_df.columns)
        ))
    return np.asarray(num_df.values, dtype=np.float32)

def window_count(T, win, stride):
    if T < win:
        return 0
    return 1 + (T - win) // stride

def stratified_trial_split(trials, seed=42):
    """
    trials: list of dict with keys: trial_id(int), label(int 0..11), vel(int)
    split by (label, vel)
    """
    rng = random.Random(seed)
    groups = {}
    for t in trials:
        key = (t["label"], t["vel"])
        if key not in groups:
            groups[key] = []
        groups[key].append(t["trial_id"])

    train_trials, val_trials, test_trials = set(), set(), set()
    for key, ids in groups.items():
        rng.shuffle(ids)
        n = len(ids)
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        n_test = n - n_train - n_val

        train_trials.update(ids[:n_train])
        val_trials.update(ids[n_train:n_train+n_val])
        test_trials.update(ids[n_train+n_val:])

        assert (len(ids[:n_train]) + len(ids[n_train:n_train+n_val]) + len(ids[n_train+n_val:])) == n

    return train_trials, val_trials, test_trials

# -----------------------
# Pass 1: scan trials + count windows
# -----------------------
print("[Pass1] scanning full_imu pickle files...")
pkl_paths = sorted(list(iter_full_imu_pkls(ROOT)))
print("Found full_imu pkls:", len(pkl_paths))

trial_rows = []
total_windows = 0

for p in pkl_paths:
    vel = parse_velocity_from_path(p)
    fname = os.path.basename(p)
    tex, epi = parse_texture_episode_from_name(fname)
    label = tex - 1  # 0..11

    df = pd.read_pickle(p)
    x = df_to_imu_array(df)
    T = x.shape[0]
    nw = window_count(T, WIN, STRIDE)
    if nw <= 0:
        continue

    trial_rows.append({
        "path": p,
        "vel": vel,
        "texture": tex,
        "episode": epi,
        "label": label,
        "T": int(T),
        "windows": int(nw),
    })
    total_windows += nw

print("Trials kept:", len(trial_rows))
print("Total windows:", total_windows)

# write trials.csv
trials_csv = os.path.join(OUT_DIR, "trials.csv")
with open(trials_csv, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=["trial_id","vel","texture","episode","label","T","windows","path"])
    w.writeheader()
    for i, row in enumerate(trial_rows):
        out_row = {"trial_id": i}
        out_row.update(row)
        w.writerow(out_row)
print("Saved:", trials_csv)

# -----------------------
# Decide trial splits
# -----------------------
trials_for_split = [{"trial_id": i, "label": r["label"], "vel": r["vel"]} for i, r in enumerate(trial_rows)]
train_trials, val_trials, test_trials = stratified_trial_split(trials_for_split, seed=SEED)

split_json = os.path.join(OUT_DIR, "split_trials.json")
with open(split_json, "w", encoding="utf-8") as f:
    json.dump({
        "seed": SEED,
        "train_trials": sorted(list(train_trials)),
        "val_trials": sorted(list(val_trials)),
        "test_trials": sorted(list(test_trials)),
        "ratios": {"train": TRAIN_RATIO, "val": VAL_RATIO, "test": TEST_RATIO},
        "win": WIN,
        "stride": STRIDE,
    }, f, indent=2)
print("Saved:", split_json)

# -----------------------
# Pass 2: write memmap arrays
# -----------------------
X_path = os.path.join(OUT_DIR, "X.npy")
y_path = os.path.join(OUT_DIR, "y.npy")
trial_id_path = os.path.join(OUT_DIR, "trial.npy")
vel_path = os.path.join(OUT_DIR, "vel.npy")

print("[Pass2] writing memmap arrays...")
X_mm = np.lib.format.open_memmap(X_path, mode="w+", dtype=np.float32, shape=(total_windows, WIN, 9))
y_mm = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64, shape=(total_windows,))
trial_mm = np.lib.format.open_memmap(trial_id_path, mode="w+", dtype=np.int32, shape=(total_windows,))
vel_mm = np.lib.format.open_memmap(vel_path, mode="w+", dtype=np.int16, shape=(total_windows,))

cursor = 0
for tid, row in enumerate(trial_rows):
    df = pd.read_pickle(row["path"])
    x = df_to_imu_array(df)  # (T,9)
    x = np.nan_to_num(x).astype(np.float32)

    T = x.shape[0]
    nw = window_count(T, WIN, STRIDE)
    if nw <= 0:
        continue

    for k in range(nw):
        s = k * STRIDE
        e = s + WIN
        X_mm[cursor] = x[s:e]
        y_mm[cursor] = row["label"]
        trial_mm[cursor] = tid
        vel_mm[cursor] = row["vel"]
        cursor += 1

assert cursor == total_windows, (cursor, total_windows)
print("Write done. N =", total_windows)

# -----------------------
# Build window indices for each split (trial-safe)
# -----------------------
print("[Index] building train/val/test window indices...")
trial_arr = np.asarray(trial_mm)

train_ids = np.array(sorted(list(train_trials)), dtype=np.int32)
val_ids   = np.array(sorted(list(val_trials)), dtype=np.int32)
test_ids  = np.array(sorted(list(test_trials)), dtype=np.int32)

train_mask = np.isin(trial_arr, train_ids)
val_mask   = np.isin(trial_arr, val_ids)
test_mask  = np.isin(trial_arr, test_ids)

train_idx = np.where(train_mask)[0].astype(np.int64)
val_idx   = np.where(val_mask)[0].astype(np.int64)
test_idx  = np.where(test_mask)[0].astype(np.int64)

np.save(os.path.join(OUT_DIR, "train_idx.npy"), train_idx)
np.save(os.path.join(OUT_DIR, "val_idx.npy"), val_idx)
np.save(os.path.join(OUT_DIR, "test_idx.npy"), test_idx)

print("train windows:", len(train_idx), "val windows:", len(val_idx), "test windows:", len(test_idx))
print("=== DONE ===")
print("OUT_DIR:", OUT_DIR)
