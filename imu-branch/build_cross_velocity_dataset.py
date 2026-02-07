import os, re, json, argparse
import numpy as np
from pathlib import Path

import sys, types, io, pickle

# =========================
# 0) Patch legacy pandas pickle paths (optional but safe)
# =========================
def patch_pandas_tseries_index():
    try:
        import pandas as pd
        from pandas.core.indexes.base import Index
        try:
            from pandas.core.indexes.datetimes import DatetimeIndex
        except Exception:
            DatetimeIndex = None

        tseries_mod = types.ModuleType("pandas.tseries")
        index_mod = types.ModuleType("pandas.tseries.index")

        index_mod.Index = Index
        if DatetimeIndex is not None:
            index_mod.DatetimeIndex = DatetimeIndex

        def _new_DatetimeIndex(data=None, freq=None, tz=None, name=None, **kwargs):
            if DatetimeIndex is not None:
                try:
                    return DatetimeIndex(data, freq=freq, tz=tz, name=name)
                except Exception:
                    try:
                        return DatetimeIndex(pd.to_datetime(data), tz=tz, name=name)
                    except Exception:
                        pass
            try:
                return Index(data, name=name)
            except Exception:
                return Index([], name=name)

        index_mod._new_DatetimeIndex = _new_DatetimeIndex
        sys.modules["pandas.tseries"] = tseries_mod
        sys.modules["pandas.tseries.index"] = index_mod
    except Exception:
        pass

patch_pandas_tseries_index()

class CompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == "pandas.tseries.index":
            mod = sys.modules.get("pandas.tseries.index")
            if mod is not None and hasattr(mod, name):
                return getattr(mod, name)
        return super().find_class(module, name)

def safe_pickle_load(pkl_path: str):
    with open(pkl_path, "rb") as f:
        raw = f.read()
    try:
        return CompatUnpickler(io.BytesIO(raw), fix_imports=True, encoding="latin1", errors="ignore").load()
    except Exception:
        return CompatUnpickler(io.BytesIO(raw), fix_imports=True, encoding="bytes", errors="ignore").load()

# =========================
# 1) Robust array collector (supports pandas DataFrame/Series)
# =========================
def collect_arrays(obj, arrays, max_nodes=20000):
    stack = [obj]
    visited = set()
    nodes = 0

    try:
        import pandas as pd
        has_pd = True
    except Exception:
        has_pd = False
        pd = None

    while stack and nodes < max_nodes:
        cur = stack.pop()
        nodes += 1

        oid = id(cur)
        if oid in visited:
            continue
        visited.add(oid)

        # pandas DataFrame/Series -> ndarray
        if has_pd:
            try:
                if isinstance(cur, pd.DataFrame):
                    arrays.append(np.asarray(cur.values))
                    continue
                if isinstance(cur, pd.Series):
                    arrays.append(np.asarray(cur.values))
                    continue
            except Exception:
                pass

        if isinstance(cur, np.ndarray):
            arrays.append(cur)
            continue

        if isinstance(cur, (list, tuple)):
            for x in cur:
                stack.append(x)
            continue

        if isinstance(cur, dict):
            for k, v in cur.items():
                stack.append(k)
                stack.append(v)
            continue

        if hasattr(cur, "__dict__"):
            stack.append(cur.__dict__)

# =========================
# 2) Pick best IMU-like array
# =========================
def score_imu(arr):
    if not isinstance(arr, np.ndarray):
        return -1e18
    if arr.size == 0 or arr.dtype == object:
        return -1e18
    if arr.ndim != 2:
        return -1e18

    # ensure numeric
    try:
        a = arr.astype(np.float32, copy=False)
    except Exception:
        return -1e18

    T, C = a.shape
    # allow (C,T)
    if T < C:
        T, C = C, T

    if T < 200:
        return -1e18

    s = 0.0
    # prefer channel around 9
    s -= abs(C - 9) * 2.0
    if 2 <= C <= 24:
        s += 5.0
    if C in (6, 9, 12):
        s += 3.0

    # prefer long sequence
    s += min(T / 1000.0, 50.0)

    return s

def pick_best_imu_array(arrays):
    best = None
    best_s = -1e18
    for a in arrays:
        sc = score_imu(a)
        if sc > best_s:
            best_s = sc
            best = a
    return best

def normalize_TC(arr):
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("need 2D")
    # cast float32
    a = a.astype(np.float32, copy=False)
    # ensure (T,C) where T > C
    if a.shape[0] < a.shape[1]:
        a = a.T
    return a

def window_cut(x, win=256, stride=128):
    T, C = x.shape
    if T < win:
        return []
    out = []
    for s in range(0, T - win + 1, stride):
        out.append(x[s:s+win])
    return out

# =========================
# 3) Parse helpers
# =========================
def parse_texture_id(p):
    m = re.search(r"texture_(\d+)", p.replace("\\", "/"))
    if not m:
        raise ValueError("no texture id in path")
    return int(m.group(1)) - 1  # 0-based

def parse_velocity(p):
    m = re.search(r"pickles_(\d+)", p.replace("\\", "/"))
    if not m:
        raise ValueError("no velocity in path")
    return int(m.group(1))

def parse_trial_id(p):
    base = os.path.basename(p)
    return os.path.splitext(base)[0]  # imu0101

# =========================
# 4) Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help=".../data/multimodal-tactile-texture-dataset")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--train_vel", type=str, default="30,35")
    ap.add_argument("--val_vel", type=str, default="35")
    ap.add_argument("--test_vel", type=str, default="40")
    ap.add_argument("--debug_first", type=int, default=5,
                    help="print first N picked shapes for sanity")
    args = ap.parse_args()

    root = args.root
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    train_vel = set(int(x) for x in args.train_vel.split(",") if x.strip())
    val_vel   = set(int(x) for x in args.val_vel.split(",") if x.strip())
    test_vel  = set(int(x) for x in args.test_vel.split(",") if x.strip())

    pkls = [str(p) for p in Path(root).rglob("full_imu/*.pkl")]
    print("Found full_imu pkls:", len(pkls))

    trial_rows = []
    trial_windows = []
    win_count = 0
    dbg = 0

    for p in pkls:
        vel = parse_velocity(p)
        y = parse_texture_id(p)
        trial = parse_trial_id(p)
        trial_key = f"v{vel}_tex{y:02d}_{trial}"

        obj = safe_pickle_load(p)
        arrays = []
        collect_arrays(obj, arrays)

        best = pick_best_imu_array(arrays)
        if best is None:
            continue

        x = normalize_TC(best)
        x = np.nan_to_num(x)
        x[~np.isfinite(x)] = 0.0

        wins = window_cut(x, win=args.win, stride=args.stride)
        if not wins:
            continue

        if dbg < args.debug_first:
            print("[DBG]", os.path.basename(p), "vel=", vel, "y=", y, "best_shape=", x.shape)
            dbg += 1

        trial_windows.append((trial_key, wins, y, vel))
        trial_rows.append({"trial": trial_key, "vel": vel, "y": y, "nwin": len(wins)})
        win_count += len(wins)

    print("Trials kept:", len(trial_rows))
    print("Total windows:", win_count)

    if len(trial_rows) == 0 or win_count == 0:
        raise RuntimeError(
            "No trials/windows were produced. "
            "This usually means pickle contents were not decoded into IMU arrays. "
            "Check pandas version/env, or add debug prints."
        )

    # save trials meta
    import csv
    trials_csv = os.path.join(out_dir, "trials.csv")
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["trial", "vel", "y", "nwin"])
        w.writeheader()
        for r in trial_rows:
            w.writerow(r)
    print("Saved:", trials_csv)

    # write memmaps
    X_path = os.path.join(out_dir, "X.npy")
    y_path = os.path.join(out_dir, "y.npy")
    X = np.lib.format.open_memmap(X_path, mode="w+", dtype=np.float32,
                                  shape=(win_count, args.win, 9))
    y_arr = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.int64,
                                      shape=(win_count,))

    cursor = 0
    trial_ranges = {}
    for trial_key, wins, y0, vel in trial_windows:
        n = len(wins)
        wstack = np.stack(wins, axis=0)  # (n, win, C)

        # If C != 9, pad/crop to 9 (稳妥起见)
        C = wstack.shape[2]
        if C < 9:
            pad = np.zeros((n, args.win, 9 - C), dtype=np.float32)
            wstack = np.concatenate([wstack, pad], axis=2)
        elif C > 9:
            wstack = wstack[:, :, :9]

        X[cursor:cursor+n] = wstack
        y_arr[cursor:cursor+n] = y0
        trial_ranges[trial_key] = [cursor, cursor+n]
        cursor += n

    assert cursor == win_count
    print("Write done. N =", win_count)

    # build indices by velocity sets (trial-level)
    train_idx, val_idx, test_idx = [], [], []
    for r in trial_rows:
        t = r["trial"]
        vel = int(r["vel"])
        s, e = trial_ranges[t]
        idx = np.arange(s, e, dtype=np.int64)
        if vel in train_vel:
            train_idx.append(idx)
        elif vel in val_vel:
            val_idx.append(idx)
        elif vel in test_vel:
            test_idx.append(idx)

    train_idx = np.concatenate(train_idx, axis=0) if train_idx else np.zeros((0,), np.int64)
    val_idx   = np.concatenate(val_idx, axis=0) if val_idx else np.zeros((0,), np.int64)
    test_idx  = np.concatenate(test_idx, axis=0) if test_idx else np.zeros((0,), np.int64)

    np.save(os.path.join(out_dir, "train_idx.npy"), train_idx)
    np.save(os.path.join(out_dir, "val_idx.npy"), val_idx)
    np.save(os.path.join(out_dir, "test_idx.npy"), test_idx)

    split_json = os.path.join(out_dir, "split_by_velocity.json")
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump({
            "train_vel": sorted(list(train_vel)),
            "val_vel": sorted(list(val_vel)),
            "test_vel": sorted(list(test_vel)),
            "train_windows": int(train_idx.shape[0]),
            "val_windows": int(val_idx.shape[0]),
            "test_windows": int(test_idx.shape[0]),
        }, f, indent=2)
    print("Saved:", split_json)
    print("train windows:", train_idx.shape[0], "val windows:", val_idx.shape[0], "test windows:", test_idx.shape[0])
    print("=== DONE ===")
    print("OUT_DIR:", out_dir)

if __name__ == "__main__":
    main()
