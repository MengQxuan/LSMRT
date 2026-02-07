# Out-of-Distribution（分布外）
import os
import re
import io
import sys
import csv
import json
import types
import pickle
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# -------------------------
# Pandas legacy pickle compat (optional)
# -------------------------
def patch_pandas_legacy_modules():
    """
    Some legacy pickles reference old pandas internal modules.
    In your case, pickles were readable under py3.6+pandas0.20.3,
    but to be safer, we keep a lightweight patch.
    """
    try:
        import pandas as pd  # noqa
        from pandas.core.indexes.base import Index  # noqa
        try:
            from pandas.core.indexes.datetimes import DatetimeIndex  # noqa
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
        # If pandas not installed, skip
        pass


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

    # Try latin1 first (py2->py3 common), fallback to bytes
    try:
        return CompatUnpickler(
            io.BytesIO(raw),
            fix_imports=True,
            encoding="latin1",
            errors="ignore",
        ).load()
    except Exception:
        return CompatUnpickler(
            io.BytesIO(raw),
            fix_imports=True,
            encoding="bytes",
            errors="ignore",
        ).load()


# -------------------------
# Recursive array collection
# -------------------------
def collect_arrays(obj: Any, arrays: List[np.ndarray], max_nodes: int = 20000):
    stack = [obj]
    visited = set()
    nodes = 0

    try:
        import pandas as pd  # noqa
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
            # try numeric conversion
            try:
                arr = np.asarray(cur)
                if isinstance(arr, np.ndarray) and arr.dtype != object:
                    arrays.append(arr)
                    continue
            except Exception:
                pass
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
            continue


def score_as_imu(arr: np.ndarray) -> float:
    """
    Prefer IMU time-series: (T,9) float-like, T large.
    """
    if not isinstance(arr, np.ndarray):
        return -1.0
    if arr.dtype == object or arr.size == 0:
        return -1.0
    try:
        _ = arr.astype(np.float32, copy=False)
    except Exception:
        return -1.0

    if arr.ndim == 2:
        T, C = arr.shape
        # allow transpose
        if T < C:
            T, C = C, T
        if C != 9:
            return -1.0
        # prefer longer sequences
        return 10.0 + min(T / 1000.0, 20.0)

    # ignore others
    return -1.0


def pick_best_imu_array(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    best = None
    best_s = -1e18
    for a in arrays:
        s = score_as_imu(a)
        if s > best_s:
            best_s = s
            best = a
    return best


def normalize_to_TC(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.ndim != 2:
        raise ValueError("normalize_to_TC expects 2D array, got ndim={}".format(a.ndim))
    if a.shape[0] < a.shape[1]:
        a = a.T
    a = a.astype(np.float32, copy=False)
    return a


def count_windows(T: int, win: int, stride: int) -> int:
    if T < win:
        return 0
    return 1 + (T - win) // stride


# -------------------------
# Path parsing
# -------------------------
def parse_texture_label_from_path(p: str) -> int:
    # .../texture_01/full_imu/imu0101.pkl
    m = re.search(r"texture_(\d+)", p.replace("\\", "/"))
    if not m:
        raise ValueError("Cannot parse texture from path: {}".format(p))
    return int(m.group(1)) - 1  # 0-based


def parse_velocity_from_path(p: str) -> int:
    # .../pickles_30/texture_01/...
    m = re.search(r"/pickles_(\d+)/", p.replace("\\", "/"))
    if not m:
        raise ValueError("Cannot parse velocity from path: {}".format(p))
    return int(m.group(1))


def list_full_imu_pkls(root: str, velocity: int) -> List[str]:
    base = Path(root) / ("pickles_{}".format(velocity))
    # Only full_imu to avoid duplicated signals (imu_ax, imu_ay...)
    return [str(p) for p in base.rglob("full_imu/*.pkl")]


def make_trial_id(pkl_path: str) -> str:
    p = pkl_path.replace("\\", "/")
    vel = parse_velocity_from_path(p)
    tex = parse_texture_label_from_path(p) + 1
    stem = Path(p).stem
    return "vel{}_tex{:02d}_{}".format(vel, tex, stem)


# -------------------------
# Split helpers (trial-level, stratified by texture)
# -------------------------
def stratified_split_trials(trials: List[Dict[str, Any]], val_ratio: float, seed: int):
    """
    trials: list of dict with key 'y'
    Return: train_trials, val_trials
    """
    rng = np.random.RandomState(seed)
    by_cls: Dict[int, List[Dict[str, Any]]] = {}
    for t in trials:
        by_cls.setdefault(int(t["y"]), []).append(t)

    train_out, val_out = [], []
    for y, items in by_cls.items():
        rng.shuffle(items)
        n_val = int(round(len(items) * val_ratio))
        val_out.extend(items[:n_val])
        train_out.extend(items[n_val:])
    rng.shuffle(train_out)
    rng.shuffle(val_out)
    return train_out, val_out


# -------------------------
# Memmap writer
# -------------------------
def create_memmap(path: str, shape: Tuple[int, ...], dtype=np.float32):
    dtype = np.dtype(dtype)
    n_elem = int(np.prod(shape))
    if n_elem <= 0:
        raise RuntimeError("Refuse to create empty memmap: shape={}".format(shape))
    nbytes = n_elem * dtype.itemsize
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.truncate(nbytes)
    return np.memmap(path, dtype=dtype, mode="r+", shape=shape)


# -------------------------
# Main
# -------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=str)
    parser.add_argument("--out_dir", required=True, type=str)
    parser.add_argument("--train_vel", required=True, type=int)
    parser.add_argument("--test_vel", required=True, type=int)
    parser.add_argument("--win", default=256, type=int)
    parser.add_argument("--stride", default=128, type=int)
    parser.add_argument("--val_ratio", default=0.15, type=float)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--debug_first", default=0, type=int)
    args = parser.parse_args()

    patch_pandas_legacy_modules()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # 1) scan pkls
    train_pkls = list_full_imu_pkls(args.root, args.train_vel)
    test_pkls = list_full_imu_pkls(args.root, args.test_vel)
    train_pkls.sort()
    test_pkls.sort()

    print("Found train full_imu pkls:", len(train_pkls), "vel=", args.train_vel)
    print("Found test  full_imu pkls:", len(test_pkls), "vel=", args.test_vel)

    # 2) Build trial list with window counts (Pass1)
    trials_train_vel: List[Dict[str, Any]] = []
    trials_test_vel: List[Dict[str, Any]] = []

    dbg_cnt = 0
    skipped_no_best = 0
    skipped_short = 0
    skipped_other = 0

    def process_one(pkl_path: str) -> Optional[Dict[str, Any]]:
        nonlocal dbg_cnt, skipped_no_best, skipped_short, skipped_other
        try:
            obj = safe_pickle_load(pkl_path)
            arrays: List[np.ndarray] = []
            collect_arrays(obj, arrays)
            arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
            best = pick_best_imu_array(arrays)
            if best is None:
                skipped_no_best += 1
                return None
            x = normalize_to_TC(best)
            if x.shape[1] != 9:
                skipped_other += 1
                return None
            nwin = count_windows(x.shape[0], args.win, args.stride)
            if nwin <= 0:
                skipped_short += 1
                return None

            vel = parse_velocity_from_path(pkl_path.replace("\\", "/"))
            y = parse_texture_label_from_path(pkl_path)
            tid = make_trial_id(pkl_path)

            if args.debug_first > 0 and dbg_cnt < args.debug_first:
                print("[DBG]", os.path.basename(pkl_path),
                      "vel=", vel, "y=", y,
                      "best_shape=", getattr(best, "shape", None),
                      "x_shape=", x.shape,
                      "nwin=", nwin)
                dbg_cnt += 1

            return {
                "trial_id": tid,
                "pkl_path": pkl_path,
                "vel": vel,
                "y": int(y),
                "T": int(x.shape[0]),
                "C": int(x.shape[1]),
                "nwin": int(nwin),
            }
        except Exception:
            skipped_other += 1
            return None

    for p in train_pkls:
        t = process_one(p)
        if t is not None:
            trials_train_vel.append(t)

    for p in test_pkls:
        t = process_one(p)
        if t is not None:
            trials_test_vel.append(t)

    # 3) Split trials (train_vel -> train/val), test_vel -> test
    rng = np.random.RandomState(args.seed)
    train_trials, val_trials = stratified_split_trials(trials_train_vel, args.val_ratio, args.seed)

    test_trials = trials_test_vel[:]  # all test velocity trials

    # 4) Compute total windows per split
    def sum_windows(trials: List[Dict[str, Any]]) -> int:
        s = 0
        for t in trials:
            s += int(t["nwin"])
        return s

    n_train = sum_windows(train_trials)
    n_val = sum_windows(val_trials)
    n_test = sum_windows(test_trials)
    total_windows = n_train + n_val + n_test

    print("[DBG] skipped_no_best =", skipped_no_best,
          "skipped_short =", skipped_short,
          "skipped_other =", skipped_other)
    print("[DBG] trials kept: train_vel:", len(trials_train_vel),
          "test_vel:", len(trials_test_vel))
    print("[DBG] windows: train=", n_train, "val=", n_val, "test=", n_test)
    print("[DBG] total_windows =", total_windows)

    if total_windows <= 0:
        raise RuntimeError(
            "total_windows==0，说明没有产生任何窗口。"
            "通常是：没取到 (T,9) 的 best IMU array 或每条序列都比 win 短。"
        )

    # 5) Write trials.csv
    trials_csv = os.path.join(out_dir, "trials.csv")
    with open(trials_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "trial_id", "vel", "y", "T", "C", "nwin", "pkl_path"])
        for t in train_trials:
            w.writerow(["train", t["trial_id"], t["vel"], t["y"], t["T"], t["C"], t["nwin"], t["pkl_path"]])
        for t in val_trials:
            w.writerow(["val", t["trial_id"], t["vel"], t["y"], t["T"], t["C"], t["nwin"], t["pkl_path"]])
        for t in test_trials:
            w.writerow(["test", t["trial_id"], t["vel"], t["y"], t["T"], t["C"], t["nwin"], t["pkl_path"]])
    print("Saved:", trials_csv)

    split_json = os.path.join(out_dir, "split_by_velocity.json")
    with open(split_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "train_vel": args.train_vel,
                "test_vel": args.test_vel,
                "win": args.win,
                "stride": args.stride,
                "val_ratio": args.val_ratio,
                "seed": args.seed,
                "trials": {
                    "train": [t["trial_id"] for t in train_trials],
                    "val": [t["trial_id"] for t in val_trials],
                    "test": [t["trial_id"] for t in test_trials],
                },
                "windows": {"train": n_train, "val": n_val, "test": n_test},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print("Saved:", split_json)

    # 6) Allocate memmap for X and ndarray for y
    X_path = os.path.join(out_dir, "X.mmap")
    y_path = os.path.join(out_dir, "y.npy")
    X = create_memmap(X_path, shape=(total_windows, args.win, 9), dtype=np.float32)
    y = np.zeros((total_windows,), dtype=np.int64)

    # 7) Write windows sequentially and build split indices
    idx_train = np.zeros((n_train,), dtype=np.int64)
    idx_val = np.zeros((n_val,), dtype=np.int64)
    idx_test = np.zeros((n_test,), dtype=np.int64)

    def write_split(trials: List[Dict[str, Any]], idx_arr: np.ndarray, write_offset: int) -> int:
        """
        Returns new write_offset after writing all windows.
        Also fills idx_arr with global indices.
        """
        ptr = 0
        off = write_offset

        for t in trials:
            pkl_path = t["pkl_path"]
            y_label = int(t["y"])

            obj = safe_pickle_load(pkl_path)
            arrays: List[np.ndarray] = []
            collect_arrays(obj, arrays)
            arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
            best = pick_best_imu_array(arrays)
            if best is None:
                continue
            x = normalize_to_TC(best)
            if x.shape[1] != 9:
                continue
            T = x.shape[0]
            nwin = count_windows(T, args.win, args.stride)
            if nwin <= 0:
                continue

            # generate windows and write
            for i in range(nwin):
                s = i * args.stride
                e = s + args.win
                w = x[s:e]  # (win, 9)
                X[off] = w
                y[off] = y_label
                idx_arr[ptr] = off
                ptr += 1
                off += 1

        # ptr should match len(idx_arr)
        if ptr != len(idx_arr):
            # If mismatch, shrink idx array (rare; happens only if some trial failed in Pass2)
            # Better to warn and truncate.
            print("[WARN] split wrote", ptr, "windows but expected", len(idx_arr),
                  "-> truncating idx array")
            return off

        return off

    print("[Pass2] writing memmap windows...")
    write_off = 0
    write_off = write_split(train_trials, idx_train, write_off)
    write_off = write_split(val_trials, idx_val, write_off)
    write_off = write_split(test_trials, idx_test, write_off)

    # If some windows were skipped in Pass2, we need to truncate.
    if write_off != total_windows:
        print("[WARN] Pass2 produced", write_off, "windows but Pass1 predicted", total_windows)
        # Save only valid part
        # Note: truncating memmap file is annoying; simplest is just keep tail unused and fix idx/y.
        # Here we will truncate y and indices to valid range and update counts.
        y = y[:write_off].copy()
        np.save(y_path, y)
        np.save(os.path.join(out_dir, "train_idx.npy"), idx_train[idx_train < write_off])
        np.save(os.path.join(out_dir, "val_idx.npy"), idx_val[idx_val < write_off])
        np.save(os.path.join(out_dir, "test_idx.npy"), idx_test[idx_test < write_off])
        print("Saved (truncated):", y_path)
        print("Saved (truncated) idx npys")
        print("=== DONE ===")
        print("OUT_DIR:", out_dir)
        return

    # Flush memmap and save
    X.flush()
    np.save(y_path, y)

    np.save(os.path.join(out_dir, "train_idx.npy"), idx_train)
    np.save(os.path.join(out_dir, "val_idx.npy"), idx_val)
    np.save(os.path.join(out_dir, "test_idx.npy"), idx_test)

    print("Write done. N =", total_windows)
    print("Saved:", X_path)
    print("Saved:", y_path)
    print("Saved:", os.path.join(out_dir, "train_idx.npy"),
          os.path.join(out_dir, "val_idx.npy"),
          os.path.join(out_dir, "test_idx.npy"))
    print("=== DONE ===")
    print("OUT_DIR:", out_dir)


if __name__ == "__main__":
    main()
