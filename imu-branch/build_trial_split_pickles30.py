# 测试数据的分布，和训练数据明显不一样。训练：30 测试：40
import os, re, sys, json, csv, io, types, pickle, argparse
import numpy as np
from pathlib import Path
from typing import Any, List, Optional, Tuple

# =========================
# Patch legacy pandas pickle path (optional; safe even if pandas absent)
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

def collect_arrays(obj: Any, arrays: List[np.ndarray], max_nodes: int = 20000):
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
                stack.append(k); stack.append(v)
            continue

        if hasattr(cur, "__dict__"):
            stack.append(cur.__dict__)
            continue

def score_as_imu(arr: np.ndarray) -> float:
    if not isinstance(arr, np.ndarray) or arr.dtype == object or arr.size == 0:
        return -1.0
    try:
        _ = arr.astype(np.float32, copy=False)
    except Exception:
        return -1.0

    if arr.ndim == 2:
        T, C = arr.shape
        if T < C:
            T, C = C, T
        score = 0.0
        if 2 <= C <= 24: score += 2.0
        if C in (3, 6, 9, 12, 15, 18): score += 1.5
        if T >= 50: score += 2.0
        if T >= 200: score += 1.0
        if T > C: score += 0.5
        return score
    return -1.0

def pick_best_imu_array(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
    if not arrays:
        return None
    best, best_s = None, -1e18
    for a in arrays:
        s = score_as_imu(a)
        if s > best_s:
            best_s, best = s, a
    return best

def normalize_to_TC(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == object or a.size == 0:
        return np.zeros((0, 1), dtype=np.float32)
    a = a.astype(np.float32, copy=False)
    if a.ndim == 2:
        if a.shape[0] < a.shape[1]:
            a = a.T
        return a
    raise ValueError(f"unsupported ndim {a.ndim}")

def window_count(T: int, win: int, stride: int) -> int:
    if T < win: return 0
    return 1 + (T - win) // stride

def list_full_imu_pkls(root: str, vel: int) -> List[str]:
    # only scan pickles_{vel}/texture_*/full_imu/*.pkl
    base = Path(root) / f"pickles_{vel}"
    return [str(p) for p in base.rglob("full_imu/*.pkl")]

def parse_texture_id(pkl_path: str) -> int:
    m = re.search(r"texture_(\d+)", pkl_path.replace("\\", "/"))
    if not m:
        raise ValueError(f"cannot parse texture from {pkl_path}")
    return int(m.group(1)) - 1  # 0-based

def trial_id_from_path(pkl_path: str, vel: int) -> str:
    # unique per file
    name = Path(pkl_path).name
    tex = re.search(r"texture_(\d+)", pkl_path.replace("\\", "/")).group(1)
    return f"v{vel}_t{tex}_{name}"

def compute_mean_std_stream(X_mmap: np.memmap, idx: np.ndarray, batch: int = 8192) -> Tuple[np.ndarray, np.ndarray]:
    # X: (N, T, C)
    # mean/std over all samples+time (global per-channel)
    C = X_mmap.shape[2]
    s1 = np.zeros((C,), dtype=np.float64)
    s2 = np.zeros((C,), dtype=np.float64)
    n = 0
    for i in range(0, len(idx), batch):
        b = idx[i:i+batch]
        xb = X_mmap[b]  # (B, T, C)
        xb = xb.reshape(-1, C).astype(np.float64, copy=False)
        s1 += xb.sum(axis=0)
        s2 += (xb * xb).sum(axis=0)
        n += xb.shape[0]
    mean = (s1 / max(n, 1)).astype(np.float32)
    var = (s2 / max(n, 1) - mean.astype(np.float64)**2)
    var = np.maximum(var, 1e-12)
    std = np.sqrt(var).astype(np.float32)
    return mean, std

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help=".../multimodal-tactile-texture-dataset")
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--vel", type=int, default=30)
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--train_ratio", type=float, default=0.70)
    ap.add_argument("--val_ratio", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--debug_first", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.RandomState(args.seed)

    print(f"[Scan] pickles_{args.vel} full_imu...")
    pkls = list_full_imu_pkls(args.root, args.vel)
    print("Found full_imu pkls:", len(pkls))
    if len(pkls) == 0:
        raise RuntimeError("No full_imu pkls found. Check root/vel/path.")

    # Pass1: build trials table + total windows
    trials = []
    total_windows = 0
    dbg = 0
    for p in pkls:
        obj = safe_pickle_load(p)
        arrays = []
        collect_arrays(obj, arrays)
        best = pick_best_imu_array(arrays)
        if best is None:
            continue
        x = normalize_to_TC(best)
        T, C = x.shape
        nw = window_count(T, args.win, args.stride)
        if nw <= 0:
            continue
        y = parse_texture_id(p)
        tid = trial_id_from_path(p, args.vel)
        trials.append((tid, p, args.vel, y, T, C, nw))
        total_windows += nw
        if args.debug_first and dbg < args.debug_first:
            print("[DBG]", Path(p).name, "vel=", args.vel, "y=", y, "best_shape=", x.shape, "nw=", nw)
            dbg += 1

    print("Trials kept:", len(trials))
    print("Total windows:", total_windows)
    if len(trials) == 0 or total_windows == 0:
        raise RuntimeError("No trials/windows produced.")

    # trial split
    idx_trials = np.arange(len(trials))
    rng.shuffle(idx_trials)
    n = len(idx_trials)
    n_train = int(n * args.train_ratio)
    n_val = int(n * args.val_ratio)
    n_test = n - n_train - n_val
    tr_ids = idx_trials[:n_train]
    va_ids = idx_trials[n_train:n_train+n_val]
    te_ids = idx_trials[n_train+n_val:]

    split = {
        "vel": args.vel,
        "seed": args.seed,
        "train_trials": [trials[i][0] for i in tr_ids],
        "val_trials":   [trials[i][0] for i in va_ids],
        "test_trials":  [trials[i][0] for i in te_ids],
        "win": args.win,
        "stride": args.stride,
    }
    with open(os.path.join(args.out_dir, "split_trials.json"), "w", encoding="utf-8") as f:
        json.dump(split, f, indent=2)

    # write trials.csv with window offset ranges
    trials_csv = os.path.join(args.out_dir, "trials.csv")
    with open(trials_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["trial_id","pkl_path","vel","label","T","C","n_windows","win_start","win_end"])
        cur = 0
        trials2 = []
        for (tid, p, vel, y, T, C, nw) in trials:
            win_start = cur
            win_end = cur + nw  # exclusive
            w.writerow([tid, p, vel, y, T, C, nw, win_start, win_end])
            trials2.append((tid, p, vel, y, T, C, nw, win_start, win_end))
            cur = win_end

    # Pass2: write memmap arrays
    X_path = os.path.join(args.out_dir, "X.mmap")
    y_path = os.path.join(args.out_dir, "y.npy")
    X = np.memmap(X_path, dtype=np.float32, mode="w+", shape=(total_windows, args.win, 9))
    y = np.zeros((total_windows,), dtype=np.int64)

    # map trial_id -> split tag
    tr_set = set(split["train_trials"])
    va_set = set(split["val_trials"])
    te_set = set(split["test_trials"])

    # build idx lists
    idx_train, idx_val, idx_test = [], [], []

    print("[Write] memmap windows...")
    for (tid, p, vel, lab, T, C, nw, win_start, win_end) in trials2:
        obj = safe_pickle_load(p)
        arrays = []
        collect_arrays(obj, arrays)
        best = pick_best_imu_array(arrays)
        x = normalize_to_TC(best)  # (T,C)
        # enforce C=9 (dataset is 9)
        if x.shape[1] != 9:
            # skip or pad/crop; here skip to keep consistent
            continue

        out_pos = win_start
        for start in range(0, x.shape[0] - args.win + 1, args.stride):
            X[out_pos] = x[start:start+args.win]
            y[out_pos] = lab
            out_pos += 1

        # append indices by trial
        trial_indices = np.arange(win_start, win_end, dtype=np.int64)
        if tid in tr_set:
            idx_train.append(trial_indices)
        elif tid in va_set:
            idx_val.append(trial_indices)
        else:
            idx_test.append(trial_indices)

    X.flush()
    np.save(y_path, y)

    idx_train = np.concatenate(idx_train) if len(idx_train) else np.zeros((0,), dtype=np.int64)
    idx_val   = np.concatenate(idx_val)   if len(idx_val) else np.zeros((0,), dtype=np.int64)
    idx_test  = np.concatenate(idx_test)  if len(idx_test) else np.zeros((0,), dtype=np.int64)

    np.save(os.path.join(args.out_dir, "idx_train.npy"), idx_train)
    np.save(os.path.join(args.out_dir, "idx_val.npy"), idx_val)
    np.save(os.path.join(args.out_dir, "idx_test.npy"), idx_test)

    # 额外兼容你之前某些脚本命名
    np.save(os.path.join(args.out_dir, "train_idx.npy"), idx_train)
    np.save(os.path.join(args.out_dir, "val_idx.npy"), idx_val)
    np.save(os.path.join(args.out_dir, "test_idx.npy"), idx_test)

    print("[Stats] train/val/test windows:", len(idx_train), len(idx_val), len(idx_test))
    print("Saved:", args.out_dir)
    print("X:", X.shape, "y:", y.shape)

    # 可选：算训练集 mean/std（后续训练用）
    mean, std = compute_mean_std_stream(X, idx_train)
    np.save(os.path.join(args.out_dir, "mean.npy"), mean)
    np.save(os.path.join(args.out_dir, "std.npy"), std)
    print("Saved mean/std:", mean, std)

if __name__ == "__main__":
    main()
