# import os
# import re
# import sys
# import types
# import io
# import pickle
# import numpy as np
# from pathlib import Path
# from typing import Any, List, Optional

# # =========================
# # 1) Patch legacy pandas pickle path: pandas.tseries.index + _new_DatetimeIndex
# # =========================
# def patch_pandas_tseries_index():
#     import pandas as pd
#     from pandas.core.indexes.base import Index
#     try:
#         from pandas.core.indexes.datetimes import DatetimeIndex
#     except Exception:
#         DatetimeIndex = None

#     tseries_mod = types.ModuleType("pandas.tseries")
#     index_mod = types.ModuleType("pandas.tseries.index")

#     index_mod.Index = Index
#     if DatetimeIndex is not None:
#         index_mod.DatetimeIndex = DatetimeIndex

#     def _new_DatetimeIndex(data=None, freq=None, tz=None, name=None, **kwargs):
#         # Old internal constructor used by legacy pickles
#         if DatetimeIndex is not None:
#             try:
#                 return DatetimeIndex(data, freq=freq, tz=tz, name=name)
#             except Exception:
#                 try:
#                     return DatetimeIndex(pd.to_datetime(data), tz=tz, name=name)
#                 except Exception:
#                     pass
#         try:
#             return Index(data, name=name)
#         except Exception:
#             return Index([], name=name)

#     index_mod._new_DatetimeIndex = _new_DatetimeIndex

#     sys.modules["pandas.tseries"] = tseries_mod
#     sys.modules["pandas.tseries.index"] = index_mod

# patch_pandas_tseries_index()

# # =========================
# # 2) Paths
# # =========================
# ROOT = r"F:\study\\LSMRT\data\\multimodal-tactile-texture-dataset"
# OUT_DIR = os.path.join(ROOT, "processed")
# os.makedirs(OUT_DIR, exist_ok=True)

# # =========================
# # 3) Compat Unpickler for py2/legacy pickle
# # =========================
# class CompatUnpickler(pickle.Unpickler):
#     def find_class(self, module, name):
#         # Route legacy pandas.tseries.index symbols to our injected module if exists
#         if module == "pandas.tseries.index":
#             mod = sys.modules.get("pandas.tseries.index")
#             if mod is not None and hasattr(mod, name):
#                 return getattr(mod, name)
#         return super().find_class(module, name)

# def safe_pickle_load(pkl_path: str):
#     with open(pkl_path, "rb") as f:
#         raw = f.read()

#     # First try: latin1 decoding (most common py2->py3 fix)
#     try:
#         return CompatUnpickler(
#             io.BytesIO(raw),
#             fix_imports=True,
#             encoding="latin1",
#             errors="ignore",
#         ).load()
#     except Exception:
#         # Fallback: keep strings as bytes
#         return CompatUnpickler(
#             io.BytesIO(raw),
#             fix_imports=True,
#             encoding="bytes",
#             errors="ignore",
#         ).load()

# # =========================
# # 4) Collect arrays (and DataFrame) recursively
# # =========================
# def collect_arrays(obj: Any, arrays: List[np.ndarray], max_nodes: int = 20000):
#     """
#     Recursively collect all numeric arrays from obj.
#     Also handles pandas.DataFrame/Series by converting to numpy.
#     """
#     stack = [obj]
#     visited = set()
#     nodes = 0

#     # Lazy import to avoid hard dependency errors
#     try:
#         import pandas as pd
#         has_pd = True
#     except Exception:
#         has_pd = False
#         pd = None

#     while stack and nodes < max_nodes:
#         cur = stack.pop()
#         nodes += 1

#         oid = id(cur)
#         if oid in visited:
#             continue
#         visited.add(oid)

#         # pandas DataFrame/Series -> numpy
#         if has_pd:
#             try:
#                 if isinstance(cur, pd.DataFrame):
#                     try:
#                         a = cur.to_numpy(dtype=np.float32, copy=False)
#                     except Exception:
#                         a = cur.values
#                     arrays.append(np.asarray(a))
#                     continue
#                 if isinstance(cur, pd.Series):
#                     try:
#                         a = cur.to_numpy(dtype=np.float32, copy=False)
#                     except Exception:
#                         a = cur.values
#                     arrays.append(np.asarray(a))
#                     continue
#             except Exception:
#                 pass

#         if isinstance(cur, np.ndarray):
#             arrays.append(cur)
#             continue

#         if isinstance(cur, (list, tuple)):
#             # Try convert entire list/tuple into ndarray (if numeric)
#             try:
#                 arr = np.asarray(cur)
#                 if isinstance(arr, np.ndarray) and arr.dtype != object:
#                     arrays.append(arr)
#                     continue
#             except Exception:
#                 pass
#             for x in cur:
#                 stack.append(x)
#             continue

#         if isinstance(cur, dict):
#             for k, v in cur.items():
#                 stack.append(k)
#                 stack.append(v)
#             continue

#         if hasattr(cur, "__dict__"):
#             stack.append(cur.__dict__)
#             continue

# # =========================
# # 5) IMU array scoring + picking
# # =========================
# def score_as_imu(arr: np.ndarray) -> float:
#     """
#     Score how likely an array is an IMU time-series (T,C).
#     Strongly penalize empty arrays and non-numeric.
#     """
#     if not isinstance(arr, np.ndarray):
#         return -1.0
#     if arr.dtype == object:
#         return -1.0
#     if arr.size == 0:
#         return -1.0

#     # Ensure it is castable to float (filter out datetime-like)
#     try:
#         _ = arr.astype(np.float32, copy=False)
#     except Exception:
#         return -1.0

#     if arr.ndim == 1:
#         return 1.0 if arr.shape[0] >= 50 else -1.0

#     if arr.ndim == 2:
#         T, C = arr.shape
#         # allow (C,T)
#         if T < C:
#             T, C = C, T
#         if T <= 0 or C <= 0:
#             return -1.0

#         score = 0.0
#         # Common IMU channels: 3/6/9/12/15/18; allow wider up to 24
#         if 2 <= C <= 24:
#             score += 2.0
#         if C in (3, 6, 9, 12, 15, 18):
#             score += 1.5

#         if T >= 50:
#             score += 2.0
#         if T >= 200:
#             score += 1.0
#         if T >= 500:
#             score += 0.5

#         # Prefer typical "more time than channels"
#         if T > C:
#             score += 0.5
#         return score

#     if arr.ndim == 3:
#         s, T, C = arr.shape
#         if s <= 0 or T <= 0 or C <= 0:
#             return -1.0
#         # Many datasets store segments x time x channels
#         score = 0.5
#         if 2 <= C <= 24:
#             score += 1.5
#         if C in (3, 6, 9, 12, 15, 18):
#             score += 1.0
#         if T >= 50:
#             score += 1.5
#         if s <= 64:
#             score += 0.5
#         return score

#     return -1.0

# def pick_best_imu_array(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
#     # Filter empty arrays early
#     arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
#     if not arrays:
#         return None

#     best = None
#     best_score = -1e18
#     for a in arrays:
#         try:
#             s = score_as_imu(a)
#         except Exception:
#             continue
#         if s > best_score:
#             best_score = s
#             best = a
#     return best

# # =========================
# # 6) Normalize to (T,C)
# # =========================
# def normalize_to_TC(arr: np.ndarray) -> np.ndarray:
#     """
#     Normalize array to float32 (T,C).
#     """
#     a = np.asarray(arr)

#     if a.dtype == object:
#         raise ValueError("object dtype array")
#     if a.size == 0:
#         return np.zeros((0, 1), dtype=np.float32)

#     # cast to float32 if possible
#     try:
#         a = a.astype(np.float32, copy=False)
#     except Exception:
#         a = a.astype(np.float32)

#     if a.ndim == 1:
#         return a[:, None]

#     if a.ndim == 2:
#         # Make T the larger dimension
#         if a.shape[0] < a.shape[1]:
#             a = a.T
#         return a

#     if a.ndim == 3:
#         # (S,T,C) -> pick first segment, then ensure (T,C)
#         s, t, c = a.shape
#         # if it's (S,C,T)
#         if t < c:
#             a = np.transpose(a, (0, 2, 1))  # (S,T,C)
#         a0 = a[0]  # (T,C)
#         if a0.shape[0] < a0.shape[1]:
#             a0 = a0.T
#         return a0

#     raise ValueError("Unsupported ndim={}".format(a.ndim))

# # =========================
# # 7) Utils
# # =========================
# def list_all_pkls(root: str) -> List[str]:
#     return [str(p) for p in Path(root).rglob("full_imu/*.pkl")]

# def parse_label_from_path(pkl_path: str) -> int:
#     m = re.search(r"texture_(\d+)", pkl_path.replace("\\", "/"))
#     if not m:
#         raise ValueError("Cannot parse label from {}".format(pkl_path))
#     return int(m.group(1))

# def window_cut(x: np.ndarray, win: int = 256, stride: int = 128) -> List[np.ndarray]:
#     T, C = x.shape
#     if T < win:
#         return []
#     out = []
#     for start in range(0, T - win + 1, stride):
#         out.append(x[start:start + win])
#     return out

# # =========================
# # 8) Build NPZ with debug stats
# # =========================
# def build_npz(root: str, out_path: str, win: int = 256, stride: int = 128, debug_print: int = 20):
#     pkls = list_all_pkls(root)
#     print("Found pkls:", len(pkls))

#     X_list, y_list = [], []
#     stats = {
#         "load_fail": 0,
#         "no_arrays": 0,
#         "no_best": 0,
#         "normalize_fail": 0,
#         "nan_clean_fail": 0,
#         "window_empty": 0,
#         "label_fail": 0,
#         "ok": 0,
#         "other_fail": 0,
#     }
#     printed = 0

#     for p in pkls:
#         try:
#             # 1) load
#             try:
#                 obj = safe_pickle_load(p)
#             except Exception as e:
#                 stats["load_fail"] += 1
#                 if printed < debug_print:
#                     print("[LOAD_FAIL]", p, "err=", repr(e))
#                     printed += 1
#                 continue

#             # 2) collect arrays
#             arrays = []
#             try:
#                 collect_arrays(obj, arrays)
#             except Exception as e:
#                 stats["no_arrays"] += 1
#                 if printed < debug_print:
#                     print("[COLLECT_FAIL]", p, "err=", repr(e))
#                     printed += 1
#                 continue

#             # Filter empty arrays
#             arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
#             if len(arrays) == 0:
#                 stats["no_arrays"] += 1
#                 if printed < debug_print:
#                     print("[NO_ARRAYS]", p)
#                     printed += 1
#                 continue

#             # Debug: print candidate shapes for first few files
#             if printed < debug_print:
#                 shapes = []
#                 for a in arrays[:15]:
#                     try:
#                         shapes.append((a.shape, str(a.dtype)))
#                     except Exception:
#                         shapes.append(("?", "?"))
#                 print("[CANDIDATES]", p, shapes)
#                 printed += 1

#             # 3) pick best
#             try:
#                 best = pick_best_imu_array(arrays)
#             except Exception as e:
#                 stats["no_best"] += 1
#                 if printed < debug_print:
#                     print("[PICK_FAIL]", p, "err=", repr(e))
#                     printed += 1
#                 continue

#             if best is None:
#                 stats["no_best"] += 1
#                 if printed < debug_print:
#                     print("[NO_BEST]", p)
#                     printed += 1
#                 continue

#             # 4) normalize to (T,C)
#             try:
#                 x = normalize_to_TC(best)
#             except Exception as e:
#                 stats["normalize_fail"] += 1
#                 if printed < debug_print:
#                     print("[NORM_FAIL]", p, "best_shape=", getattr(best, "shape", None), "dtype=", getattr(best, "dtype", None), "err=", repr(e))
#                     printed += 1
#                 continue

#             # If x is empty, skip
#             if x.size == 0 or x.shape[0] == 0:
#                 stats["window_empty"] += 1
#                 if printed < debug_print:
#                     print("[EMPTY_TS]", p, "x_shape=", x.shape)
#                     printed += 1
#                 continue

#             # 5) clean NaN/Inf (numpy 1.16 compatible)
#             try:
#                 x = np.nan_to_num(x)  # no keyword args for old numpy
#                 x[~np.isfinite(x)] = 0.0
#             except Exception as e:
#                 stats["nan_clean_fail"] += 1
#                 if printed < debug_print:
#                     print("[NAN_CLEAN_FAIL]", p, "x_shape=", x.shape, "err=", repr(e))
#                     printed += 1
#                 continue

#             # 6) window cut
#             windows = window_cut(x, win=win, stride=stride)
#             if not windows:
#                 stats["window_empty"] += 1
#                 if printed < debug_print:
#                     print("[WINDOW_EMPTY]", p, "x_shape=", x.shape, "win=", win, "stride=", stride)
#                     printed += 1
#                 continue

#             # 7) label parse
#             try:
#                 label = parse_label_from_path(p) - 1  # 0-based
#             except Exception as e:
#                 stats["label_fail"] += 1
#                 if printed < debug_print:
#                     print("[LABEL_FAIL]", p, "err=", repr(e))
#                     printed += 1
#                 continue

#             for w in windows:
#                 X_list.append(w)
#                 y_list.append(label)

#             stats["ok"] += 1

#         except Exception as e:
#             stats["other_fail"] += 1
#             if printed < debug_print:
#                 print("[OTHER_FAIL]", p, "err=", repr(e))
#                 printed += 1
#             continue

#     print("=== STATS ===")
#     for k, v in stats.items():
#         print(k, v)

#     if len(X_list) == 0:
#         raise RuntimeError("No windows were produced. Check STATS above and printed samples.")

#     X = np.stack(X_list, axis=0).astype(np.float32)  # (N, win, C)
#     y = np.array(y_list, dtype=np.int64)
#     np.savez_compressed(out_path, X=X, y=y)
#     print("Saved:", out_path)
#     print("X shape:", X.shape, "y shape:", y.shape)

# # =========================
# # 9) Run
# # =========================
# out_path = os.path.join(OUT_DIR, "imu_windows_w256_s128.npz")
# build_npz(ROOT, out_path, win=256, stride=128, debug_print=10)




import os
import re
import sys
import types
import io
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Any, List, Optional

# =========================
# 1) Patch legacy pandas pickle path: pandas.tseries.index + _new_DatetimeIndex
# =========================
def patch_pandas_tseries_index():
    """
    Some legacy pickles reference pandas.tseries.index symbols.
    We inject a tiny shim module so unpickling doesn't fail.
    """
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


# =========================
# 2) Compat Unpickler for py2/legacy pickle
# =========================
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

    # Try latin1 (common py2->py3)
    try:
        return CompatUnpickler(
            io.BytesIO(raw),
            fix_imports=True,
            encoding="latin1",
            errors="ignore",
        ).load()
    except Exception:
        # Fallback: keep strings as bytes
        return CompatUnpickler(
            io.BytesIO(raw),
            fix_imports=True,
            encoding="bytes",
            errors="ignore",
        ).load()


# =========================
# 3) Collect arrays recursively
# =========================
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
                    try:
                        a = cur.to_numpy(dtype=np.float32, copy=False)
                    except Exception:
                        a = cur.values
                    arrays.append(np.asarray(a))
                    continue
                if isinstance(cur, pd.Series):
                    try:
                        a = cur.to_numpy(dtype=np.float32, copy=False)
                    except Exception:
                        a = cur.values
                    arrays.append(np.asarray(a))
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


# =========================
# 4) Pick best IMU array
# =========================
def score_as_imu(arr: np.ndarray) -> float:
    if not isinstance(arr, np.ndarray) or arr.dtype == object or arr.size == 0:
        return -1.0
    try:
        _ = arr.astype(np.float32, copy=False)
    except Exception:
        return -1.0

    if arr.ndim == 1:
        return 1.0 if arr.shape[0] >= 50 else -1.0

    if arr.ndim == 2:
        T, C = arr.shape
        if T < C:
            T, C = C, T
        if T <= 0 or C <= 0:
            return -1.0
        score = 0.0
        if 2 <= C <= 24:
            score += 2.0
        if C in (3, 6, 9, 12, 15, 18):
            score += 1.5
        if T >= 50:
            score += 2.0
        if T >= 200:
            score += 1.0
        if T >= 500:
            score += 0.5
        if T > C:
            score += 0.5
        return score

    if arr.ndim == 3:
        s, T, C = arr.shape
        if s <= 0 or T <= 0 or C <= 0:
            return -1.0
        score = 0.5
        if 2 <= C <= 24:
            score += 1.5
        if C in (3, 6, 9, 12, 15, 18):
            score += 1.0
        if T >= 50:
            score += 1.5
        if s <= 64:
            score += 0.5
        return score

    return -1.0


def pick_best_imu_array(arrays: List[np.ndarray]) -> Optional[np.ndarray]:
    arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
    if not arrays:
        return None
    best, best_score = None, -1e18
    for a in arrays:
        s = score_as_imu(a)
        if s > best_score:
            best_score, best = s, a
    return best


# =========================
# 5) Normalize to (T,C)
# =========================
def normalize_to_TC(arr: np.ndarray) -> np.ndarray:
    a = np.asarray(arr)
    if a.dtype == object:
        raise ValueError("object dtype array")
    if a.size == 0:
        return np.zeros((0, 1), dtype=np.float32)

    try:
        a = a.astype(np.float32, copy=False)
    except Exception:
        a = a.astype(np.float32)

    if a.ndim == 1:
        return a[:, None]
    if a.ndim == 2:
        if a.shape[0] < a.shape[1]:
            a = a.T
        return a
    if a.ndim == 3:
        s, t, c = a.shape
        if t < c:
            a = np.transpose(a, (0, 2, 1))  # (S,T,C)
        a0 = a[0]
        if a0.shape[0] < a0.shape[1]:
            a0 = a0.T
        return a0
    raise ValueError(f"Unsupported ndim={a.ndim}")


def window_cut(x: np.ndarray, win: int = 256, stride: int = 128) -> List[np.ndarray]:
    T, C = x.shape
    if T < win:
        return []
    out = []
    for start in range(0, T - win + 1, stride):
        out.append(x[start:start + win])
    return out


def list_all_pkls(root: str) -> List[str]:
    return [str(p) for p in Path(root).rglob("full_imu/*.pkl")]


def parse_label_from_path(pkl_path: str) -> int:
    m = re.search(r"texture_(\d+)", pkl_path.replace("\\", "/"))
    if not m:
        raise ValueError(f"Cannot parse label from {pkl_path}")
    return int(m.group(1))


def build_npz(
    root: str,
    out_path: str,
    win: int,
    stride: int,
    debug_print: int,
    max_pkls: int,
    seed: int,
):
    pkls = list_all_pkls(root)
    print("Found pkls:", len(pkls))

    if max_pkls > 0 and max_pkls < len(pkls):
        rng = np.random.default_rng(seed)
        pkls = list(rng.choice(pkls, size=max_pkls, replace=False))
        print("Sampled pkls:", len(pkls))

    X_list, y_list = [], []
    stats = {k: 0 for k in [
        "load_fail", "no_arrays", "no_best", "normalize_fail", "nan_clean_fail",
        "window_empty", "label_fail", "ok", "other_fail"
    ]}
    printed = 0

    for p in pkls:
        try:
            try:
                obj = safe_pickle_load(p)
            except Exception as e:
                stats["load_fail"] += 1
                if printed < debug_print:
                    print("[LOAD_FAIL]", p, "err=", repr(e))
                    printed += 1
                continue

            arrays = []
            try:
                collect_arrays(obj, arrays)
            except Exception as e:
                stats["no_arrays"] += 1
                if printed < debug_print:
                    print("[COLLECT_FAIL]", p, "err=", repr(e))
                    printed += 1
                continue

            arrays = [a for a in arrays if isinstance(a, np.ndarray) and a.size > 0 and a.dtype != object]
            if not arrays:
                stats["no_arrays"] += 1
                if printed < debug_print:
                    print("[NO_ARRAYS]", p)
                    printed += 1
                continue

            if printed < debug_print:
                shapes = [(getattr(a, "shape", None), str(getattr(a, "dtype", None))) for a in arrays[:10]]
                print("[CANDIDATES]", p, shapes)
                printed += 1

            best = pick_best_imu_array(arrays)
            if best is None:
                stats["no_best"] += 1
                continue

            try:
                x = normalize_to_TC(best)
            except Exception as e:
                stats["normalize_fail"] += 1
                if printed < debug_print:
                    print("[NORM_FAIL]", p, "best_shape=", getattr(best, "shape", None), "err=", repr(e))
                    printed += 1
                continue

            if x.size == 0 or x.shape[0] == 0:
                stats["window_empty"] += 1
                continue

            try:
                x = np.nan_to_num(x)
                x[~np.isfinite(x)] = 0.0
            except Exception as e:
                stats["nan_clean_fail"] += 1
                if printed < debug_print:
                    print("[NAN_CLEAN_FAIL]", p, "err=", repr(e))
                    printed += 1
                continue

            windows = window_cut(x, win=win, stride=stride)
            if not windows:
                stats["window_empty"] += 1
                continue

            try:
                label = parse_label_from_path(p) - 1
            except Exception as e:
                stats["label_fail"] += 1
                if printed < debug_print:
                    print("[LABEL_FAIL]", p, "err=", repr(e))
                    printed += 1
                continue

            for w in windows:
                X_list.append(w)
                y_list.append(label)

            stats["ok"] += 1

        except Exception as e:
            stats["other_fail"] += 1
            if printed < debug_print:
                print("[OTHER_FAIL]", p, "err=", repr(e))
                printed += 1

    print("=== STATS ===")
    for k, v in stats.items():
        print(k, v)

    if len(X_list) == 0:
        raise RuntimeError("No windows were produced. Check STATS above.")

    X = np.stack(X_list, axis=0).astype(np.float32)  # (N, win, C)
    y = np.array(y_list, dtype=np.int64)
    np.savez_compressed(out_path, X=X, y=y)
    print("Saved:", out_path)
    print("X shape:", X.shape, "y shape:", y.shape)


def main():
    patch_pandas_tseries_index()

    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="dataset root, contains pickles_30/")
    ap.add_argument("--out", type=str, default="", help="output npz path")
    ap.add_argument("--win", type=int, default=256)
    ap.add_argument("--stride", type=int, default=128)
    ap.add_argument("--debug_print", type=int, default=10)
    ap.add_argument("--max_pkls", type=int, default=0, help=">0 to randomly sample N pkls (debug)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out_path = args.out
    if not out_path:
        out_dir = os.path.join(args.root, "processed")
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, f"imu_windows_w{args.win}_s{args.stride}.npz")

    build_npz(
        root=args.root,
        out_path=out_path,
        win=args.win,
        stride=args.stride,
        debug_print=args.debug_print,
        max_pkls=args.max_pkls,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()