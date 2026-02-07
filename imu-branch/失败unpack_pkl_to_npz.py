# import os
# import sys
# import types
# import pickle
# import numpy as np
# import pandas as pd

# # ====== 你的数据根目录（按你工程结构）======
# DATA_ROOT = r"F:\study\\LSMRT\data\\multimodal-tactile-texture-dataset"
# OUT_ROOT  = r"F:\study\\LSMRT\data\\multimodal-tactile-texture-dataset\\mttd_npz"  # 解包后的输出目录

# SPEEDS = ["pickles_30", "pickles_35", "pickles_40"]  # 你也可以先只跑 pickles_30


# def ensure_pandas_legacy_modules():
#     """
#     兼容旧 pickle 里引用的 pandas 内部模块路径/符号：
#         - pandas.core.index.Index
#         - pandas.core.index._new_Index
#         - pandas.tseries.index.Index / DatetimeIndex / ...
#         - pandas.tseries.index._new_DatetimeIndex
#     """
#     import types, sys
#     import pandas as pd
#     import numpy as np

#     # ---------- pandas.core.index ----------
#     m_core_index = types.ModuleType("pandas.core.index")
#     m_core_index.Index = pd.Index

#     def _new_Index(data=None, dtype=None, copy=False, name=None, **kwargs):
#         """
#         兼容旧 pickle：
#         - dtype 可能不可解析 -> 忽略
#         - data 可能是标量/0维 -> 包一层变成1维
#         - data 可能为 None -> 空 Index
#         """
#         import numpy as np
#         import pandas as pd

#         # ---- 1) 规整 data：保证变成“可一维化”的序列 ----
#         if data is None:
#             data2 = []
#         else:
#             try:
#                 arr = np.asarray(data)
#                 if arr.ndim == 0:
#                     data2 = [arr.item()] if hasattr(arr, "item") else [data]
#                 else:
#                     data2 = data
#             except Exception:
#                 data2 = [data]

#         # ---- 2) 处理 dtype：尽量解析，失败则忽略 ----
#         dtype_ok = None
#         if dtype is not None:
#             try:
#                 dtype_ok = np.dtype(dtype)
#             except Exception:
#                 dtype_ok = None

#         # ---- 3) 构造 Index：尽量保留 copy/name；失败就逐级降级 ----
#         try:
#             if dtype_ok is not None:
#                 return pd.Index(data2, dtype=dtype_ok, copy=copy, name=name)
#             else:
#                 return pd.Index(data2, copy=copy, name=name)
#         except Exception:
#             try:
#                 return pd.Index(data2)
#             except Exception:
#                 try:
#                     return pd.Index(list(data2))
#                 except Exception:
#                     return pd.Index([])

#     m_core_index._new_Index = _new_Index

#     # 多挂一些可能被旧 pickle 引用的类（兜底）
#     if hasattr(pd, "RangeIndex"):
#         m_core_index.RangeIndex = pd.RangeIndex
#     if hasattr(pd, "DatetimeIndex"):
#         m_core_index.DatetimeIndex = pd.DatetimeIndex
#     if hasattr(pd, "TimedeltaIndex"):
#         m_core_index.TimedeltaIndex = pd.TimedeltaIndex
#     if hasattr(pd, "PeriodIndex"):
#         m_core_index.PeriodIndex = pd.PeriodIndex

#     sys.modules["pandas.core.index"] = m_core_index

#     # ---------- pandas.tseries.index ----------
#     m_tseries_index = types.ModuleType("pandas.tseries.index")
#     m_tseries_index.Index = pd.Index
#     if hasattr(pd, "DatetimeIndex"):
#         m_tseries_index.DatetimeIndex = pd.DatetimeIndex
#     if hasattr(pd, "TimedeltaIndex"):
#         m_tseries_index.TimedeltaIndex = pd.TimedeltaIndex
#     if hasattr(pd, "PeriodIndex"):
#         m_tseries_index.PeriodIndex = pd.PeriodIndex

#     def _new_DatetimeIndex(data=None, dtype=None, copy=False, name=None, tz=None, freq=None, **kwargs):
#         """
#         兼容旧 pickle 的 DatetimeIndex 工厂函数。
#         data 可能是 None/标量/奇怪对象；尽量构造 DatetimeIndex，失败就退化为 Index。
#         """
#         import numpy as np
#         import pandas as pd

#         # data 规整：None -> []，标量 -> [标量]
#         if data is None:
#             data2 = []
#         else:
#             try:
#                 arr = np.asarray(data)
#                 if arr.ndim == 0:
#                     data2 = [arr.item()] if hasattr(arr, "item") else [data]
#                 else:
#                     data2 = data
#             except Exception:
#                 data2 = [data]

#         # dtype 可能是旧结构，尝试解析，失败忽略
#         dtype_ok = None
#         if dtype is not None:
#             try:
#                 dtype_ok = np.dtype(dtype)
#             except Exception:
#                 dtype_ok = None

#         # 优先构造 DatetimeIndex
#         try:
#             # pd.DatetimeIndex 的签名各版本略有差异，尽量传 name/tz/freq
#             return pd.DatetimeIndex(data2, name=name, tz=tz, freq=freq)
#         except Exception:
#             try:
#                 return pd.DatetimeIndex(data2)
#             except Exception:
#                 # 退化为普通 Index（不影响我们最终取数值）
#                 try:
#                     if dtype_ok is not None:
#                         return pd.Index(data2, dtype=dtype_ok, copy=copy, name=name)
#                     else:
#                         return pd.Index(data2, copy=copy, name=name)
#                 except Exception:
#                     return pd.Index([])

#     m_tseries_index._new_DatetimeIndex = _new_DatetimeIndex

#     # 注册模块
#     sys.modules["pandas.tseries.index"] = m_tseries_index


# def to_numpy(sample):
#     """
#     把各种可能的数据结构统一成 numpy 数组。
#     目标输出：imu -> (T, C) float32
#     """
#     # DataFrame -> values
#     if isinstance(sample, pd.DataFrame):
#         arr = sample.values
#         return arr.astype(np.float32), list(sample.columns)

#     # dict -> 尝试找到 imu/acc/gyro
#     if isinstance(sample, dict):
#         for key in ["imu", "IMU", "data", "signal"]:
#             if key in sample:
#                 v = sample[key]
#                 if isinstance(v, pd.DataFrame):
#                     return v.values.astype(np.float32), list(v.columns)
#                 arr = np.asarray(v)
#                 if arr.ndim == 1:
#                     arr = arr[:, None]
#                 return arr.astype(np.float32), [f"c{i}" for i in range(arr.shape[-1])]

#         # acc + gyro 合并
#         if "acc" in sample and "gyro" in sample:
#             acc = np.asarray(sample["acc"])
#             gyro = np.asarray(sample["gyro"])
#             if acc.ndim == 1:
#                 acc = acc[:, None]
#             if gyro.ndim == 1:
#                 gyro = gyro[:, None]
#             arr = np.concatenate([acc, gyro], axis=-1)
#             return arr.astype(np.float32), ["acc_x", "acc_y", "acc_z", "gyro_x", "gyro_y", "gyro_z"]

#         # 否则：取第一个像数组的东西
#         for _, v in sample.items():
#             try:
#                 arr = np.asarray(v)
#                 if arr.ndim == 1:
#                     arr = arr[:, None]
#                 if arr.ndim >= 2:
#                     return arr.astype(np.float32), [f"c{i}" for i in range(arr.shape[-1])]
#             except Exception:
#                 pass

#     # numpy array / list
#     arr = np.asarray(sample)
#     if arr.ndim == 0:
#         arr = np.array([arr])
#     if arr.ndim == 1:
#         arr = arr[:, None]
#     return arr.astype(np.float32), [f"c{i}" for i in range(arr.shape[-1])]


# def safe_pickle_load(path: str):
#     import pickle
#     import pandas as pd

#     # pandas 提供的“跨版本 pickle 兼容加载器”
#     try:
#         from pandas.compat import pickle_compat
#     except Exception:
#         pickle_compat = None

#     with open(path, "rb") as f:
#         # 1) 先试最标准
#         try:
#             return pickle.load(f)
#         except UnicodeDecodeError:
#             pass
#         except Exception:
#             # 其他异常也继续往下试
#             pass

#         # 2) 试 python2 -> python3 常用编码兼容
#         try:
#             f.seek(0)
#             return pickle.load(f, encoding="latin1")
#         except Exception:
#             pass

#         # 3) 关键：用 pandas 的 pickle_compat（最可能解决你现在的 BlockManager 重建错误）
#         if pickle_compat is not None:
#             try:
#                 f.seek(0)
#                 return pickle_compat.load(f, encoding="latin1")
#             except Exception:
#                 pass

#         # 4) 最后兜底：encoding=bytes
#         f.seek(0)
#         return pickle.load(f, encoding="bytes")




# def main():
#     print("numpy", np.__version__, "pandas", pd.__version__)
#     ensure_pandas_legacy_modules()

#     os.makedirs(OUT_ROOT, exist_ok=True)

#     total = 0
#     for sp in SPEEDS:
#         sp_dir = os.path.join(DATA_ROOT, sp)
#         if not os.path.exists(sp_dir):
#             print("[skip] not found:", sp_dir)
#             continue

#         textures = sorted([d for d in os.listdir(sp_dir) if d.startswith("texture_")])
#         for tex in textures:
#             imu_dir = os.path.join(sp_dir, tex, "full_imu")
#             if not os.path.exists(imu_dir):
#                 continue

#             out_dir = os.path.join(OUT_ROOT, sp, tex, "full_imu")
#             os.makedirs(out_dir, exist_ok=True)

#             pkls = [f for f in os.listdir(imu_dir) if f.lower().endswith(".pkl")]
#             for fn in pkls:
#                 in_path = os.path.join(imu_dir, fn)
#                 out_path = os.path.join(out_dir, fn[:-4] + ".npz")

#                 if os.path.exists(out_path):
#                     continue

#                 # sample = safe_pickle_load(in_path)
#                 # imu, cols = to_numpy(sample)

#                 # np.savez_compressed(out_path, imu=imu, cols=np.array(cols, dtype=object))
#                 # total += 1
#                 try:
#                     sample = safe_pickle_load(in_path)
#                     imu, cols = to_numpy(sample)
#                     np.savez_compressed(out_path, imu=imu, cols=np.array(cols, dtype=object))
#                     total += 1
#                 except Exception as e:
#                     # 记录错误但不中断
#                     err_log = os.path.join(OUT_ROOT, "convert_errors.txt")
#                     with open(err_log, "a", encoding="utf-8") as f:
#                         f.write(f"[FAIL] {in_path}\n{repr(e)}\n\n")
#                     continue


#         print("[done]", sp)

#     print("=== ALL DONE ===")
#     print("total converted:", total)
#     print("out:", OUT_ROOT)


# if __name__ == "__main__":
#     main()


import os
import sys
import types
import pickle
import traceback
import numpy as np

# ===========================
# 配置：按你的目录改
# ===========================
DATA_ROOT = r"F:\study\LSMRT\data\multimodal-tactile-texture-dataset"
OUT_ROOT  = r"F:\study\LSMRT\data\multimodal-tactile-texture-dataset\mttd_npz"

SPEEDS  = ["pickles_30", "pickles_35", "pickles_40"]
SENSORS = ["full_imu"]  # 只做 IMU；如需要气压计：["full_imu","full_baro"]

ERROR_LOG = os.path.join(OUT_ROOT, "convert_errors.txt")


def ensure_legacy_pandas_symbols():
    """
    兼容旧 pickle 里引用的 pandas 内部路径/符号：
      - pandas.core.index._new_Index
      - pandas.tseries.index._new_DatetimeIndex
    关键：不要 np.asarray(data)，避免 __array__ missing self（data 是类/函数对象）
    """
    import pandas as pd
    import inspect

    # ---- helper: 安全把 data 变成 list ----
    def _to_list_safe(data):
        # 旧 pickle 有时会把“类/函数/模块”塞进来，这种直接丢弃
        if data is None:
            return []
        if inspect.isclass(data) or inspect.isfunction(data) or inspect.ismodule(data):
            return []
        # bytes/str 也别拆成字符序列
        if isinstance(data, (str, bytes)):
            return [data]
        # 尝试迭代
        try:
            return list(data)
        except Exception:
            return [data]

    # ---- pandas.core.index ----
    m_core_index = types.ModuleType("pandas.core.index")
    m_core_index.Index = pd.Index

    def _new_Index(data=None, dtype=None, copy=False, name=None, **kwargs):
        data_list = _to_list_safe(data)
        # dtype 在旧 pickle 里经常是奇怪对象，忽略掉更稳
        try:
            # pandas 0.24 对 copy/name 支持不完全，所以逐级降级
            return pd.Index(data_list, name=name)
        except Exception:
            try:
                return pd.Index(data_list)
            except Exception:
                return pd.Index([])

    m_core_index._new_Index = _new_Index

    # 可能会被引用到的 Index 类
    for attr in ["RangeIndex", "DatetimeIndex", "TimedeltaIndex", "PeriodIndex"]:
        if hasattr(pd, attr):
            setattr(m_core_index, attr, getattr(pd, attr))

    sys.modules["pandas.core.index"] = m_core_index

    # ---- pandas.tseries.index ----
    m_tseries_index = types.ModuleType("pandas.tseries.index")
    m_tseries_index.Index = pd.Index
    m_tseries_index._new_Index = _new_Index  # 有些 pickle 会从这里找

    def _new_DatetimeIndex(data=None, tz=None, freq=None, name=None, **kwargs):
        # 同样保守：不要 np.asarray，避免奇怪对象
        data_list = _to_list_safe(data)
        try:
            return pd.DatetimeIndex(data_list, tz=tz, freq=freq, name=name)
        except Exception:
            try:
                return pd.DatetimeIndex(data_list)
            except Exception:
                return pd.DatetimeIndex([])

    m_tseries_index._new_DatetimeIndex = _new_DatetimeIndex

    for attr in ["DatetimeIndex", "TimedeltaIndex", "PeriodIndex"]:
        if hasattr(pd, attr):
            setattr(m_tseries_index, attr, getattr(pd, attr))

    sys.modules["pandas.tseries.index"] = m_tseries_index


def scan_pkls(speed_dir: str):
    """
    扫描 pickles_xx 下 texture_*/full_imu/*.pkl
    """
    pkls = []
    textures = sorted([d for d in os.listdir(speed_dir) if d.startswith("texture_")])
    for tex in textures:
        for sensor in SENSORS:
            p = os.path.join(speed_dir, tex, sensor)
            if not os.path.isdir(p):
                continue
            for fn in os.listdir(p):
                if fn.lower().endswith(".pkl"):
                    pkls.append(os.path.join(p, fn))
    return pkls


def safe_load_one(pkl_path: str):
    """
    最关键：优先用 pandas.read_pickle（带 pandas 自己的 pickle_compat）
    失败后再退回 pickle.load。
    """
    import pandas as pd

    # 1) pandas.read_pickle（强烈推荐）
    try:
        return pd.read_pickle(pkl_path)
    except Exception:
        pass

    # 2) pickle.load 默认
    with open(pkl_path, "rb") as f:
        try:
            return pickle.load(f)
        except UnicodeDecodeError:
            # 兼容 python2 pickle
            f.seek(0)
            return pickle.load(f, encoding="latin1")


def to_numpy(sample):
    """
    把 sample 统一成 (T, C) float32
    支持：DataFrame / dict / ndarray / list
    """
    import pandas as pd

    if isinstance(sample, pd.DataFrame):
        arr = sample.values
        cols = list(sample.columns)
        return arr.astype(np.float32), cols

    if isinstance(sample, dict):
        # 常见 key
        for key in ["imu", "IMU", "data", "signal"]:
            if key in sample:
                v = sample[key]
                if isinstance(v, pd.DataFrame):
                    return v.values.astype(np.float32), list(v.columns)
                arr = np.asarray(v)
                if arr.ndim == 1:
                    arr = arr[:, None]
                return arr.astype(np.float32), [f"c{i}" for i in range(arr.shape[-1])]

        # acc + gyro 合并
        if "acc" in sample and "gyro" in sample:
            acc = np.asarray(sample["acc"])
            gyro = np.asarray(sample["gyro"])
            if acc.ndim == 1:
                acc = acc[:, None]
            if gyro.ndim == 1:
                gyro = gyro[:, None]
            arr = np.concatenate([acc, gyro], axis=-1)
            return arr.astype(np.float32), ["acc_x","acc_y","acc_z","gyro_x","gyro_y","gyro_z"]

        # 兜底：找第一个像数组的值
        for k, v in sample.items():
            try:
                arr = np.asarray(v)
                if arr.ndim >= 1 and arr.size > 0:
                    if arr.ndim == 1:
                        arr = arr[:, None]
                    return arr.astype(np.float32), [f"c{i}" for i in range(arr.shape[-1])]
            except Exception:
                continue

    # ndarray/list
    arr = np.asarray(sample)
    if arr.ndim == 0:
        arr = np.asarray([arr.item()])
    if arr.ndim == 1:
        arr = arr[:, None]
    return arr.astype(np.float32), [f"c{i}" for i in range(arr.shape[-1])]


def save_npz(out_path: str, arr: np.ndarray, cols):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(out_path, imu=arr, cols=np.array(cols, dtype=object))


def main():
    os.makedirs(OUT_ROOT, exist_ok=True)

    # 写日志前清空
    with open(ERROR_LOG, "w", encoding="utf-8") as f:
        f.write("")

    # 一定要先打补丁
    ensure_legacy_pandas_symbols()

    total = 0
    failed = 0

    for sp in SPEEDS:
        sp_dir = os.path.join(DATA_ROOT, sp)
        if not os.path.isdir(sp_dir):
            print("[skip] not found:", sp_dir)
            continue

        pkls = scan_pkls(sp_dir)
        print(f"[scan] {sp}: found {len(pkls)} pkl files")

        for in_path in pkls:
            # out 对齐目录结构：mttd_npz/pickles_30/texture_01/full_imu/imu0101.npz
            rel = os.path.relpath(in_path, DATA_ROOT)
            out_path = os.path.join(OUT_ROOT, os.path.splitext(rel)[0] + ".npz")

            if os.path.exists(out_path):
                continue

            try:
                sample = safe_load_one(in_path)
                arr, cols = to_numpy(sample)

                # 基本 sanity check：IMU 一般是 T x C 且 C>=3
                if arr.ndim != 2:
                    raise ValueError(f"Unexpected imu shape: {arr.shape}")
                save_npz(out_path, arr, cols)
                total += 1

            except Exception as e:
                failed += 1
                with open(ERROR_LOG, "a", encoding="utf-8") as f:
                    f.write(f"\n[FAIL] {in_path}\n{repr(e)}\n")
                    f.write(traceback.format_exc())
                    f.write("\n")

        print("[done]", sp)

    print("=== ALL DONE ===")
    print("total converted:", total)
    print("failed:", failed)
    print("out:", OUT_ROOT)
    print("see error log:", ERROR_LOG)


if __name__ == "__main__":
    main()


