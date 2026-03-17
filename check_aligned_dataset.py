import os
import re
import csv
import json
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


IMG_DIR = Path("/root/mqx/LSMRT/data/aligned-data/img")
IMU_DIR = Path("/root/mqx/LSMRT/data/aligned-data/imu")
OUT_DIR = Path("/root/mqx/LSMRT/data/aligned-data/check_results")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PAIR_CSV = OUT_DIR / "pairs.csv"
SUMMARY_JSON = OUT_DIR / "summary.json"

IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".bmp", ".webp"}

# 9轴传感器列
ACC_COLS = ["AccelerationX", "AccelerationY", "AccelerationZ"]
GYRO_COLS = ["GyroX", "GyroY", "GyroZ"]
MAG_COLS = ["MagneticFieldX", "MagneticFieldY", "MagneticFieldZ"]

SENSOR_COLS_6 = ACC_COLS + GYRO_COLS
SENSOR_COLS_9 = ACC_COLS + GYRO_COLS + MAG_COLS


def get_stem(path: Path):
    return path.stem


def infer_label_from_name(stem: str):
    """
    例如:
    carpet01 -> carpet
    tile02   -> tile
    placemat01 -> placemat
    """
    m = re.match(r"([A-Za-z_]+)", stem)
    if m:
        return m.group(1).lower()
    return stem.lower()


def scan_files():
    img_files = {}
    imu_files = {}

    for p in IMG_DIR.iterdir():
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            img_files[get_stem(p)] = p

    for p in IMU_DIR.iterdir():
        if p.is_file() and p.suffix.lower() == ".csv":
            imu_files[get_stem(p)] = p

    return img_files, imu_files


def check_image(img_path: Path):
    try:
        with Image.open(img_path) as img:
            w, h = img.size
            mode = img.mode
        return {"ok": True, "width": w, "height": h, "mode": mode}
    except Exception as e:
        return {"ok": False, "error": str(e)}


def check_imu_csv(csv_path: Path):
    try:
        df = pd.read_csv(csv_path)

        # 去掉列名前后的空格
        df.columns = [c.strip() for c in df.columns]

        info = {
            "ok": True,
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
            "columns": list(df.columns),
        }

        # 尝试推断时间列
        time_col = None
        for c in df.columns:
            c_low = c.lower()
            if "time" in c_low or "timestamp" in c_low:
                time_col = c
                break

        if time_col is not None and len(df) >= 2:
            t0 = df[time_col].iloc[0]
            t1 = df[time_col].iloc[-1]
            info["time_col"] = time_col
            try:
                info["duration"] = float(t1 - t0)
            except Exception:
                info["duration"] = None
        else:
            info["time_col"] = None
            info["duration"] = None

        # 识别实际存在的传感器列
        existing_sensor_cols = [c for c in SENSOR_COLS_9 if c in df.columns]
        info["sensor_cols"] = existing_sensor_cols
        info["num_sensor_cols"] = len(existing_sensor_cols)

        if len(existing_sensor_cols) == 6:
            info["imu_mode"] = "6-axis"
        elif len(existing_sensor_cols) == 9:
            info["imu_mode"] = "9-axis"
        else:
            info["imu_mode"] = f"partial-{len(existing_sensor_cols)}"

        if len(existing_sensor_cols) == 0:
            info["has_nan"] = None
            info["all_zero"] = None
            info["mean_abs"] = None
            info["valid_rows"] = 0
            return info

        # 只处理真正的传感器列，不把时间列算进去
        for c in existing_sensor_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")

        sensor_df = df[existing_sensor_cols].copy()

        # 有效行：至少有一个传感器值不是 NaN
        valid_mask = ~sensor_df.isna().all(axis=1)
        sensor_df_valid = sensor_df[valid_mask].copy()

        info["valid_rows"] = int(sensor_df_valid.shape[0])

        if sensor_df_valid.shape[0] == 0:
            info["has_nan"] = True
            info["all_zero"] = None
            info["mean_abs"] = None
            return info

        # 再看有效部分里是否还有 NaN
        arr = sensor_df_valid.to_numpy(dtype=np.float32)
        info["has_nan"] = bool(np.isnan(arr).any())

        # 这里用 nan_to_num 做稳健判断
        arr_nonan = np.nan_to_num(arr, nan=0.0)
        info["all_zero"] = bool(np.allclose(arr_nonan, 0.0))
        info["mean_abs"] = float(np.mean(np.abs(arr_nonan)))

        # 分三组统计，方便后面判断数据范围
        def safe_group_mean(cols):
            cols_exist = [c for c in cols if c in sensor_df_valid.columns]
            if len(cols_exist) == 0:
                return None
            return float(np.mean(np.abs(sensor_df_valid[cols_exist].to_numpy(dtype=np.float32))))

        info["mean_abs_acc"] = safe_group_mean(ACC_COLS)
        info["mean_abs_gyro"] = safe_group_mean(GYRO_COLS)
        info["mean_abs_mag"] = safe_group_mean(MAG_COLS)

        return info

    except Exception as e:
        return {"ok": False, "error": str(e)}


def save_pairs_csv(records):
    with open(PAIR_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sample_id", "label",
            "image_path", "imu_path",
            "img_ok", "img_width", "img_height", "img_mode",
            "imu_ok", "imu_rows", "imu_valid_rows", "imu_cols",
            "imu_time_col", "imu_duration",
            "imu_mode", "imu_num_sensor_cols", "imu_sensor_cols",
            "imu_has_nan", "imu_all_zero",
            "imu_mean_abs", "imu_mean_abs_acc", "imu_mean_abs_gyro", "imu_mean_abs_mag"
        ])

        for r in records:
            imu_info = r["imu_info"]
            writer.writerow([
                r["sample_id"], r["label"],
                r["image_path"], r["imu_path"],
                r["img_info"].get("ok"),
                r["img_info"].get("width"),
                r["img_info"].get("height"),
                r["img_info"].get("mode"),
                imu_info.get("ok"),
                imu_info.get("rows"),
                imu_info.get("valid_rows"),
                imu_info.get("cols"),
                imu_info.get("time_col"),
                imu_info.get("duration"),
                imu_info.get("imu_mode"),
                imu_info.get("num_sensor_cols"),
                "|".join(imu_info.get("sensor_cols", [])),
                imu_info.get("has_nan"),
                imu_info.get("all_zero"),
                imu_info.get("mean_abs"),
                imu_info.get("mean_abs_acc"),
                imu_info.get("mean_abs_gyro"),
                imu_info.get("mean_abs_mag"),
            ])


def plot_some_imu(records, max_plots=5):
    chosen = records[:max_plots]

    for r in chosen:
        csv_path = Path(r["imu_path"])
        sample_id = r["sample_id"]

        try:
            df = pd.read_csv(csv_path)
            df.columns = [c.strip() for c in df.columns]

            existing_sensor_cols = [c for c in SENSOR_COLS_9 if c in df.columns]
            if len(existing_sensor_cols) == 0:
                print(f"[WARN] {sample_id}: no sensor columns found")
                continue

            for c in existing_sensor_cols:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=existing_sensor_cols, how="all").reset_index(drop=True)

            if len(df) == 0:
                print(f"[WARN] {sample_id}: no valid sensor rows")
                continue

            plt.figure(figsize=(12, 6))
            for c in existing_sensor_cols:
                plt.plot(df[c].values, label=c)

            plt.title(f"IMU Signal - {sample_id}")
            plt.xlabel("timestep")
            plt.ylabel("value")
            plt.legend(ncol=3, fontsize=8)
            plt.tight_layout()

            save_path = OUT_DIR / f"{sample_id}_imu.png"
            plt.savefig(save_path, dpi=150)
            plt.close()
            print(f"[OK] saved plot: {save_path}")

        except Exception as e:
            print(f"[FAIL] plot {sample_id}: {e}")


def main():
    img_files, imu_files = scan_files()

    img_keys = set(img_files.keys())
    imu_keys = set(imu_files.keys())

    common = sorted(img_keys & imu_keys)
    only_img = sorted(img_keys - imu_keys)
    only_imu = sorted(imu_keys - img_keys)

    print("=== Scan Result ===")
    print(f"images found   : {len(img_files)}")
    print(f"imu csv found  : {len(imu_files)}")
    print(f"paired samples : {len(common)}")
    print(f"only in img    : {len(only_img)}")
    print(f"only in imu    : {len(only_imu)}")

    if only_img:
        print("\n[Only in img]")
        for x in only_img:
            print(" ", x)

    if only_imu:
        print("\n[Only in imu]")
        for x in only_imu:
            print(" ", x)

    records = []
    label_count = {}
    imu_mode_count = {}

    for key in common:
        img_path = img_files[key]
        imu_path = imu_files[key]
        label = infer_label_from_name(key)

        img_info = check_image(img_path)
        imu_info = check_imu_csv(imu_path)

        records.append({
            "sample_id": key,
            "label": label,
            "image_path": str(img_path),
            "imu_path": str(imu_path),
            "img_info": img_info,
            "imu_info": imu_info,
        })

        label_count[label] = label_count.get(label, 0) + 1
        imu_mode = imu_info.get("imu_mode", "unknown")
        imu_mode_count[imu_mode] = imu_mode_count.get(imu_mode, 0) + 1

    save_pairs_csv(records)

    summary = {
        "num_images": len(img_files),
        "num_imu": len(imu_files),
        "num_paired": len(common),
        "only_img": only_img,
        "only_imu": only_imu,
        "label_count": label_count,
        "imu_mode_count": imu_mode_count,
    }

    with open(SUMMARY_JSON, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\n=== Label Count ===")
    for k, v in sorted(label_count.items()):
        print(f"{k}: {v}")

    print("\n=== IMU Mode Count ===")
    for k, v in sorted(imu_mode_count.items()):
        print(f"{k}: {v}")

    print(f"\n[OK] pairs saved to: {PAIR_CSV}")
    print(f"[OK] summary saved to: {SUMMARY_JSON}")

    print("\n=== Plotting IMU examples ===")
    plot_some_imu(records, max_plots=min(5, len(records)))


if __name__ == "__main__":
    main()