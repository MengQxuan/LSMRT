#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


def infer_label(sample_id: str) -> str:
    m = re.match(r"([A-Za-z_]+)", sample_id)
    if m:
        return m.group(1).lower()
    return sample_id.lower()


def path_str(p: Path, root: Path) -> str:
    try:
        return str(p.relative_to(root))
    except Exception:
        return str(p)


def build_pairs(img_dir: Path, imu_dir: Path, root: Path):
    img_map = {p.stem: p for p in sorted(img_dir.glob("*.jpg"))}
    imu_map = {p.stem: p for p in sorted(imu_dir.glob("*.csv"))}

    shared = sorted(set(img_map.keys()) & set(imu_map.keys()))
    only_img = sorted(set(img_map.keys()) - set(imu_map.keys()))
    only_imu = sorted(set(imu_map.keys()) - set(img_map.keys()))

    rows = []
    for sid in shared:
        rows.append(
            {
                "sample_id": sid,
                "label": infer_label(sid),
                "image_path": path_str(img_map[sid], root),
                "imu_path": path_str(imu_map[sid], root),
            }
        )

    return rows, only_img, only_imu


def stratified_split_per_class(rows, test_ratio: float, seed: int):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    train_rows, test_rows = [], []

    for label in sorted(by_label.keys()):
        items = sorted(by_label[label], key=lambda x: x["sample_id"])
        rng.shuffle(items)

        n = len(items)
        if n <= 1:
            n_test = 0
        else:
            n_test = int(round(n * test_ratio))
            n_test = max(1, n_test)
            n_test = min(n - 1, n_test)

        test_rows.extend(items[:n_test])
        train_rows.extend(items[n_test:])

    return train_rows, test_rows


def add_label_ids(rows, label_to_idx):
    out = []
    for r in rows:
        d = dict(r)
        d["label_id"] = label_to_idx[d["label"]]
        out.append(d)
    return out


def verify_no_leakage(train_rows, test_rows):
    tr_ids = {r["sample_id"] for r in train_rows}
    te_ids = {r["sample_id"] for r in test_rows}
    overlap = tr_ids & te_ids
    if overlap:
        raise RuntimeError(f"Train/Test overlap found: {len(overlap)}")


def save_outputs(all_rows, train_rows, test_rows, out_dir: Path, args):
    labels = sorted({r["label"] for r in all_rows})
    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    all_rows = add_label_ids(all_rows, label_to_idx)
    train_rows = add_label_ids(train_rows, label_to_idx)
    test_rows = add_label_ids(test_rows, label_to_idx)

    verify_no_leakage(train_rows, test_rows)

    for r in all_rows:
        r["split"] = "train" if r["sample_id"] in {x["sample_id"] for x in train_rows} else "test"

    cols = ["sample_id", "label", "label_id", "image_path", "imu_path", "split"]
    df_all = pd.DataFrame(all_rows)[cols]
    df_train = pd.DataFrame(train_rows)[["sample_id", "label", "label_id", "image_path", "imu_path"]]
    df_test = pd.DataFrame(test_rows)[["sample_id", "label", "label_id", "image_path", "imu_path"]]

    df_all = df_all.sort_values("sample_id").reset_index(drop=True)
    df_train = df_train.sort_values("sample_id").reset_index(drop=True)
    df_test = df_test.sort_values("sample_id").reset_index(drop=True)

    df_all.to_csv(out_dir / "all_samples.csv", index=False, encoding="utf-8")
    df_train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    df_test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    label_df = pd.DataFrame(
        [{"label_id": i, "label": lb} for lb, i in label_to_idx.items()]
    ).sort_values("label_id")
    label_df.to_csv(out_dir / "label_map.csv", index=False, encoding="utf-8")

    with open(out_dir / "label_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)

    with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
        for lb in labels:
            f.write(lb + "\n")

    c_all = Counter(df_all["label"].tolist())
    c_train = Counter(df_train["label"].tolist())
    c_test = Counter(df_test["label"].tolist())

    stats_rows = []
    for lb in labels:
        t = c_all[lb]
        tr = c_train[lb]
        te = c_test[lb]
        stats_rows.append(
            {
                "label": lb,
                "label_id": label_to_idx[lb],
                "total": t,
                "train": tr,
                "test": te,
                "test_ratio": float(te / t) if t > 0 else 0.0,
            }
        )

    df_stats = pd.DataFrame(stats_rows).sort_values("label_id").reset_index(drop=True)
    df_stats.to_csv(out_dir / "class_counts.csv", index=False, encoding="utf-8")

    summary = {
        "img_dir": str(args.img_dir),
        "imu_dir": str(args.imu_dir),
        "num_samples_total": int(len(df_all)),
        "num_train": int(len(df_train)),
        "num_test": int(len(df_test)),
        "num_classes": int(len(labels)),
        "test_ratio": args.test_ratio,
        "seed": args.seed,
        "class_distribution": {lb: int(c_all[lb]) for lb in labels},
        "leakage_overlap": 0,
    }

    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Split Done ===")
    print("out_dir:", out_dir)
    print("total/train/test:", len(df_all), len(df_train), len(df_test))
    print("num_classes:", len(labels))


def parse_args():
    p = argparse.ArgumentParser("Prepare aligned image+imu train/test split")
    p.add_argument("--img_dir", type=Path, default=Path("data/aligned-data/img_jpg"))
    p.add_argument("--imu_dir", type=Path, default=Path("data/aligned-data/imu"))
    p.add_argument("--out_dir", type=Path, default=Path("data/aligned-data/split_fusion_v1"))
    p.add_argument("--test_ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()

    img_dir = args.img_dir
    imu_dir = args.imu_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, only_img, only_imu = build_pairs(img_dir, imu_dir, root)
    if not rows:
        raise RuntimeError("No paired samples found.")

    if only_img:
        print(f"[WARN] only image samples: {len(only_img)}")
    if only_imu:
        print(f"[WARN] only imu samples: {len(only_imu)}")

    train_rows, test_rows = stratified_split_per_class(rows, args.test_ratio, args.seed)
    save_outputs(rows, train_rows, test_rows, out_dir, args)


if __name__ == "__main__":
    main()
