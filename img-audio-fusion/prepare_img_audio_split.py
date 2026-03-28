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


def collect_by_stem(folder: Path, exts):
    ext_set = {e.lower() for e in exts}
    mp = {}
    for p in sorted(folder.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() not in ext_set:
            continue
        if p.stem not in mp:
            mp[p.stem] = p
    return mp


def split_counts_for_label(n: int, train_ratio: float, val_ratio: float, test_ratio: float):
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 0, 1

    n_val = int(round(n * val_ratio))
    n_test = int(round(n * test_ratio))
    n_val = max(1, n_val)
    n_test = max(1, n_test)

    max_holdout = n - 1
    if n_val + n_test > max_holdout:
        overflow = n_val + n_test - max_holdout
        while overflow > 0:
            if n_test >= n_val and n_test > 1:
                n_test -= 1
            elif n_val > 1:
                n_val -= 1
            elif n_test > 0:
                n_test -= 1
            elif n_val > 0:
                n_val -= 1
            overflow -= 1

    n_train = n - n_val - n_test
    if n_train <= 0:
        n_train = 1
        if n_val >= n_test and n_val > 0:
            n_val -= 1
        elif n_test > 0:
            n_test -= 1
    return n_train, n_val, n_test


def stratified_split(rows, train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    by_label = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    rng = random.Random(seed)
    train_rows, val_rows, test_rows = [], [], []

    for label in sorted(by_label.keys()):
        items = sorted(by_label[label], key=lambda x: x["sample_id"])
        rng.shuffle(items)

        n_train, n_val, n_test = split_counts_for_label(
            n=len(items),
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio,
        )

        train_rows.extend(items[:n_train])
        val_rows.extend(items[n_train : n_train + n_val])
        test_rows.extend(items[n_train + n_val : n_train + n_val + n_test])

    return train_rows, val_rows, test_rows


def add_label_ids(rows, label_to_idx):
    out = []
    for r in rows:
        d = dict(r)
        d["label_id"] = int(label_to_idx[d["label"]])
        out.append(d)
    return out


def verify_no_leakage(train_rows, val_rows, test_rows):
    tr_ids = {r["sample_id"] for r in train_rows}
    va_ids = {r["sample_id"] for r in val_rows}
    te_ids = {r["sample_id"] for r in test_rows}

    overlap_tv = tr_ids & va_ids
    overlap_tt = tr_ids & te_ids
    overlap_vt = va_ids & te_ids
    if overlap_tv or overlap_tt or overlap_vt:
        raise RuntimeError(
            f"Leakage detected: "
            f"train-val={len(overlap_tv)}, train-test={len(overlap_tt)}, val-test={len(overlap_vt)}"
        )


def parse_args():
    p = argparse.ArgumentParser("Prepare aligned image+audio split with 8:1:1 ratio")
    p.add_argument("--img_dir", type=Path, default=Path("data/fusiondata/img"))
    p.add_argument("--audio_dir", type=Path, default=Path("data/fusiondata/audio"))
    p.add_argument(
        "--img_runtime_dir",
        type=Path,
        default=Path("data/fusiondata/img_jpg"),
        help="Actual image directory used for training. If absent, img_dir is used.",
    )
    p.add_argument("--out_dir", type=Path, default=Path("img-audio-fusion/split_fusiondata_8_1_1"))
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    root = Path.cwd()

    img_dir = args.img_dir
    audio_dir = args.audio_dir
    img_runtime_dir = args.img_runtime_dir
    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not img_dir.exists():
        raise FileNotFoundError(f"img_dir not found: {img_dir}")
    if not audio_dir.exists():
        raise FileNotFoundError(f"audio_dir not found: {audio_dir}")

    img_map = collect_by_stem(img_dir, exts=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"])
    audio_map = collect_by_stem(audio_dir, exts=[".wav", ".flac", ".ogg", ".ogx", ".mp3", ".m4a"])
    runtime_map = {}
    if img_runtime_dir and img_runtime_dir.exists():
        runtime_map = collect_by_stem(
            img_runtime_dir,
            exts=[".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".bmp"],
        )

    shared = sorted(set(img_map.keys()) & set(audio_map.keys()))
    only_img = sorted(set(img_map.keys()) - set(audio_map.keys()))
    only_audio = sorted(set(audio_map.keys()) - set(img_map.keys()))

    if not shared:
        raise RuntimeError("No aligned pairs found between img_dir and audio_dir.")

    rows = []
    missing_runtime = []
    for sid in shared:
        raw_img_path = img_map[sid]
        train_img_path = runtime_map.get(sid, raw_img_path)
        if runtime_map and sid not in runtime_map:
            missing_runtime.append(sid)

        rows.append(
            {
                "sample_id": sid,
                "label": infer_label(sid),
                "image_path": path_str(train_img_path, root),
                "audio_path": path_str(audio_map[sid], root),
                "image_raw_path": path_str(raw_img_path, root),
            }
        )

    if runtime_map and missing_runtime:
        raise RuntimeError(
            f"img_runtime_dir misses {len(missing_runtime)} paired samples; "
            f"examples={missing_runtime[:10]}"
        )

    train_rows, val_rows, test_rows = stratified_split(
        rows=rows,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    labels = sorted({r["label"] for r in rows})
    label_to_idx = {lb: i for i, lb in enumerate(labels)}

    rows = add_label_ids(rows, label_to_idx)
    train_rows = add_label_ids(train_rows, label_to_idx)
    val_rows = add_label_ids(val_rows, label_to_idx)
    test_rows = add_label_ids(test_rows, label_to_idx)
    verify_no_leakage(train_rows, val_rows, test_rows)

    split_map = {r["sample_id"]: "train" for r in train_rows}
    split_map.update({r["sample_id"]: "val" for r in val_rows})
    split_map.update({r["sample_id"]: "test" for r in test_rows})
    for r in rows:
        r["split"] = split_map[r["sample_id"]]

    cols_all = ["sample_id", "label", "label_id", "image_path", "audio_path", "image_raw_path", "split"]
    cols_split = ["sample_id", "label", "label_id", "image_path", "audio_path", "image_raw_path"]

    df_all = pd.DataFrame(rows)[cols_all].sort_values("sample_id").reset_index(drop=True)
    df_train = pd.DataFrame(train_rows)[cols_split].sort_values("sample_id").reset_index(drop=True)
    df_val = pd.DataFrame(val_rows)[cols_split].sort_values("sample_id").reset_index(drop=True)
    df_test = pd.DataFrame(test_rows)[cols_split].sort_values("sample_id").reset_index(drop=True)

    df_all.to_csv(out_dir / "all_samples.csv", index=False, encoding="utf-8")
    df_train.to_csv(out_dir / "train.csv", index=False, encoding="utf-8")
    df_val.to_csv(out_dir / "val.csv", index=False, encoding="utf-8")
    df_test.to_csv(out_dir / "test.csv", index=False, encoding="utf-8")

    pd.DataFrame([{"label_id": i, "label": lb} for lb, i in label_to_idx.items()]).sort_values(
        "label_id"
    ).to_csv(out_dir / "label_map.csv", index=False, encoding="utf-8")

    with open(out_dir / "label_to_idx.json", "w", encoding="utf-8") as f:
        json.dump(label_to_idx, f, ensure_ascii=False, indent=2)

    with open(out_dir / "labels.txt", "w", encoding="utf-8") as f:
        for lb in labels:
            f.write(lb + "\n")

    c_all = Counter(df_all["label"].tolist())
    c_tr = Counter(df_train["label"].tolist())
    c_va = Counter(df_val["label"].tolist())
    c_te = Counter(df_test["label"].tolist())

    stats_rows = []
    for lb in labels:
        t = c_all[lb]
        tr = c_tr[lb]
        va = c_va[lb]
        te = c_te[lb]
        stats_rows.append(
            {
                "label": lb,
                "label_id": label_to_idx[lb],
                "total": t,
                "train": tr,
                "val": va,
                "test": te,
                "train_ratio": float(tr / t) if t > 0 else 0.0,
                "val_ratio": float(va / t) if t > 0 else 0.0,
                "test_ratio": float(te / t) if t > 0 else 0.0,
            }
        )
    pd.DataFrame(stats_rows).sort_values("label_id").reset_index(drop=True).to_csv(
        out_dir / "class_counts.csv", index=False, encoding="utf-8"
    )

    summary = {
        "img_dir": str(img_dir),
        "audio_dir": str(audio_dir),
        "img_runtime_dir": str(img_runtime_dir) if img_runtime_dir else "",
        "num_img_samples": len(img_map),
        "num_audio_samples": len(audio_map),
        "num_paired_samples": len(shared),
        "num_only_img": len(only_img),
        "num_only_audio": len(only_audio),
        "num_train": len(df_train),
        "num_val": len(df_val),
        "num_test": len(df_test),
        "num_classes": len(labels),
        "ratios_target": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "class_distribution": {lb: int(c_all[lb]) for lb in labels},
    }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("=== Split Done ===")
    print("out_dir:", out_dir)
    print("paired total:", len(shared))
    print("train/val/test:", len(df_train), len(df_val), len(df_test))
    print("num_classes:", len(labels))
    if only_img:
        print(f"[WARN] only image samples: {len(only_img)}")
    if only_audio:
        print(f"[WARN] only audio samples: {len(only_audio)}")


if __name__ == "__main__":
    main()
