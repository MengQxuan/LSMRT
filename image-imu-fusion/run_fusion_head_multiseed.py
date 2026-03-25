#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd


def parse_seeds(seeds_text: str):
    out = []
    for x in seeds_text.split(','):
        x = x.strip()
        if not x:
            continue
        out.append(int(x))
    if not out:
        raise ValueError("No valid seeds provided.")
    return out


def run_one_seed(seed: int, args, root: Path):
    run_out_dir = (root / args.base_out_dir / f"seed{seed}").resolve()
    run_out_dir.mkdir(parents=True, exist_ok=True)

    # Disable early stopping by setting patience > epochs.
    patience = args.epochs + 1

    cmd = [
        sys.executable,
        str((root / "scripts" / "train_fusion_head.py").resolve()),
        "--split_dir",
        str(args.split_dir),
        "--image_ckpt",
        str(args.image_ckpt),
        "--imu_ckpt",
        str(args.imu_ckpt),
        "--out_dir",
        str(run_out_dir),
        "--img_size",
        str(args.img_size),
        "--imu_len",
        str(args.imu_len),
        "--val_ratio",
        str(args.val_ratio),
        "--epochs",
        str(args.epochs),
        "--batch",
        str(args.batch),
        "--lr",
        str(args.lr),
        "--wd",
        str(args.wd),
        "--dropout",
        str(args.dropout),
        "--hidden_dim",
        str(args.hidden_dim),
        "--label_smoothing",
        str(args.label_smoothing),
        "--patience",
        str(patience),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(seed),
    ]

    if args.use_cuda:
        cmd.append("--use_cuda")

    print("\n=== RUN seed={} ===".format(seed))
    print("CMD:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(root))

    metrics_path = run_out_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.json not found for seed {seed}: {metrics_path}")

    with open(metrics_path, "r", encoding="utf-8") as f:
        m = json.load(f)

    row = {
        "seed": seed,
        "run_dir": str(run_out_dir),
        "best_model_path": str((run_out_dir / "best_head.pt").resolve()),
        "best_epoch": m.get("best_epoch"),
        "best_val_acc": m.get("best_val_acc"),
        "best_val_f1": m.get("best_val_f1"),
        "test_acc": m.get("test_acc"),
        "test_macro_f1": m.get("test_macro_f1"),
        "num_train_total": m.get("num_train_total"),
        "num_test": m.get("num_test"),
        "num_classes": m.get("num_classes"),
        "feature_dim": m.get("feature_dim"),
        "device": m.get("device"),
    }
    return row


def main():
    ap = argparse.ArgumentParser("Batch run fusion-head training across multiple seeds and summarize mean/std")

    ap.add_argument("--seeds", type=str, default="41,42,43,44,45")
    ap.add_argument("--base_out_dir", type=Path, default=Path("runs/fusion_head_multiseed"))

    ap.add_argument("--split_dir", type=Path, default=Path("data/aligned-data/split_fusion_v1"))
    ap.add_argument("--image_ckpt", type=Path, default=Path("image-branch/runs/effb0_s224_split1_resize/best.pt"))
    ap.add_argument("--imu_ckpt", type=Path, default=Path("imu-branch/runs/imu_trial_split_cnn/best.pt"))

    ap.add_argument("--img_size", type=int, default=224)
    ap.add_argument("--imu_len", type=int, default=256)
    ap.add_argument("--val_ratio", type=float, default=0.2)

    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--dropout", type=float, default=0.3)
    ap.add_argument("--hidden_dim", type=int, default=512)
    ap.add_argument("--label_smoothing", type=float, default=0.05)
    ap.add_argument("--num_workers", type=int, default=0)

    ap.add_argument("--use_cuda", action="store_true")
    ap.add_argument(
        "--best_by",
        type=str,
        default="test_macro_f1",
        choices=["test_macro_f1", "test_acc", "best_val_f1", "best_val_acc"],
    )
    args = ap.parse_args()

    root = Path.cwd()
    seeds = parse_seeds(args.seeds)

    rows = []
    for seed in seeds:
        row = run_one_seed(seed, args, root)
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("seed").reset_index(drop=True)

    out_dir = (root / args.base_out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    runs_csv = out_dir / "runs_summary.csv"
    df.to_csv(runs_csv, index=False, encoding="utf-8")

    metric_cols = ["test_acc", "test_macro_f1", "best_val_acc", "best_val_f1"]
    stats = {}
    for c in metric_cols:
        stats[c] = {
            "mean": float(df[c].mean()),
            "std": float(df[c].std(ddof=1) if len(df) > 1 else 0.0),
            "min": float(df[c].min()),
            "max": float(df[c].max()),
        }

    # Pick best run by selected metric; tie-break by test_acc then best_val_f1.
    best_df = df.sort_values([args.best_by, "test_acc", "best_val_f1"], ascending=False).reset_index(drop=True)
    best = best_df.iloc[0].to_dict()

    best_model_src = Path(best["best_model_path"])
    best_model_dst = out_dir / "best_model_overall.pt"
    if best_model_src.exists():
        shutil.copy2(best_model_src, best_model_dst)

    summary = {
        "seeds": seeds,
        "num_runs": len(seeds),
        "best_by": args.best_by,
        "aggregate": stats,
        "best_run": best,
        "best_model_copied_to": str(best_model_dst.resolve()),
        "runs_csv": str(runs_csv.resolve()),
    }

    summary_json = out_dir / "aggregate_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    best_info_txt = out_dir / "best_model_info.txt"
    with open(best_info_txt, "w", encoding="utf-8") as f:
        f.write("Best model info\n")
        f.write("==============\n")
        f.write(f"best_by: {args.best_by}\n")
        f.write(f"seed: {int(best['seed'])}\n")
        f.write(f"run_dir: {best['run_dir']}\n")
        f.write(f"best_model_path: {best['best_model_path']}\n")
        f.write(f"test_acc: {best['test_acc']:.6f}\n")
        f.write(f"test_macro_f1: {best['test_macro_f1']:.6f}\n")
        f.write(f"best_val_acc: {best['best_val_acc']:.6f}\n")
        f.write(f"best_val_f1: {best['best_val_f1']:.6f}\n")
        f.write("\nAggregate mean/std\n")
        for c in metric_cols:
            f.write(f"{c}: mean={stats[c]['mean']:.6f}, std={stats[c]['std']:.6f}\n")

    print("\n=== ALL DONE ===")
    print("runs summary:", runs_csv)
    print("aggregate summary:", summary_json)
    print("best model info:", best_info_txt)
    print("best model copied:", best_model_dst)
    print(
        "best run: seed={} test_acc={:.4f} test_macro_f1={:.4f}"
        .format(int(best["seed"]), best["test_acc"], best["test_macro_f1"])
    )
    print(
        "mean±std: test_acc={:.4f}±{:.4f}, test_macro_f1={:.4f}±{:.4f}"
        .format(
            stats["test_acc"]["mean"],
            stats["test_acc"]["std"],
            stats["test_macro_f1"]["mean"],
            stats["test_macro_f1"]["std"],
        )
    )


if __name__ == "__main__":
    main()
