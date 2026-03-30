#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import train_fusion_head as tfh


def parse_int_list(raw: str) -> List[int]:
    vals: List[int] = []
    for part in raw.replace(",", " ").split():
        part = part.strip()
        if not part:
            continue
        vals.append(int(part))
    if not vals:
        raise ValueError("seed list is empty")
    return vals


def mean_std(arr: List[float]) -> Dict[str, float]:
    if not arr:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    x = np.asarray(arr, dtype=np.float64)
    return {
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]):
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def maybe_prepare_split(args, split_seed: int, split_dir: Path, env: Dict[str, str]):
    need_files = [split_dir / "train.csv", split_dir / "val.csv", split_dir / "test.csv"]
    ready = all(p.exists() for p in need_files)
    if ready and not args.force_prepare:
        print(f"[SKIP] split exists: {split_dir}", flush=True)
        return

    split_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str((args.repo_root / "img-audio-fusion" / "prepare_img_audio_split.py").resolve()),
        "--img_dir",
        str(args.img_dir),
        "--audio_dir",
        str(args.audio_dir),
        "--img_runtime_dir",
        str(args.img_runtime_dir),
        "--out_dir",
        str(split_dir),
        "--train_ratio",
        str(args.train_ratio),
        "--val_ratio",
        str(args.val_ratio),
        "--test_ratio",
        str(args.test_ratio),
        "--seed",
        str(split_seed),
    ]
    run_cmd(cmd, cwd=args.repo_root, env=env)


def maybe_train_one(
    args,
    split_seed: int,
    train_seed: int,
    split_dir: Path,
    run_dir: Path,
    env: Dict[str, str],
):
    metrics_path = run_dir / "metrics.json"
    if metrics_path.exists() and not args.force_retrain:
        print(f"[SKIP] trained run exists: {run_dir}", flush=True)
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str((args.repo_root / "img-audio-fusion" / "train_fusion_head.py").resolve()),
        "--split_dir",
        str(split_dir),
        "--image_ckpt",
        str(args.image_ckpt),
        "--audio_ckpt",
        str(args.audio_ckpt),
        "--out_dir",
        str(run_dir),
        "--img_size",
        str(args.img_size),
        "--target_sr",
        str(args.target_sr),
        "--clip_sec",
        str(args.clip_sec),
        "--n_fft",
        str(args.n_fft),
        "--hop_length",
        str(args.hop_length),
        "--n_mels",
        str(args.n_mels),
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
        str(args.patience),
        "--num_workers",
        str(args.num_workers),
        "--seed",
        str(train_seed),
    ]
    if not args.cpu:
        cmd.append("--use_cuda")
        if not args.no_data_parallel:
            cmd.append("--use_data_parallel")
    run_cmd(cmd, cwd=args.repo_root, env=env)


def load_test_df(split_dir: Path, repo_root: Path) -> pd.DataFrame:
    test_csv = split_dir / "test.csv"
    if not test_csv.exists():
        raise FileNotFoundError(f"missing test csv: {test_csv}")
    df = pd.read_csv(test_csv)
    for col in ["sample_id", "label", "label_id", "image_path", "audio_path"]:
        if col not in df.columns:
            raise RuntimeError(f"missing column {col} in {test_csv}")
    df["image_path"] = df["image_path"].astype(str).apply(lambda x: tfh.resolve_path(x, repo_root))
    df["audio_path"] = df["audio_path"].astype(str).apply(lambda x: tfh.resolve_path(x, repo_root))
    return df


def build_feature_extractors(args, device: torch.device, use_dp: bool):
    img_backbone = tfh.ImageEffB0Backbone(str(args.image_ckpt)).to(device)
    audio_backbone = tfh.AudioBackboneFromCkpt(
        str(args.audio_ckpt),
        target_sr_override=args.target_sr,
        n_fft_override=args.n_fft,
        hop_length_override=args.hop_length,
        n_mels_override=args.n_mels,
    ).to(device)

    if use_dp:
        img_backbone = nn.DataParallel(img_backbone)
        audio_backbone = nn.DataParallel(audio_backbone)

    for p in img_backbone.parameters():
        p.requires_grad = False
    for p in audio_backbone.parameters():
        p.requires_grad = False

    return img_backbone, audio_backbone


@torch.no_grad()
def eval_single_head(
    head_ckpt: Path,
    X_test: torch.Tensor,
    y_true: np.ndarray,
    device: torch.device,
    batch: int,
) -> Tuple[torch.Tensor, Dict[str, float], Dict[int, str]]:
    ck = torch.load(head_ckpt, map_location="cpu")
    model = tfh.FusionHead(
        in_dim=int(ck["input_dim"]),
        num_classes=int(ck["num_classes"]),
        hidden_dim=int(ck.get("hidden_dim", 512)),
        dropout=float(ck.get("dropout", 0.3)),
    ).to(device)
    model.load_state_dict(ck["model"], strict=True)

    logits = tfh.predict_logits(model, X_test, device=device, batch=batch).cpu()
    acc, f1, _, pred, conf = tfh.eval_from_logits(y_true, logits)
    out = {
        "test_acc": float(acc),
        "test_macro_f1": float(f1),
        "avg_confidence": float(np.mean(conf)) if len(conf) else 0.0,
    }
    idx_to_label = {int(k): str(v) for k, v in ck.get("idx_to_label", {}).items()}
    return logits, out, idx_to_label


def ensemble_eval_and_dump(
    split_seed: int,
    split_dir: Path,
    run_dirs: List[Path],
    args,
    device: torch.device,
    use_dp: bool,
    out_dir: Path,
):
    df_test = load_test_df(split_dir, args.repo_root)
    y_true = df_test["label_id"].astype(int).to_numpy()

    ds_test = tfh.PairedImageAudioDataset(df_test, args.img_size, args.target_sr, args.clip_sec)
    dl_test = DataLoader(
        ds_test,
        batch_size=args.batch,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        persistent_workers=(args.num_workers > 0),
    )

    img_backbone, audio_backbone = build_feature_extractors(args, device=device, use_dp=use_dp)
    X_test, y_test, sid_test = tfh.extract_features(img_backbone, audio_backbone, dl_test, device, desc=f"split{split_seed}_test")
    y_true = y_test.numpy()

    logits_all = []
    per_model = []
    idx_to_label = None

    for run_dir in run_dirs:
        head_ckpt = run_dir / "best_head.pt"
        if not head_ckpt.exists():
            raise FileNotFoundError(f"missing head checkpoint: {head_ckpt}")
        logits, metrics, cur_idx_to_label = eval_single_head(
            head_ckpt=head_ckpt,
            X_test=X_test,
            y_true=y_true,
            device=device,
            batch=max(256, args.batch),
        )
        if idx_to_label is None and cur_idx_to_label:
            idx_to_label = cur_idx_to_label
        logits_all.append(logits)
        per_model.append(
            {
                "run_dir": str(run_dir),
                "test_acc_recomputed": metrics["test_acc"],
                "test_macro_f1_recomputed": metrics["test_macro_f1"],
                "avg_confidence_recomputed": metrics["avg_confidence"],
            }
        )

    if not logits_all:
        raise RuntimeError(f"no runs for split seed {split_seed}")

    prob_list = [torch.softmax(lg, dim=1) for lg in logits_all]
    ens_prob = torch.stack(prob_list, dim=0).mean(dim=0)
    ens_pred = torch.argmax(ens_prob, dim=1).numpy()

    ens_acc = float(accuracy_score(y_true, ens_pred))
    ens_f1 = float(f1_score(y_true, ens_pred, average="macro"))
    ens_cm = confusion_matrix(y_true, ens_pred).tolist()
    ens_conf = ens_prob.max(dim=1).values.numpy()

    if idx_to_label is None:
        idx_to_label = {int(i): str(i) for i in sorted(np.unique(y_true))}

    pred_rows = []
    df_sid = df_test.set_index("sample_id")
    for sid, yt, yp, cf in zip(sid_test, y_true.tolist(), ens_pred.tolist(), ens_conf.tolist()):
        r = df_sid.loc[sid]
        pred_rows.append(
            {
                "sample_id": sid,
                "label_true": idx_to_label.get(int(yt), str(yt)),
                "label_pred": idx_to_label.get(int(yp), str(yp)),
                "label_id_true": int(yt),
                "label_id_pred": int(yp),
                "confidence": float(cf),
                "image_path": r["image_path"],
                "audio_path": r["audio_path"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / f"ensemble_test_predictions_split{split_seed}.csv"
    pd.DataFrame(pred_rows).to_csv(pred_csv, index=False, encoding="utf-8")

    return {
        "split_seed": int(split_seed),
        "num_models": int(len(run_dirs)),
        "num_test": int(len(y_true)),
        "ensemble_test_acc": ens_acc,
        "ensemble_test_macro_f1": ens_f1,
        "ensemble_avg_confidence": float(np.mean(ens_conf)) if len(ens_conf) else 0.0,
        "ensemble_confusion_matrix": ens_cm,
        "ensemble_predictions_csv": str(pred_csv),
        "per_model_recomputed": per_model,
    }


def parse_args():
    p = argparse.ArgumentParser("Multi-seed retrain + mean/std summary + ensemble evaluation")

    p.add_argument("--repo_root", type=Path, default=Path.cwd())
    p.add_argument("--img_dir", type=Path, default=Path("data/fusiondata/img"))
    p.add_argument("--audio_dir", type=Path, default=Path("data/fusiondata/audio"))
    p.add_argument("--img_runtime_dir", type=Path, default=Path("data/fusiondata/img_jpg"))

    p.add_argument("--split_seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--train_seeds", type=str, default="42,43,44")
    p.add_argument("--train_ratio", type=float, default=0.8)
    p.add_argument("--val_ratio", type=float, default=0.1)
    p.add_argument("--test_ratio", type=float, default=0.1)

    p.add_argument("--split_out_root", type=Path, default=Path("img-audio-fusion/multi_seed/splits"))
    p.add_argument("--run_out_root", type=Path, default=Path("img-audio-fusion/multi_seed/runs"))
    p.add_argument("--summary_out_dir", type=Path, default=Path("img-audio-fusion/multi_seed/summary"))

    p.add_argument("--image_ckpt", type=Path, default=Path("image-branch/runs/effb0_s224_split1_resize/best.pt"))
    p.add_argument(
        "--audio_ckpt",
        type=Path,
        default=Path("audio-branch/runs/horizontal_robot_impact_sr24000_clip1.0_b32x2_e50_pretrain_invSqrt_c5/best.pt"),
    )

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--target_sr", type=int, default=24000)
    p.add_argument("--clip_sec", type=float, default=1.0)
    p.add_argument("--n_fft", type=int, default=0)
    p.add_argument("--hop_length", type=int, default=0)
    p.add_argument("--n_mels", type=int, default=0)

    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=25)
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--cuda_visible_devices", type=str, default="")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--no_data_parallel", action="store_true")

    p.add_argument("--force_prepare", action="store_true")
    p.add_argument("--force_retrain", action="store_true")
    return p.parse_args()


def main():
    t0 = time.time()
    args = parse_args()
    args.repo_root = args.repo_root.resolve()
    args.img_dir = (args.repo_root / args.img_dir).resolve()
    args.audio_dir = (args.repo_root / args.audio_dir).resolve()
    args.img_runtime_dir = (args.repo_root / args.img_runtime_dir).resolve()
    args.split_out_root = (args.repo_root / args.split_out_root).resolve()
    args.run_out_root = (args.repo_root / args.run_out_root).resolve()
    args.summary_out_dir = (args.repo_root / args.summary_out_dir).resolve()
    args.image_ckpt = (args.repo_root / args.image_ckpt).resolve()
    args.audio_ckpt = (args.repo_root / args.audio_ckpt).resolve()

    split_seeds = parse_int_list(args.split_seeds)
    train_seeds = parse_int_list(args.train_seeds)

    args.split_out_root.mkdir(parents=True, exist_ok=True)
    args.run_out_root.mkdir(parents=True, exist_ok=True)
    args.summary_out_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    if args.cuda_visible_devices:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    train_records = []
    run_dir_map: Dict[int, List[Path]] = {}

    # Stage 1: prepare split + train all runs
    for split_seed in split_seeds:
        split_dir = args.split_out_root / f"split_seed{split_seed}_8_1_1"
        maybe_prepare_split(args=args, split_seed=split_seed, split_dir=split_dir, env=env)

        run_dir_map[split_seed] = []
        for train_seed in train_seeds:
            run_dir = args.run_out_root / f"split{split_seed}_train{train_seed}"
            maybe_train_one(
                args=args,
                split_seed=split_seed,
                train_seed=train_seed,
                split_dir=split_dir,
                run_dir=run_dir,
                env=env,
            )
            run_dir_map[split_seed].append(run_dir)

            metrics_path = run_dir / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"missing metrics after training: {metrics_path}")
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            train_records.append(
                {
                    "split_seed": int(split_seed),
                    "train_seed": int(train_seed),
                    "run_dir": str(run_dir),
                    "best_epoch": int(m.get("best_epoch", -1)),
                    "best_val_acc": float(m.get("best_val_acc", 0.0)),
                    "best_val_f1": float(m.get("best_val_f1", 0.0)),
                    "test_acc": float(m.get("test_acc", 0.0)),
                    "test_macro_f1": float(m.get("test_macro_f1", 0.0)),
                    "elapsed_sec": float(m.get("elapsed_sec", 0.0)),
                }
            )

    # Stage 2: ensemble eval for each split seed
    use_cuda = (not args.cpu) and torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    num_gpu = torch.cuda.device_count() if use_cuda else 0
    use_dp = bool(use_cuda and (not args.no_data_parallel) and num_gpu > 1)
    if use_cuda:
        torch.backends.cudnn.benchmark = True

    ensemble_records = []
    for split_seed in split_seeds:
        split_dir = args.split_out_root / f"split_seed{split_seed}_8_1_1"
        rec = ensemble_eval_and_dump(
            split_seed=split_seed,
            split_dir=split_dir,
            run_dirs=run_dir_map[split_seed],
            args=args,
            device=device,
            use_dp=use_dp,
            out_dir=args.summary_out_dir,
        )
        ensemble_records.append(rec)

    # Stage 3: aggregate summary
    df_runs = pd.DataFrame(train_records)
    per_split = {}
    for split_seed in split_seeds:
        sub = df_runs[df_runs["split_seed"] == split_seed]
        ens = [x for x in ensemble_records if x["split_seed"] == split_seed][0]
        per_split[str(split_seed)] = {
            "num_runs": int(len(sub)),
            "single_test_acc": mean_std(sub["test_acc"].tolist()),
            "single_test_macro_f1": mean_std(sub["test_macro_f1"].tolist()),
            "single_best_val_acc": mean_std(sub["best_val_acc"].tolist()),
            "single_best_val_f1": mean_std(sub["best_val_f1"].tolist()),
            "ensemble_test_acc": float(ens["ensemble_test_acc"]),
            "ensemble_test_macro_f1": float(ens["ensemble_test_macro_f1"]),
            "ensemble_predictions_csv": ens["ensemble_predictions_csv"],
        }

    summary = {
        "time": {
            "start_unix": float(t0),
            "elapsed_sec_total": float(time.time() - t0),
        },
        "config": {
            "repo_root": str(args.repo_root),
            "split_seeds": split_seeds,
            "train_seeds": train_seeds,
            "img_dir": str(args.img_dir),
            "audio_dir": str(args.audio_dir),
            "img_runtime_dir": str(args.img_runtime_dir),
            "image_ckpt": str(args.image_ckpt),
            "audio_ckpt": str(args.audio_ckpt),
            "device": str(device),
            "num_gpu": int(num_gpu),
            "use_data_parallel": bool(use_dp),
            "hyperparams": {
                "img_size": args.img_size,
                "target_sr": args.target_sr,
                "clip_sec": args.clip_sec,
                "n_fft": args.n_fft,
                "hop_length": args.hop_length,
                "n_mels": args.n_mels,
                "epochs": args.epochs,
                "batch": args.batch,
                "lr": args.lr,
                "wd": args.wd,
                "dropout": args.dropout,
                "hidden_dim": args.hidden_dim,
                "label_smoothing": args.label_smoothing,
                "patience": args.patience,
                "num_workers": args.num_workers,
            },
        },
        "single_run_aggregate": {
            "test_acc": mean_std(df_runs["test_acc"].tolist()),
            "test_macro_f1": mean_std(df_runs["test_macro_f1"].tolist()),
            "best_val_acc": mean_std(df_runs["best_val_acc"].tolist()),
            "best_val_f1": mean_std(df_runs["best_val_f1"].tolist()),
            "elapsed_sec": mean_std(df_runs["elapsed_sec"].tolist()),
        },
        "ensemble_aggregate": {
            "test_acc": mean_std([x["ensemble_test_acc"] for x in ensemble_records]),
            "test_macro_f1": mean_std([x["ensemble_test_macro_f1"] for x in ensemble_records]),
        },
        "per_split": per_split,
        "runs": train_records,
        "ensemble_runs": ensemble_records,
    }

    summary_json = args.summary_out_dir / "multi_seed_summary.json"
    summary_txt = args.summary_out_dir / "multi_seed_summary.txt"
    runs_csv = args.summary_out_dir / "single_runs_metrics.csv"

    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    df_runs.to_csv(runs_csv, index=False, encoding="utf-8")

    lines = []
    lines.append("Multi-Seed Retrain + Ensemble Summary")
    lines.append(f"split_seeds: {split_seeds}")
    lines.append(f"train_seeds: {train_seeds}")
    lines.append(f"device: {device}, num_gpu: {num_gpu}, use_data_parallel: {use_dp}")
    lines.append("")
    agg = summary["single_run_aggregate"]
    lines.append(
        "Single-run test_acc mean±std: "
        f"{agg['test_acc']['mean']:.4f} ± {agg['test_acc']['std']:.4f}"
    )
    lines.append(
        "Single-run test_macro_f1 mean±std: "
        f"{agg['test_macro_f1']['mean']:.4f} ± {agg['test_macro_f1']['std']:.4f}"
    )
    eagg = summary["ensemble_aggregate"]
    lines.append(
        "Ensemble test_acc mean±std: "
        f"{eagg['test_acc']['mean']:.4f} ± {eagg['test_acc']['std']:.4f}"
    )
    lines.append(
        "Ensemble test_macro_f1 mean±std: "
        f"{eagg['test_macro_f1']['mean']:.4f} ± {eagg['test_macro_f1']['std']:.4f}"
    )
    lines.append("")
    lines.append("Per split:")
    for k in sorted(per_split.keys(), key=lambda x: int(x)):
        s = per_split[k]
        lines.append(
            f"  split_seed={k}: "
            f"single_acc_mean={s['single_test_acc']['mean']:.4f}, "
            f"ensemble_acc={s['ensemble_test_acc']:.4f}, "
            f"ensemble_f1={s['ensemble_test_macro_f1']:.4f}"
        )
    lines.append("")
    lines.append(f"summary_json: {summary_json}")
    lines.append(f"summary_txt: {summary_txt}")
    lines.append(f"runs_csv: {runs_csv}")

    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()

