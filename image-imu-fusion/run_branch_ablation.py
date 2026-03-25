#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

import pandas as pd

import torch

from train_fusion_head import (
    resolve_path,
    stratified_split_train_val,
    compute_train_imu_norm,
    PairedAlignedDataset,
    ImageEffB0Backbone,
    IMUBackboneFromCkpt,
    extract_features,
    train_head,
    predict_logits,
    eval_from_logits,
    set_seed,
)


def parse_args():
    p = argparse.ArgumentParser("Run Image-only / IMU-only / Fusion ablation on aligned split")

    p.add_argument("--split_dir", type=Path, default=Path("data/aligned-data/split_fusion_v1"))
    p.add_argument(
        "--image_ckpt",
        type=Path,
        default=Path("image-branch/runs/effb0_s224_split1_resize/best.pt"),
    )
    p.add_argument(
        "--imu_ckpt",
        type=Path,
        default=Path("imu-branch/runs/imu_trial_split_cnn/best.pt"),
    )
    p.add_argument("--out_dir", type=Path, default=Path("runs/branch_ablation"))

    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--imu_len", type=int, default=256)
    p.add_argument("--val_ratio", type=float, default=0.2)

    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--label_smoothing", type=float, default=0.05)
    p.add_argument("--patience", type=int, default=12)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=43)
    p.add_argument("--use_cuda", action="store_true")
    return p.parse_args()


def build_mode_predictions(out_csv: Path, sid_test, y_true, pred, conf, idx_to_label, test_df):
    rows = []
    test_sid = test_df.set_index("sample_id")

    for sid, yt, yp, cf in zip(sid_test, y_true.tolist(), pred.tolist(), conf.tolist()):
        r = test_sid.loc[sid]
        rows.append(
            {
                "sample_id": sid,
                "label_true": idx_to_label[int(yt)],
                "label_pred": idx_to_label[int(yp)],
                "label_id_true": int(yt),
                "label_id_pred": int(yp),
                "confidence": float(cf),
                "image_path": r["image_path"],
                "imu_path": r["imu_path"],
                "correct": int(int(yt) == int(yp)),
            }
        )

    pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")


def run_mode(mode_name, X_train, y_train, X_val, y_val, X_test, y_test, sid_test, idx_to_label, test_df, args, device, out_dir):
    mode_out_dir = out_dir / mode_name
    mode_out_dir.mkdir(parents=True, exist_ok=True)

    model, best, history = train_head(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        num_classes=int(y_train.max().item() + 1),
        args=args,
        device=device,
    )

    test_logits = predict_logits(model, X_test, device=device, batch=max(args.batch, 256))
    test_acc, test_f1, test_cm, test_pred, test_conf = eval_from_logits(y_test.numpy(), test_logits)

    model_path = mode_out_dir / "best_head.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "mode": mode_name,
            "input_dim": int(X_train.shape[1]),
            "num_classes": int(y_train.max().item() + 1),
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "idx_to_label": idx_to_label,
        },
        model_path,
    )

    with open(mode_out_dir / "history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)

    with open(mode_out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "mode": mode_name,
                "best_epoch": int(best["epoch"]),
                "best_val_acc": float(best["val_acc"]),
                "best_val_f1": float(best["val_f1"]),
                "test_acc": float(test_acc),
                "test_macro_f1": float(test_f1),
                "test_confusion_matrix": test_cm.tolist(),
                "model_path": str(model_path.resolve()),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    build_mode_predictions(
        out_csv=mode_out_dir / "test_predictions.csv",
        sid_test=sid_test,
        y_true=y_test.numpy(),
        pred=test_pred,
        conf=test_conf,
        idx_to_label=idx_to_label,
        test_df=test_df,
    )

    return {
        "mode": mode_name,
        "best_epoch": int(best["epoch"]),
        "best_val_acc": float(best["val_acc"]),
        "best_val_f1": float(best["val_f1"]),
        "test_acc": float(test_acc),
        "test_macro_f1": float(test_f1),
        "model_path": str(model_path.resolve()),
    }


def main():
    args = parse_args()
    set_seed(args.seed)

    root = Path.cwd()
    split_dir = (root / args.split_dir).resolve()
    out_dir = (root / args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    train_csv = split_dir / "train.csv"
    test_csv = split_dir / "test.csv"
    if not train_csv.exists() or not test_csv.exists():
        raise FileNotFoundError(f"train.csv/test.csv not found in {split_dir}")

    df_train = pd.read_csv(train_csv)
    df_test = pd.read_csv(test_csv)

    req_cols = ["sample_id", "label", "label_id", "image_path", "imu_path"]
    for c in req_cols:
        if c not in df_train.columns or c not in df_test.columns:
            raise RuntimeError(f"Missing required column: {c}")

    for df in (df_train, df_test):
        df["image_path"] = df["image_path"].astype(str).apply(lambda x: resolve_path(x, root))
        df["imu_path"] = df["imu_path"].astype(str).apply(lambda x: resolve_path(x, root))

    df_train_sub, df_val = stratified_split_train_val(df_train, args.val_ratio, args.seed)

    tr_ids = set(df_train_sub["sample_id"].tolist())
    va_ids = set(df_val["sample_id"].tolist())
    te_ids = set(df_test["sample_id"].tolist())

    assert tr_ids.isdisjoint(va_ids)
    assert tr_ids.isdisjoint(te_ids)
    assert va_ids.isdisjoint(te_ids)

    num_classes = int(max(df_train["label_id"].max(), df_test["label_id"].max()) + 1)
    label_pairs = pd.concat(
        [df_train[["label_id", "label"]], df_test[["label_id", "label"]]],
        ignore_index=True,
    ).drop_duplicates()
    idx_to_label = {int(r.label_id): str(r.label) for r in label_pairs.itertuples(index=False)}

    print("split_dir:", split_dir)
    print("train/test:", len(df_train), len(df_test))
    print("train_sub/val:", len(df_train_sub), len(df_val))
    print("num_classes:", num_classes)

    imu_mean, imu_std = compute_train_imu_norm(df_train_sub, imu_len=args.imu_len)
    pd.DataFrame({"mean": imu_mean, "std": imu_std}).to_csv(out_dir / "imu_norm_stats.csv", index=False)

    ds_train = PairedAlignedDataset(df_train_sub, args.img_size, args.imu_len, imu_mean, imu_std)
    ds_val = PairedAlignedDataset(df_val, args.img_size, args.imu_len, imu_mean, imu_std)
    ds_test = PairedAlignedDataset(df_test, args.img_size, args.imu_len, imu_mean, imu_std)

    dl_train = torch.utils.data.DataLoader(ds_train, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    dl_val = torch.utils.data.DataLoader(ds_val, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)
    dl_test = torch.utils.data.DataLoader(ds_test, batch_size=args.batch, shuffle=False, num_workers=args.num_workers)

    device = "cuda" if args.use_cuda and torch.cuda.is_available() else "cpu"
    print("device:", device)

    img_backbone = ImageEffB0Backbone(str((root / args.image_ckpt).resolve())).to(device)
    imu_backbone = IMUBackboneFromCkpt(str((root / args.imu_ckpt).resolve())).to(device)

    for p in img_backbone.parameters():
        p.requires_grad = False
    for p in imu_backbone.parameters():
        p.requires_grad = False

    X_train_fusion, y_train, sid_train = extract_features(img_backbone, imu_backbone, dl_train, device, desc="train_sub")
    X_val_fusion, y_val, sid_val = extract_features(img_backbone, imu_backbone, dl_val, device, desc="val")
    X_test_fusion, y_test, sid_test = extract_features(img_backbone, imu_backbone, dl_test, device, desc="test")

    img_dim = img_backbone.out_dim
    imu_dim = imu_backbone.out_dim

    X_train_img = X_train_fusion[:, :img_dim]
    X_val_img = X_val_fusion[:, :img_dim]
    X_test_img = X_test_fusion[:, :img_dim]

    X_train_imu = X_train_fusion[:, img_dim: img_dim + imu_dim]
    X_val_imu = X_val_fusion[:, img_dim: img_dim + imu_dim]
    X_test_imu = X_test_fusion[:, img_dim: img_dim + imu_dim]

    rows = []
    rows.append(
        run_mode(
            mode_name="image_only",
            X_train=X_train_img,
            y_train=y_train,
            X_val=X_val_img,
            y_val=y_val,
            X_test=X_test_img,
            y_test=y_test,
            sid_test=sid_test,
            idx_to_label=idx_to_label,
            test_df=df_test,
            args=args,
            device=device,
            out_dir=out_dir,
        )
    )
    rows.append(
        run_mode(
            mode_name="imu_only",
            X_train=X_train_imu,
            y_train=y_train,
            X_val=X_val_imu,
            y_val=y_val,
            X_test=X_test_imu,
            y_test=y_test,
            sid_test=sid_test,
            idx_to_label=idx_to_label,
            test_df=df_test,
            args=args,
            device=device,
            out_dir=out_dir,
        )
    )
    rows.append(
        run_mode(
            mode_name="fusion",
            X_train=X_train_fusion,
            y_train=y_train,
            X_val=X_val_fusion,
            y_val=y_val,
            X_test=X_test_fusion,
            y_test=y_test,
            sid_test=sid_test,
            idx_to_label=idx_to_label,
            test_df=df_test,
            args=args,
            device=device,
            out_dir=out_dir,
        )
    )

    summary_df = pd.DataFrame(rows).sort_values("test_macro_f1", ascending=False).reset_index(drop=True)
    summary_csv = out_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_csv, index=False, encoding="utf-8")

    best = summary_df.iloc[0].to_dict()
    with open(out_dir / "ablation_summary.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "seed": args.seed,
                "device": device,
                "split_dir": str(split_dir),
                "num_train_total": int(len(df_train)),
                "num_test": int(len(df_test)),
                "num_train_sub": int(len(df_train_sub)),
                "num_val": int(len(df_val)),
                "summary_csv": str(summary_csv.resolve()),
                "best_mode": best,
                "rows": rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print("\n=== ABLATION DONE ===")
    print("summary_csv:", summary_csv)
    print(summary_df.to_string(index=False))
    print("best mode:", best["mode"], "test_acc=", round(best["test_acc"], 4), "test_macro_f1=", round(best["test_macro_f1"], 4))


if __name__ == "__main__":
    main()
