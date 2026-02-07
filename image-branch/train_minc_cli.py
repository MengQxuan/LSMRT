import os, json, time, argparse
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

from torchvision.models import (
    mobilenet_v3_large, mobilenet_v3_small,
    MobileNet_V3_Large_Weights, MobileNet_V3_Small_Weights,
    efficientnet_b0, EfficientNet_B0_Weights
)

def parse_args():
    p = argparse.ArgumentParser("MINC-2500 image branch trainer")

    p.add_argument("--minc_root", type=str, default="/root/mqx/LSMRT/data/minc-2500")

    p.add_argument("--split", type=int, default=1)
    p.add_argument("--arch", type=str, default="mnv3_large", choices=["mnv3_large", "mnv3_small", "effb0"])
    p.add_argument("--img_size", type=int, default=192)
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=5e-2)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--strong_aug", action="store_true", help="use RandomResizedCrop-based stronger augmentation")
    p.add_argument("--rrc_scale_low", type=float, default=0.8, help="lower bound of RandomResizedCrop scale when --strong_aug")
    return p.parse_args()

def seed_everything(seed: int):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def find_split_file(label_dir, prefix, split_id):
    for c in [f"{prefix}{split_id}.txt", f"{prefix}{split_id}.TXT"]:
        p = os.path.join(label_dir, c)
        if os.path.exists(p):
            return p
    return None

def find_val_file(label_dir, split_id):
    for prefix in ["validate", "validation", "val"]:
        p = find_split_file(label_dir, prefix, split_id)
        if p is not None:
            return p
    return None

def parse_class_from_relpath(relpath: str):
    parts = relpath.replace("\\", "/").split("/")
    # 兼容 "images/brick/xxx.jpg"
    if len(parts) >= 3:
        return parts[1]
    raise ValueError(f"Unexpected path format: {relpath}")

class MINCFromTxt(Dataset):
    def __init__(self, root, list_file, class_to_idx, transform):
        self.transform = transform
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]

        for rel in lines:
            # rel 可能是：
            #   images/brick/brick_000168.jpg
            # 或者（极少数情况）没有 "images/" 前缀：brick/brick_000168.jpg
            rel_norm = rel.replace("\\", "/")
            if not rel_norm.startswith("images/") and not os.path.isabs(rel_norm):
                # 若没有 images/ 前缀，给它补上（不改变你原txt也能跑）
                rel_norm = "images/" + rel_norm.lstrip("/")

            cls = parse_class_from_relpath(rel_norm)
            if cls not in class_to_idx:
                continue
            y = class_to_idx[cls]
            self.samples.append((os.path.join(root, rel_norm), y))

        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {list_file}. Check paths.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro")
    cm = confusion_matrix(y, p)
    return float(acc), float(f1), cm

def build_model(arch: str, num_classes: int):
    if arch == "mnv3_large":
        weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
        model = mobilenet_v3_large(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model, "mobilenet_v3_large(IMAGENET1K_V2)"
    if arch == "mnv3_small":
        weights = MobileNet_V3_Small_Weights.IMAGENET1K_V1
        model = mobilenet_v3_small(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model, "mobilenet_v3_small(IMAGENET1K_V1)"
    if arch == "effb0":
        weights = EfficientNet_B0_Weights.IMAGENET1K_V1
        model = efficientnet_b0(weights=weights)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model, "efficientnet_b0(IMAGENET1K_V1)"
    raise ValueError(arch)

def main():
    args = parse_args()
    seed_everything(args.seed)

    minc_root = args.minc_root
    label_dir = os.path.join(minc_root, "labels")
    img_root = os.path.join(minc_root, "images")

    # ✅ 不要在代码里硬编码 CUDA_VISIBLE_DEVICES
    #    你要用哪张卡/多卡，就在命令行里 export CUDA_VISIBLE_DEVICES=...
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.backends.cudnn.benchmark = True

    train_list = find_split_file(label_dir, "train", args.split)
    val_list   = find_val_file(label_dir, args.split)
    test_list  = find_split_file(label_dir, "test", args.split)

    print("Train list:", train_list)
    print("Val list  :", val_list)
    print("Test list :", test_list)
    if train_list is None or val_list is None or test_list is None:
        raise FileNotFoundError("Cannot find train/validate/test txt files under labels/. Check split id and filenames.")

    # classes 从 images/ 子目录扫描（与原逻辑一致）
    classes = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}
    num_classes = len(classes)
    print("Num classes:", num_classes)

    # transforms
    if args.strong_aug:
        train_tf = transforms.Compose([
            transforms.RandomResizedCrop(args.img_size, scale=(args.rrc_scale_low, 1.0)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((args.img_size, args.img_size)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
            transforms.RandomGrayscale(0.1),
            transforms.ToTensor(),
            transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
        ])

    eval_tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    train_ds = MINCFromTxt(minc_root, train_list, class_to_idx, train_tf)
    val_ds   = MINCFromTxt(minc_root, val_list,   class_to_idx, eval_tf)
    test_ds  = MINCFromTxt(minc_root, test_list,  class_to_idx, eval_tf)

    pin_memory = (device == "cuda")
    dl_kwargs = dict(
        batch_size=args.batch,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        persistent_workers=(args.num_workers > 0),
    )
    if args.num_workers > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
    val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

    model, model_name = build_model(args.arch, num_classes)
    model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    scaler = torch.amp.GradScaler("cuda", enabled=(device == "cuda"))

    tag_aug = f"rrc{args.rrc_scale_low}" if args.strong_aug else "resize"
    out_dir = os.path.join(
        os.path.dirname(__file__),
        "runs",
        f"{args.arch}_s{args.img_size}_split{args.split}_{tag_aug}"
    )
    os.makedirs(out_dir, exist_ok=True)
    best_path = os.path.join(out_dir, "best.pt")

    best_val = -1.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(device == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        scheduler.step()

        val_acc, val_f1, _ = evaluate(model, val_loader, device)
        print(f"[{epoch}] loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "img_size": args.img_size,
                "split_id": args.split,
                "arch": args.arch,
                "aug": tag_aug,
            }, best_path)

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc, test_f1, cm = evaluate(model, test_loader, device)

    metrics = {
        "minc_root": minc_root,
        "split_id": args.split,
        "arch": args.arch,
        "model_name": model_name,
        "img_size": args.img_size,
        "aug": tag_aug,
        "batch": args.batch,
        "epochs": args.epochs,
        "lr": args.lr,
        "wd": args.wd,
        "num_workers": args.num_workers,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if device == "cuda" else None,
        "val_best_acc": best_val,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
        "best_path": best_path,
    }

    with open(os.path.join(out_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== DONE ===")
    print("out_dir:", out_dir)
    print("best:", best_path)
    print("test_acc:", test_acc, "macro_f1:", test_f1)

if __name__ == "__main__":
    main()
