# import os, json, time
# import numpy as np
# from PIL import Image
# from tqdm import tqdm

# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
# from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# # =======================
# # 0) Paths
# # =======================
# MINC_ROOT = r"F:\study\LSMRT\data\minc-2500"
# LABEL_DIR = os.path.join(MINC_ROOT, "labels")

# # =======================
# # 1) Split
# # =======================
# SPLIT_ID = 1

# # =======================
# # 2) Training config (RTX3050 4GB friendly)
# # =======================
# IMG_SIZE = 224
# BATCH = 16          # 若OOM改为8
# EPOCHS = 40
# LR = 3e-4
# WD = 5e-2

# NUM_WORKERS = 2     # Windows若卡住改0
# PIN_MEMORY = True

# OUT_DIR = f"runs/minc_mobilenet_split{SPLIT_ID}"
# os.makedirs(OUT_DIR, exist_ok=True)

# # Force NVIDIA GPU 0
# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # 让 cuDNN 为固定输入尺寸找更快算法（GPU上加速明显）
# if DEVICE == "cuda":
#     torch.backends.cudnn.benchmark = True

# # =======================
# # 3) Transforms
# # =======================
# train_tf = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.RandomHorizontalFlip(0.5),
#     transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
#     transforms.RandomGrayscale(0.1),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# test_tf = transforms.Compose([
#     transforms.Resize((IMG_SIZE, IMG_SIZE)),
#     transforms.ToTensor(),
#     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
# ])

# # =======================
# # 4) Utils: find split files
# # =======================
# def find_split_file(prefix, split_id):
#     cands = [f"{prefix}{split_id}.txt", f"{prefix}{split_id}.TXT"]
#     for c in cands:
#         p = os.path.join(LABEL_DIR, c)
#         if os.path.exists(p):
#             return p
#     return None

# def find_val_file(split_id):
#     for prefix in ["validate", "validation", "val"]:
#         p = find_split_file(prefix, split_id)
#         if p is not None:
#             return p
#     return None

# def parse_class_from_relpath(relpath: str):
#     # e.g. images/brick/brick_002089.jpg
#     parts = relpath.replace("\\", "/").split("/")
#     if len(parts) >= 3:
#         return parts[1]
#     raise ValueError(f"Unexpected path format: {relpath}")

# # =======================
# # 5) Dataset
# # =======================
# class MINCFromTxt(Dataset):
#     def __init__(self, root, list_file, class_to_idx, transform):
#         self.transform = transform
#         self.samples = []  # (abs_path, y)

#         with open(list_file, "r", encoding="utf-8") as f:
#             lines = [ln.strip() for ln in f if ln.strip()]

#         for rel in lines:
#             cls = parse_class_from_relpath(rel)
#             if cls not in class_to_idx:
#                 continue
#             y = class_to_idx[cls]
#             abs_path = os.path.join(root, rel)
#             self.samples.append((abs_path, y))

#         if len(self.samples) == 0:
#             raise RuntimeError(f"No samples loaded from {list_file}. Check paths.")

#     def __len__(self):
#         return len(self.samples)

#     def __getitem__(self, idx):
#         path, y = self.samples[idx]
#         img = Image.open(path).convert("RGB")
#         x = self.transform(img)
#         return x, y

# # =======================
# # 6) Evaluate
# # =======================
# @torch.no_grad()
# def evaluate(model, loader):
#     model.eval()
#     ys, ps = [], []
#     for x, y in loader:
#         x = x.to(DEVICE, non_blocking=True)
#         logits = model(x)
#         pred = logits.argmax(1).cpu().numpy()
#         ys.append(y.numpy())          # 修复 numpy DeprecationWarning
#         ps.append(pred)
#     y = np.concatenate(ys)
#     p = np.concatenate(ps)
#     acc = accuracy_score(y, p)
#     f1 = f1_score(y, p, average="macro")
#     cm = confusion_matrix(y, p)
#     return float(acc), float(f1), cm

# # =======================
# # 7) Main
# # =======================
# def main():
#     train_list = find_split_file("train", SPLIT_ID)
#     val_list   = find_val_file(SPLIT_ID)
#     test_list  = find_split_file("test", SPLIT_ID)

#     print("Train list:", train_list)
#     print("Val list  :", val_list)
#     print("Test list :", test_list)

#     if train_list is None or val_list is None or test_list is None:
#         raise FileNotFoundError("Cannot find train/validate/test txt files in labels/.")

#     # 类别列表：images/ 下的子目录名
#     img_root = os.path.join(MINC_ROOT, "images")
#     classes = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
#     class_to_idx = {c: i for i, c in enumerate(classes)}

#     print("Num classes:", len(classes))
#     print("Classes:", classes)

#     train_ds = MINCFromTxt(MINC_ROOT, train_list, class_to_idx, train_tf)
#     val_ds   = MINCFromTxt(MINC_ROOT, val_list,   class_to_idx, test_tf)
#     test_ds  = MINCFromTxt(MINC_ROOT, test_list,  class_to_idx, test_tf)

#     # Windows：NUM_WORKERS=0 最稳；=2 可能更快
#     # persistent_workers 只有在 workers>0 时可用
#     dl_kwargs = dict(
#         batch_size=BATCH,
#         num_workers=NUM_WORKERS,
#         pin_memory=PIN_MEMORY if DEVICE == "cuda" else False,
#         persistent_workers=(NUM_WORKERS > 0),
#     )
#     # prefetch_factor 也只有 workers>0 才能设
#     if NUM_WORKERS > 0:
#         dl_kwargs["prefetch_factor"] = 2

#     train_loader = DataLoader(train_ds, shuffle=True,  **dl_kwargs)
#     val_loader   = DataLoader(val_ds,   shuffle=False, **dl_kwargs)
#     test_loader  = DataLoader(test_ds,  shuffle=False, **dl_kwargs)

#     # Model
#     weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
#     model = mobilenet_v3_large(weights=weights)
#     model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
#     model.to(DEVICE)

#     criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

#     # ✅ 新 AMP API（PyTorch 2.6+ 推荐）
#     scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda"))

#     best_val = -1.0
#     best_path = os.path.join(OUT_DIR, "best.pt")

#     for epoch in range(1, EPOCHS + 1):
#         model.train()
#         t0 = time.time()
#         losses = []

#         for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
#             x = x.to(DEVICE, non_blocking=True)
#             y = y.to(DEVICE, non_blocking=True)

#             optimizer.zero_grad(set_to_none=True)

#             # ✅ 新 autocast API
#             with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
#                 logits = model(x)
#                 loss = criterion(logits, y)

#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()

#             losses.append(float(loss.item()))

#         scheduler.step()

#         val_acc, val_f1, _ = evaluate(model, val_loader)
#         dt = time.time() - t0
#         print(f"[{epoch}] loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={dt:.1f}s")

#         if val_acc > best_val:
#             best_val = val_acc
#             torch.save({
#                 "model": model.state_dict(),
#                 "classes": classes,
#                 "class_to_idx": class_to_idx,
#                 "img_size": IMG_SIZE,
#                 "split_id": SPLIT_ID,
#                 "torch_version": torch.__version__,
#                 "device": DEVICE
#             }, best_path)

#     # Test best
#     ckpt = torch.load(best_path, map_location=DEVICE)
#     model.load_state_dict(ckpt["model"])
#     test_acc, test_f1, cm = evaluate(model, test_loader)

#     metrics = {
#         "minc_root": MINC_ROOT,
#         "split_id": SPLIT_ID,
#         "model": "mobilenet_v3_large(IMAGENET1K_V2)",
#         "img_size": IMG_SIZE,
#         "batch": BATCH,
#         "epochs": EPOCHS,
#         "lr": LR,
#         "wd": WD,
#         "num_workers": NUM_WORKERS,
#         "pin_memory": (PIN_MEMORY if DEVICE == "cuda" else False),
#         "torch": torch.__version__,
#         "cuda": torch.version.cuda if DEVICE == "cuda" else None,
#         "test_acc": test_acc,
#         "test_macro_f1": test_f1,
#         "classes": classes,
#         "confusion_matrix": cm.tolist(),
#     }

#     with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
#         json.dump(metrics, f, ensure_ascii=False, indent=2)

#     print("=== DONE ===")
#     print("best:", best_path)
#     print("metrics:", os.path.join(OUT_DIR, "metrics.json"))
#     print("test_acc:", test_acc, "macro_f1:", test_f1)


# if __name__ == "__main__":
#     main()


import os, json, time
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

# # 脚本所在目录：.../LSMRT/image-branch
# HERE = os.path.dirname(os.path.abspath(__file__))
# # 项目根目录：.../LSMRT
# PROJECT_ROOT = os.path.abspath(os.path.join(HERE, ".."))

# # 数据目录：.../LSMRT/data/minc-2500
# MINC_ROOT = os.path.join(PROJECT_ROOT, "data", "minc-2500")
# LABEL_DIR = os.path.join(MINC_ROOT, "labels")

# SPLIT_ID = 1

# # 输出目录
# OUT_DIR = os.path.join(HERE, "runs", f"minc_mobilenet_split{SPLIT_ID}")
# os.makedirs(OUT_DIR, exist_ok=True)

MINC_ROOT = r"F:\study\\LSMRT\data\\minc-2500"
LABEL_DIR = os.path.join(MINC_ROOT, "labels")
SPLIT_ID = 1 # 数据划分 ID

IMG_SIZE = 192 # 分辨率，测哪个就填哪个
BATCH = 16
EPOCHS = 40
LR = 3e-4
WD = 5e-2
NUM_WORKERS = 2
PIN_MEMORY = True
MODEL_TAG = "mobilenet_v3_large"

# OUT_DIR = f"runs/minc_mobilenet_split{SPLIT_ID}"
OUT_DIR = f"runs/minc_{MODEL_TAG}_s{IMG_SIZE}_split{SPLIT_ID}"
os.makedirs(OUT_DIR, exist_ok=True)

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cuda":
    torch.backends.cudnn.benchmark = True

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ColorJitter(0.2, 0.2, 0.2, 0.02),
    transforms.RandomGrayscale(0.1),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def find_split_file(prefix, split_id):
    for c in [f"{prefix}{split_id}.txt", f"{prefix}{split_id}.TXT"]:
        p = os.path.join(LABEL_DIR, c)
        if os.path.exists(p):
            return p
    return None

def find_val_file(split_id):
    for prefix in ["validate", "validation", "val"]:
        p = find_split_file(prefix, split_id)
        if p is not None:
            return p
    return None

def parse_class_from_relpath(relpath: str):
    parts = relpath.replace("\\", "/").split("/")
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
            cls = parse_class_from_relpath(rel)
            if cls not in class_to_idx:
                continue
            y = class_to_idx[cls]
            self.samples.append((os.path.join(root, rel), y))
        if len(self.samples) == 0:
            raise RuntimeError(f"No samples loaded from {list_file}. Check paths.")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in loader:
        x = x.to(DEVICE, non_blocking=True)
        logits = model(x)
        pred = logits.argmax(1).cpu().numpy()
        ys.append(y.numpy())
        ps.append(pred)
    y = np.concatenate(ys)
    p = np.concatenate(ps)
    acc = accuracy_score(y, p)
    f1 = f1_score(y, p, average="macro") # 对每个类别算 F1，再平均（对类别不均衡更敏感）。 macro_f1 更能反映“每个类都要尽量好”，不会被样本多的大类“掩盖”。
    cm = confusion_matrix(y, p)
    return float(acc), float(f1), cm

def main():
    train_list = find_split_file("train", SPLIT_ID)
    val_list = find_val_file(SPLIT_ID)
    test_list = find_split_file("test", SPLIT_ID)

    print("Train list:", train_list)
    print("Val list  :", val_list)
    print("Test list :", test_list)

    if train_list is None or val_list is None or test_list is None:
        raise FileNotFoundError("Cannot find train/validate/test txt files in labels/.")

    img_root = os.path.join(MINC_ROOT, "images")
    classes = sorted([d for d in os.listdir(img_root) if os.path.isdir(os.path.join(img_root, d))])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    train_ds = MINCFromTxt(MINC_ROOT, train_list, class_to_idx, train_tf)
    val_ds   = MINCFromTxt(MINC_ROOT, val_list, class_to_idx, test_tf)
    test_ds  = MINCFromTxt(MINC_ROOT, test_list, class_to_idx, test_tf)

    dl_kwargs = dict(
        batch_size=BATCH,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY if DEVICE == "cuda" else False),
        persistent_workers=(NUM_WORKERS > 0),
    )
    if NUM_WORKERS > 0:
        dl_kwargs["prefetch_factor"] = 2

    train_loader = DataLoader(train_ds, shuffle=True, **dl_kwargs)
    val_loader   = DataLoader(val_ds, shuffle=False, **dl_kwargs)
    test_loader  = DataLoader(test_ds, shuffle=False, **dl_kwargs)

    weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
    model = mobilenet_v3_large(weights=weights) # 预训练模型
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) # 损失函数
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD) # 优化器
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS) # 学习率按余弦曲线逐步下降
    scaler = torch.amp.GradScaler("cuda", enabled=(DEVICE == "cuda")) # AMP 混合精度  原理：大部分算子用 FP16/BF16 计算提升速度、减显存；梯度缩放避免 FP16 下梯度下溢。

    best_val = -1.0
    best_path = os.path.join(OUT_DIR, "best.pt")

    for epoch in range(1, EPOCHS + 1):
        """每个 epoch 的步骤
            model.train()
            
            遍历 batch：
                把 x,y 放到 GPU
                autocast 下前向计算得到 logits
                用 cross entropy 算 loss
                scaler.scale(loss).backward() → scaler.step(optimizer) → scaler.update()
            
            scheduler.step()
            调用 evaluate() 跑一遍 val，得到 val_acc/val_f1
            若 val_acc 更好则保存 best.pt
        """
        model.train()
        t0 = time.time()
        losses = []

        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            x = x.to(DEVICE, non_blocking=True)
            y = y.to(DEVICE, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=(DEVICE == "cuda")):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            losses.append(float(loss.item()))

        scheduler.step()
        val_acc, val_f1, _ = evaluate(model, val_loader)
        print(f"[{epoch}] loss={np.mean(losses):.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f} time={time.time()-t0:.1f}s")

        if val_acc > best_val:
            best_val = val_acc
            torch.save({
                "model": model.state_dict(),
                "classes": classes,
                "class_to_idx": class_to_idx,
                "img_size": IMG_SIZE,
                "split_id": SPLIT_ID
            }, best_path)

    # 显式 weights_only=False，加载自己生成的 ckpt 安全
    ckpt = torch.load(best_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model"])
    test_acc, test_f1, cm = evaluate(model, test_loader)

    metrics = {
        "minc_root": MINC_ROOT,
        "split_id": SPLIT_ID,
        "model": "mobilenet_v3_large(IMAGENET1K_V2)",
        "img_size": IMG_SIZE,
        "batch": BATCH,
        "epochs": EPOCHS,
        "lr": LR,
        "wd": WD,
        "num_workers": NUM_WORKERS,
        "torch": torch.__version__,
        "cuda": torch.version.cuda if DEVICE == "cuda" else None,
        "test_acc": test_acc,
        "test_macro_f1": test_f1,
        "classes": classes,
        "confusion_matrix": cm.tolist(),
    }

    with open(os.path.join(OUT_DIR, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print("=== DONE ===")
    print("best:", best_path)
    print("metrics:", os.path.join(OUT_DIR, "metrics.json"))
    print("test_acc:", test_acc, "macro_f1:", test_f1)

if __name__ == "__main__":
    main()
