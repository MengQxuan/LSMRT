import os
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix

MINC_ROOT = r"F:\study\\LSMRT\data\\minc-2500"
SPLIT_ID = 1
# CKPT_PATH = f"runs/minc_mobilenet_split{SPLIT_ID}/best.pt"
MODEL_TAG = "mobilenet_v3_large"
IMG_SIZE = 192  # 测哪个就填哪个
# CKPT_PATH = f"runs/minc_{MODEL_TAG}_s{IMG_SIZE}_split{SPLIT_ID}/best.pt"
CKPT_PATH = f"runs/effb0_s192_split1_resize/best.pt"

LABEL_DIR = os.path.join(MINC_ROOT, "labels")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH = 64
NUM_WORKERS = 2
PIN_MEMORY = True

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

def parse_class_from_relpath(relpath: str):
    parts = relpath.replace("\\", "/").split("/")
    return parts[1]

class MINCFromTxt(Dataset):
    def __init__(self, root, list_file, class_to_idx, transform):
        self.transform = transform
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        for rel in lines:
            cls = parse_class_from_relpath(rel)
            y = class_to_idx[cls]
            self.samples.append((os.path.join(root, rel), y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y

@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    ys, ps = [], []
    for x, y in tqdm(loader, desc="Testing"):
        x = x.to(DEVICE, non_blocking=True)
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

def main():
    # ✅ 关键：显式 weights_only=False（你信任本地文件即可）
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    num_classes = len(classes)

    model = mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    test_list = os.path.join(LABEL_DIR, f"test{SPLIT_ID}.txt")
    test_ds = MINCFromTxt(MINC_ROOT, test_list, class_to_idx, test_tf)

    loader = DataLoader(
        test_ds, batch_size=BATCH, shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(PIN_MEMORY if DEVICE=="cuda" else False),
        persistent_workers=(NUM_WORKERS > 0),
    )

    acc, f1, cm = evaluate(model, loader)
    print("test_acc:", acc)
    print("test_macro_f1:", f1)
    print("confusion_matrix shape:", cm.shape)

if __name__ == "__main__":
    main()
