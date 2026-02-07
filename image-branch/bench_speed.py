# 加载 best.pt，随机取 N 张 test 图，统计平均推理耗时 ms/张
import os, time, random
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models import mobilenet_v3_large

MINC_ROOT = r"F:\study\\LSMRT\data\\minc-2500"
LABEL_DIR = os.path.join(MINC_ROOT, "labels")

SPLIT_ID = 1
CKPT_PATH = r"runs/effb0_s192_split1_resize/best.pt"  # 改成你要测的ckpt
NUM_SAMPLES = 512          # 取多少张做测速
BATCH = 32                 # 测速batch
NUM_WORKERS = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_class_from_relpath(relpath: str):
    parts = relpath.replace("\\", "/").split("/")
    return parts[1]

class MINCPaths(Dataset):
    def __init__(self, root, list_file, class_to_idx, transform, max_n=None):
        self.transform = transform
        self.samples = []
        with open(list_file, "r", encoding="utf-8") as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        random.shuffle(lines)
        if max_n is not None:
            lines = lines[:max_n]
        for rel in lines:
            cls = parse_class_from_relpath(rel)
            y = class_to_idx[cls]
            self.samples.append((os.path.join(root, rel), y))

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        p, y = self.samples[idx]
        img = Image.open(p).convert("RGB")
        return self.transform(img), y

def main():
    ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)
    classes = ckpt["classes"]
    class_to_idx = ckpt["class_to_idx"]
    img_size = ckpt.get("img_size", 224)
    num_classes = len(classes)

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    test_list = os.path.join(LABEL_DIR, f"test{SPLIT_ID}.txt")
    ds = MINCPaths(MINC_ROOT, test_list, class_to_idx, tf, max_n=NUM_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
                        num_workers=NUM_WORKERS,
                        pin_memory=(DEVICE=="cuda"),
                        persistent_workers=(NUM_WORKERS>0))

    model = mobilenet_v3_large(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(DEVICE).eval()

    # warmup
    with torch.no_grad():
        for x, _ in loader:
            x = x.to(DEVICE, non_blocking=True)
            _ = model(x)
            break
    if DEVICE == "cuda":
        torch.cuda.synchronize()

    # benchmark
    n_images = 0
    t0 = time.time()
    with torch.no_grad():
        for x, _ in tqdm(loader, desc="Benchmark"):
            x = x.to(DEVICE, non_blocking=True)
            _ = model(x)
            n_images += x.size(0)
    if DEVICE == "cuda":
        torch.cuda.synchronize()
    t = time.time() - t0

    ms_per_img = t / n_images * 1000
    print("device:", DEVICE)
    print("img_size:", img_size)
    print("images:", n_images)
    print("total_sec:", t)
    print("ms_per_img:", ms_per_img)

if __name__ == "__main__":
    main()
