import time
import argparse
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import (
    mobilenet_v3_large,
    MobileNet_V3_Large_Weights,
    efficientnet_b0,
    EfficientNet_B0_Weights,
)
from PIL import Image
import os

def parse_args():
    p = argparse.ArgumentParser("Model inference speed benchmark")
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--img_size", type=int, default=192)
    p.add_argument("--num_images", type=int, default=512)
    return p.parse_args()

def build_model(arch: str, num_classes: int):
    if arch == "mnv3_large":
        model = mobilenet_v3_large(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model
    elif arch == "effb0":
        model = efficientnet_b0(weights=None)
        in_f = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_f, num_classes)
        return model
    else:
        raise ValueError(f"Unknown arch: {arch}")

@torch.no_grad()
def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    arch = ckpt.get("arch", "mnv3_large")
    classes = ckpt["classes"]
    num_classes = len(classes)

    model = build_model(arch, num_classes)
    model.load_state_dict(ckpt["model"])
    model.to(device)
    model.eval()

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225)),
    ])

    # fake image
    img = Image.new("RGB", (args.img_size, args.img_size), color=(128,128,128))
    x = tf(img).unsqueeze(0).to(device)

    # warm-up
    for _ in range(20):
        _ = model(x)

    torch.cuda.synchronize() if device == "cuda" else None

    t0 = time.time()
    for _ in range(args.num_images):
        _ = model(x)
    torch.cuda.synchronize() if device == "cuda" else None
    t1 = time.time()

    total = t1 - t0
    ms_per_img = total / args.num_images * 1000

    print("=== Benchmark Result ===")
    print("ckpt:", args.ckpt)
    print("arch:", arch)
    print("img_size:", args.img_size)
    print("images:", args.num_images)
    print("total_sec:", total)
    print("ms_per_img:", ms_per_img)

if __name__ == "__main__":
    main()
