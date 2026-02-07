import torch
from safetensors.torch import load_file

in_path  = "/root/mqx/LSMRT/model.safetensors"
out_path = "/root/mqx/LSMRT/model.pth"

sd = load_file(in_path, device="cpu")
torch.save(sd, out_path)
print("saved:", out_path, "keys:", len(sd))
