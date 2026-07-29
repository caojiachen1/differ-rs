#!/usr/bin/env python3
"""Convert PyTorch DINOv3 .pth weights to safetensors for Candle backend."""

import sys
from pathlib import Path

import torch
from safetensors.torch import save_file

def main():
    input_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("models/dinov3_vits16_pretrain.pth")
    output_path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("models/dinov3_vits16.safetensors")

    print(f"Loading {input_path} ...")
    state_dict = torch.load(input_path, map_location="cpu", weights_only=True)

    if "model" in state_dict:
        state_dict = state_dict["model"]
    elif "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    # Strip "module." prefix if present
    cleaned = {}
    for k, v in state_dict.items():
        key = k.removeprefix("module.")
        cleaned[key] = v.contiguous()

    print(f"Saving {len(cleaned)} tensors to {output_path} ...")
    save_file(cleaned, str(output_path))
    print(f"Done! ({output_path.stat().st_size / 1024 / 1024:.1f} MB)")

if __name__ == "__main__":
    main()
