"""torch<2.6 (SBV2 pin) + transformers>=4.50 refuse to torch.load *.bin → convert JP BERT to model.safetensors once.

Usage: python mac/bin_to_safetensors.py <bert_dir>  (idempotent)
"""
import sys
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert(bert_dir: Path) -> bool:
    src, dst = bert_dir / "pytorch_model.bin", bert_dir / "model.safetensors"
    if dst.exists() or not src.exists():
        return False
    sd = torch.load(src, map_location="cpu", weights_only=True)
    save_file({k: v.contiguous().clone() for k, v in sd.items()}, dst, metadata={"format": "pt"})
    return True


if __name__ == "__main__":
    d = Path(sys.argv[1])
    print("converted" if convert(d) else "skip (already safetensors or no .bin)", d)
