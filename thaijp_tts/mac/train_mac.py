"""Train Style-Bert-VITS2 (JP-Extra) on Apple Silicon (MPS) — experimental.

SBV2's `train_ms_jp_extra.py` is hard-wired to CUDA + NCCL + DDP. We do NOT edit the cloned repo:
we generate `train_ms_jp_extra_mac.py` next to it by rewriting the source text, and build a
Mac-local dataset root (symlinked wavs, path-rewritten lists/config) so the Colab/Drive data is untouched.

    python mac/train_mac.py prepare --lab ~/LABDIR/Laboratory --name amitaro_runrun
    python mac/train_mac.py run     --lab ~/LABDIR/Laboratory --name amitaro_runrun [--smoke]
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

COLAB_PREFIX = "/content/drive/MyDrive/Laboratory"
PRETRAINED = ("G_0.safetensors", "D_0.safetensors", "WD_0.safetensors", "DUR_0.safetensors")

_PRELUDE = '''
# --- thaijp_tts mac patch: MPS/CPU, single process, no DDP/NCCL/AMP ---
from contextlib import nullcontext as _nullcontext
DEV = torch.device("mps" if torch.backends.mps.is_available()
                   and os.environ.get("THAIJP_TRAIN_DEVICE", "mps") == "mps" else "cpu")

def autocast(*_a, **_k):
    return _nullcontext()

class _NoDDP(torch.nn.Module):
    """Same `.module.` state_dict prefix as DDP so SBV2's checkpoint code keeps working."""
    def __init__(self, module):
        super().__init__()
        self.module = module
    def forward(self, *a, **k):
        return self.module(*a, **k)

def _empty_cache():
    if DEV.type == "mps":
        torch.mps.empty_cache()
# --- end patch ---
'''


def patch_source(src: str) -> str:
    """Rewrite SBV2 train_ms_jp_extra.py source for MPS/CPU. Raises if an expected anchor is missing."""
    out = src
    # 1. CUDA-only backend flags (tf32/cudnn/sdp) — remove the whole statements, parentheses included
    out = re.sub(r"^torch\.backends\.cudnn\.allow_tf32 = \(.*?\n\)\n", "", out, flags=re.M | re.S)
    out = re.sub(r"^torch\.backends\.cuda\.enable_mem_efficient_sdp\(.*?\n\)[^\n]*\n", "", out, flags=re.M | re.S)
    out = re.sub(r"^torch\.backends\.cuda\.[^\n]*\n", "", out, flags=re.M)
    # 2. prelude after the last top-level import block marker (before `config = get_config()`)
    anchor = "config = get_config()"
    if anchor not in out:
        raise ValueError("anchor 'config = get_config()' not found")
    out = out.replace(anchor, _PRELUDE + "\n" + anchor, 1)
    # 3. distributed init → single process
    pat = re.compile(r'    backend = "nccl".*?n_gpus = dist\.get_world_size\(\)\n', re.S)
    if not pat.search(out):
        raise ValueError("distributed init block not found")
    out = pat.sub("    rank = 0\n    local_rank = 0\n    n_gpus = 1\n", out, count=1)
    out = out.replace("torch.cuda.set_device(local_rank)", "pass  # (mac) no cuda device")
    # 4. tensors/modules → DEV
    out = re.sub(r"\.cuda\(\s*(?:local_rank\s*(?:,\s*non_blocking=True\s*)?)?\)", ".to(DEV)", out)
    out = out.replace(").to(local_rank)", ").to(DEV)")
    # 5. DDP → passthrough
    out = re.sub(r"\bDDP\(\s*(\w+),\s*device_ids=\[local_rank\],.*?\n\s*\)", r"_NoDDP(\1)", out, flags=re.S)
    # 6. misc
    out = out.replace("torch.cuda.empty_cache()", "_empty_cache()")
    out = out.replace("pin_memory=True", "pin_memory=False")
    if ".cuda(" in out or "DDP(" in out.replace("_NoDDP(", "").replace("import DistributedDataParallel as DDP", ""):
        raise ValueError("unpatched CUDA/DDP usage remains")
    return out


def rewrite_paths(text: str, new_prefix: str, old_prefix: str = COLAB_PREFIX) -> str:
    return text.replace(old_prefix, new_prefix.rstrip("/"))


def prepare(lab: Path, name: str, sbv2: Path, mac_root: Path) -> Path:
    """Build mac_root/<name> from lab/sbv2/Data/<name>; point SBV2 paths.yml at it. Returns dataset dir."""
    src = lab / "sbv2" / "Data" / name
    assert src.is_dir(), f"ไม่พบข้อมูลเทรน {src}"
    dst = mac_root / name
    (dst / "models").mkdir(parents=True, exist_ok=True)
    wavs = dst / "wavs"
    if not wavs.exists():
        wavs.symlink_to(src / "wavs", target_is_directory=True)
    prefix = str(lab / "sbv2" / "Data" / name)
    for f in ("train.list", "val.list", "esd.list", "esd.list.cleaned"):
        if (src / f).exists():
            (dst / f).write_text(rewrite_paths((src / f).read_text(encoding="utf-8"), str(lab)), encoding="utf-8")
    cfg = json.loads(rewrite_paths((src / "config.json").read_text(encoding="utf-8"), str(lab)))
    cfg["data"]["training_files"] = str(dst / "train.list")
    cfg["data"]["validation_files"] = str(dst / "val.list")
    cfg["train"]["bf16_run"] = False
    cfg["train"]["fp16_run"] = False
    (dst / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
    for f in PRETRAINED:
        s = src / "models" / f
        if s.exists() and not (dst / "models" / f).exists():
            shutil.copy2(s, dst / "models" / f)
    assets = lab / "sbv2" / "model_assets"
    assets.mkdir(exist_ok=True)
    (sbv2 / "configs" / "paths.yml").write_text(
        f"dataset_root: {mac_root}\nassets_root: {assets}\n", encoding="utf-8")
    # wavs in the list point into lab/sbv2/Data/<name>/wavs — same files as the symlink target
    assert prefix  # (kept for clarity)
    return dst


def build_trainer(sbv2: Path) -> Path:
    out = sbv2 / "train_ms_jp_extra_mac.py"
    out.write_text(patch_source((sbv2 / "train_ms_jp_extra.py").read_text(encoding="utf-8")), encoding="utf-8")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "run"])
    ap.add_argument("--lab", type=Path, default=Path(__file__).resolve().parents[2])
    ap.add_argument("--name", default="amitaro_runrun")
    ap.add_argument("--home", type=Path, default=None, help="THAIJP_HOME (default <lab>/thaijp_lab)")
    ap.add_argument("--smoke", action="store_true", help="log every step, stop after --max-seconds")
    ap.add_argument("--max-seconds", type=int, default=240)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--segment-size", type=int, default=None)
    ap.add_argument("--epochs", type=int, default=None)
    a = ap.parse_args(argv)
    a.lab = a.lab.expanduser().resolve()
    home = a.home or Path(os.environ.get("THAIJP_HOME", a.lab / "thaijp_lab"))
    sbv2, mac_root = home / "Style-Bert-VITS2", home / "mac_data"
    dst = prepare(a.lab, a.name, sbv2, mac_root)
    trainer = build_trainer(sbv2)
    print("dataset:", dst, "\ntrainer:", trainer)
    if a.cmd == "prepare":
        return 0
    cfg = dst / "config.json"
    over = {"batch_size": a.batch_size, "segment_size": a.segment_size, "epochs": a.epochs}
    over = {k: v for k, v in over.items() if v is not None}
    if a.smoke or over:
        c = json.loads(cfg.read_text(encoding="utf-8"))
        c["train"].update(over)
        if a.smoke:
            c["train"]["log_interval"] = 1
        cfg = dst / ("config.smoke.json" if a.smoke else "config.mac.json")
        cfg.write_text(json.dumps(c, indent=2), encoding="utf-8")
    env = {**os.environ, "PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
    cmd = [sys.executable, str(trainer), "-c", str(cfg), "-m", str(dst), "--skip_default_style"]
    try:
        return subprocess.run(cmd, cwd=sbv2, env=env, timeout=a.max_seconds if a.smoke else None).returncode
    except subprocess.TimeoutExpired:
        print(f"\n(smoke) stopped after {a.max_seconds}s")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
