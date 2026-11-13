"""Colab runtime helpers: GPU check, batch size, Drive paths, run manifest."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path


def ensure_gpu() -> dict:
    """Return {'name', 'vram_gb'}; refuse to continue on a CPU-only runtime."""
    if shutil.which("nvidia-smi") is None:
        raise RuntimeError("No GPU: Runtime > Change runtime type > T4 GPU, then run again")
    out = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout.strip().splitlines()[0]
    name, mem = [x.strip() for x in out.split(",")]
    return {"name": name, "vram_gb": round(float(mem) / 1024, 1)}


def pick_batch_size(vram_gb: float) -> int:
    for need, bs in ((22, 8), (15, 4), (11, 2)):
        if vram_gb >= need:
            return bs
    return 1


def run_paths(drive_root: Path, run_id: str) -> dict[str, Path]:
    root = Path(drive_root) / run_id
    paths = {"root": root, "data": root / "data", "models": root / "models",
             "logs": root / "logs", "manifest": root / "manifest.json"}
    for k in ("data", "models", "logs"):
        paths[k].mkdir(parents=True, exist_ok=True)
    return paths


def data_hash(directory: Path) -> str:
    h = hashlib.sha256()
    d = Path(directory)
    if d.exists():
        for f in sorted(p for p in d.rglob("*") if p.is_file()):
            h.update(f"{f.relative_to(d)}:{f.stat().st_size}\n".encode())
    return h.hexdigest()[:16]


def write_manifest(path: Path, **fields) -> None:
    fields.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S%z"))
    Path(path).write_text(json.dumps(fields, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def write_sbv2_config_yml(sbv2_dir: Path, model_name: str, dataset_path: Path, keep_ckpts: int = 3) -> Path:
    """SBV2's trainer reads model_name / dataset_path / keep_ckpts from config.yml (not config.json).

    Must be rewritten before every training launch: it decides where checkpoints and
    model_assets/<model_name> land, and a fresh VM starts from the defaults.
    """
    import yaml

    sbv2_dir = Path(sbv2_dir)
    cfg_path = sbv2_dir / "config.yml"
    base = cfg_path if cfg_path.exists() else sbv2_dir / "default_config.yml"
    cfg = yaml.safe_load(base.read_text(encoding="utf-8")) or {}
    cfg["model_name"] = model_name
    cfg["dataset_path"] = str(dataset_path)
    cfg.setdefault("train_ms", {})["keep_ckpts"] = keep_ckpts
    cfg_path.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return cfg_path


def copy_pretrained_init(src_models: Path, dst_models: Path) -> int:
    """Copy the pretrained JP-Extra init weights (*_0.safetensors) so a copied dataset doesn't train from scratch."""
    dst_models = Path(dst_models)
    dst_models.mkdir(parents=True, exist_ok=True)
    n = 0
    for f in sorted(Path(src_models).glob("*_0.safetensors")):
        if not (dst_models / f.name).exists():
            shutil.copy2(f, dst_models / f.name)
        n += 1
    return n


INIT_FILES = ("G_0.safetensors", "D_0.safetensors", "WD_0.safetensors")


def ensure_pretrained_init(pretrained_dir: Path, models_dir: Path) -> int:
    """Make sure models/ has the JP-Extra init weights, copying from SBV2's pretrained_jp_extra/.

    SBV2's checkpoint cleanup deletes G_0/D_0/WD_0 once 3 checkpoints exist, and
    without them the trainer silently starts from random weights.
    """
    pretrained_dir, models_dir = Path(pretrained_dir), Path(models_dir)
    missing = [f for f in INIT_FILES if not (pretrained_dir / f).exists()]
    if missing:
        raise RuntimeError(f"pretrained JP-Extra weights missing in {pretrained_dir}: {missing} "
                           "(run SBV2 initialize.py)")
    models_dir.mkdir(parents=True, exist_ok=True)
    for f in INIT_FILES:
        if not (models_dir / f).exists():
            shutil.copy2(pretrained_dir / f, models_dir / f)
    return len(INIT_FILES)
