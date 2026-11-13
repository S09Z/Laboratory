"""Checkpoint safety for Style-Bert-VITS2 runs that may be killed mid-write."""
from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Callable

PARTS = ("G", "D", "WD")
_RX = re.compile(r"^(G|D|WD)_(\d+)\.pth$")
_ANY = re.compile(r"^(G|D|WD|DUR)_(\d+)\.pth$")
_ST = re.compile(r"_s(\d+)\.safetensors$")


def _steps(models_dir: Path) -> dict[int, set[str]]:
    found: dict[int, set[str]] = {}
    d = Path(models_dir)
    if not d.exists():
        return found
    for f in d.iterdir():
        m = _RX.match(f.name)
        if m:
            found.setdefault(int(m.group(2)), set()).add(m.group(1))
    return found


def complete_steps(models_dir: Path) -> list[int]:
    return sorted(s for s, parts in _steps(models_dir).items() if set(PARTS) <= parts)


def _default_loader(path: Path):
    import torch
    torch.load(path, map_location="cpu", weights_only=False)


def _move_newer_aside(d: Path, keep_step: int | None) -> None:
    """SBV2 resumes from the highest-numbered G_*.pth, so anything newer than the
    chosen step (half-written or unloadable) must leave models/."""
    for f in list(d.iterdir()) if d.exists() else []:
        m = _ANY.match(f.name)
        if m and (keep_step is None or int(m.group(2)) > keep_step):
            (d / "corrupt").mkdir(exist_ok=True)
            shutil.move(str(f), d / "corrupt" / f.name)


def find_resumable_checkpoint(models_dir: Path, loader: Callable | None = None) -> int | None:
    """Newest step whose G/D/WD all load. Every file of a newer step is moved to corrupt/."""
    loader = loader or _default_loader
    d = Path(models_dir)
    chosen = None
    for step in reversed(complete_steps(d)):
        try:
            for p in PARTS:
                loader(d / f"{p}_{step}.pth")
            chosen = step
            break
        except Exception:  # noqa: BLE001
            continue
    _move_newer_aside(d, chosen)
    return chosen


def prune_checkpoints(models_dir: Path, keep: int = 3) -> list[Path]:
    d = Path(models_dir)
    removed = []
    for step in complete_steps(d)[:-keep] if keep > 0 else []:
        for p in PARTS:
            f = d / f"{p}_{step}.pth"
            if f.exists():
                f.unlink()
                removed.append(f)
    return removed


def safetensors_steps(assets_dir: Path) -> list[tuple[int, Path]]:
    d = Path(assets_dir)
    out = []
    for f in d.glob("*.safetensors") if d.exists() else []:
        m = _ST.search(f.name)
        if m:
            out.append((int(m.group(1)), f))
    return sorted(out)


def prune_safetensors(assets_dir: Path, keep: int = 3) -> list[Path]:
    """SBV2 never deletes old inference files in model_assets; keep Drive from filling up."""
    removed = []
    for _, f in safetensors_steps(assets_dir)[:-keep] if keep > 0 else []:
        f.unlink()
        removed.append(f)
    return removed
