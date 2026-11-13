"""Device and path helpers for running inference on a Mac (Apple Silicon)."""
from __future__ import annotations

from pathlib import Path
from typing import Callable


def pick_device(available: dict[str, bool] | None = None, probe: Callable[[str], None] | None = None) -> str:
    """'mps' if Apple GPU is available and a probe run works there, else 'cpu'.

    ``probe(device)`` should run a tiny real inference; any exception (e.g. an op
    not implemented for MPS) means we fall back to CPU instead of failing later.
    """
    if available is None:
        import torch
        available = {"mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    if available.get("mps"):
        try:
            if probe is not None:
                probe("mps")
            return "mps"
        except Exception as e:  # noqa: BLE001
            print(f"MPS ใช้ไม่ได้กับโมเดลนี้ ({type(e).__name__}: {e}) → ใช้ CPU")
    return "cpu"


def find_drive_lab(home: Path | None = None, folder: str = "Laboratory") -> Path | None:
    """Locate MyDrive/Laboratory synced by Google Drive for desktop (any account, any UI language)."""
    base = Path(home or Path.home()) / "Library" / "CloudStorage"
    if not base.exists():
        return None
    for acct in sorted(base.glob("GoogleDrive-*")):
        for top in sorted(acct.iterdir()):
            cand = top / folder
            if cand.is_dir():
                return cand
    return None
