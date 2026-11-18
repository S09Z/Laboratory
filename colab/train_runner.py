"""Run SBV2 training as a subprocess with a time budget.

When the budget runs out we do NOT kill immediately: we wait until the
trainer writes its next checkpoint, then send SIGINT, so no progress is lost.
"""
from __future__ import annotations

import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .ckpt import complete_steps, prune_checkpoints, prune_safetensors, safetensors_steps

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


def detect_oom(log_text: str) -> bool:
    return any(m in log_text for m in OOM_MARKERS)


def run_training(cmd: list[str], models_dir: Path, budget_s: float, poll_s: float = 30,
                 popen=subprocess.Popen, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, log_path: Path | None = None,
                 keep: int = 3, loader=None, assets_dir: Path | None = None,
                 grace_s: float = 60, on_prune: Callable[[], None] | None = None) -> str:
    """Return "finished" | "budget_stop" | "oom" | "failed"."""
    models_dir = Path(models_dir)
    log_path = Path(log_path or models_dir / "train.log")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    offset = log_path.stat().st_size if log_path.exists() else 0
    with open(log_path, "a", encoding="utf-8") as logf:
        proc = popen(cmd, stdout=logf, stderr=subprocess.STDOUT)
    start = clock()
    last = (complete_steps(models_dir) or [0])[-1]
    try:
        status = _watch(proc, models_dir, assets_dir, budget_s, poll_s, clock, sleep, keep,
                        grace_s, on_prune, start, last)
    except BaseException:
        # Stop button / error in the notebook: never leave the trainer running in the background
        if proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=120)
            except subprocess.TimeoutExpired:
                proc.kill()
        raise
    if status:
        return status
    text = ""
    if log_path.exists():
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            f.seek(offset)
            text = f.read()
    if proc.returncode == 0:
        return "finished"
    return "oom" if detect_oom(text) else "failed"


def _watch(proc, models_dir, assets_dir, budget_s, poll_s, clock, sleep, keep, grace_s, on_prune,
           start, last) -> str | None:
    stop_after: int | None = None
    while proc.poll() is None:
        sleep(poll_s)
        steps = complete_steps(models_dir)
        newest = steps[-1] if steps else 0
        if newest > last:
            last = newest
            prune_checkpoints(models_dir, keep)
            if assets_dir is not None:
                prune_safetensors(assets_dir, keep)
            if on_prune is not None:
                try:
                    on_prune()
                except Exception as e:  # noqa: BLE001 — cleanup must never stop training
                    print("on_prune failed:", e)
        if stop_after is None and clock() - start >= budget_s:
            stop_after = newest
        if stop_after is not None and newest > stop_after and _inference_saved(assets_dir, newest):
            sleep(grace_s)  # let the last writes finish before interrupting
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=600)
            except subprocess.TimeoutExpired:
                proc.kill()
            return "budget_stop"
    return None


def _inference_saved(assets_dir: Path | None, step: int) -> bool:
    if assets_dir is None:
        return True
    return any(s >= step for s, _ in safetensors_steps(assets_dir))
