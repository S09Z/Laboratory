"""Run SBV2 training as a subprocess with a time budget.

When the budget runs out we do NOT kill immediately: we wait until the
trainer writes its next checkpoint, then send SIGINT, so no progress is lost.
"""
from __future__ import annotations

import re
import signal
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

from .ckpt import complete_steps, prune_checkpoints, prune_safetensors, safetensors_steps

OOM_MARKERS = ("CUDA out of memory", "OutOfMemoryError")


_EPOCH_RE = re.compile(r"Train Epoch: (\d+) \[(\d+)%\]")
_LOSS_RE = re.compile(r"\[((?:-?[\d.]+(?:e-?\d+)?,\s*){3,}-?[\d.]+(?:e-?\d+)?)\]\s*$", re.M)


def parse_progress(text: str) -> dict | None:
    """Latest {step, epoch, pct, lr, losses} from SBV2 log text, or None.

    SBV2 logs ``Train Epoch: N [P%]`` then ``[loss..., global_step, lr]``.
    """
    losses = None
    for m in _LOSS_RE.finditer(text):
        nums = [float(x) for x in m.group(1).split(",")]
        if len(nums) >= 4 and nums[-2] == int(nums[-2]):
            losses = nums
    if losses is None:
        return None
    eps = list(_EPOCH_RE.finditer(text))
    ep = eps[-1] if eps else None
    return {"step": int(losses[-2]), "epoch": int(ep.group(1)) if ep else None,
            "pct": int(ep.group(2)) if ep else None, "lr": losses[-1], "losses": losses[:-2]}


def _read_tail(log_path: Path, offset: int, limit: int = 20000) -> str:
    try:
        size = log_path.stat().st_size
        with open(log_path, encoding="utf-8", errors="ignore") as f:
            f.seek(max(offset, size - limit))
            return f.read()
    except OSError:
        return ""


def _report(log_path: Path, offset: int, elapsed: float) -> None:
    p = parse_progress(_read_tail(log_path, offset))
    t = f"{int(elapsed // 3600)}h{int(elapsed % 3600 // 60):02d}m"
    if p is None:
        print(f"[{t}] ยังไม่มี step ใน train.log (กำลังโหลด/เริ่มเทรน)", flush=True)
        return
    loss = ", ".join(f"{x:.2f}" for x in p["losses"])
    ep = f" · epoch {p['epoch']} {p['pct']}%" if p["epoch"] is not None else ""
    print(f"[{t}] step {p['step']}{ep} · loss [{loss}] · lr {p['lr']:.2e}", flush=True)


def detect_oom(log_text: str) -> bool:
    return any(m in log_text for m in OOM_MARKERS)


def run_training(cmd: list[str], models_dir: Path, budget_s: float, poll_s: float = 30,
                 popen=subprocess.Popen, clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep, log_path: Path | None = None,
                 keep: int = 3, loader=None, assets_dir: Path | None = None,
                 grace_s: float = 60, on_prune: Callable[[], None] | None = None,
                 progress_s: float = 120) -> str:
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
                        grace_s, on_prune, start, last,
                        (log_path, offset, progress_s))
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
           start, last, prog=None) -> str | None:
    stop_after: int | None = None
    next_report = start + prog[2] if prog else None
    while proc.poll() is None:
        sleep(poll_s)
        if prog and clock() >= next_report:
            _report(prog[0], prog[1], clock() - start)
            next_report = clock() + prog[2]
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
