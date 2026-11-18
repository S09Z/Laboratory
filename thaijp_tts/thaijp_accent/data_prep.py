"""Dataset preparation: QC, ITA-corpus transcript matching, esd.list, BERT neutralization (E2)."""
from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path
from typing import Literal

import numpy as np

MIN_S, MAX_S = 1.0, 15.0
CLIP_LEVEL, CLIP_FRAC = 0.999, 0.001
SILENT_RMS = 1e-3


def qc_clip(wav_path: Path) -> list[str]:
    import soundfile as sf

    x, sr = sf.read(str(wav_path), always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    problems = []
    dur = len(x) / sr
    if dur < MIN_S:
        problems.append("too_short")
    if dur > MAX_S:
        problems.append("too_long")
    if len(x) and np.mean(np.abs(x) >= CLIP_LEVEL) > CLIP_FRAC:
        problems.append("clipping")
    if len(x) == 0 or np.sqrt(np.mean(x ** 2)) < SILENT_RMS:
        problems.append("silent")
    return problems


def parse_ita_transcript(text: str) -> dict[tuple[str, int], str]:
    """'EMOTION100_001:漢字かな。,ヨミ。' → {('EMOTION', 1): '漢字かな。'}"""
    out = {}
    for line in text.splitlines():
        m = re.match(r"^(EMOTION|RECITATION)\d+_(\d+):([^,]+)", line.strip())
        if m:
            out[(m.group(1), int(m.group(2)))] = m.group(3)
    return out


def _kind(name: str) -> str | None:
    n = name.lower()
    if "recitation" in n or n.startswith("rec"):
        return "RECITATION"
    if "emo" in n:
        return "EMOTION"
    return None


def match_ita_files(files: list[Path], transcripts: dict[tuple[str, int], str]
                    ) -> tuple[list[tuple[str, str]], list[str]]:
    """Pair wav files with ITA sentences by (corpus kind, last number in the name)."""
    entries, unmatched = [], []
    for f in sorted(files, key=lambda p: Path(p).name):
        name = Path(f).name
        nums = re.findall(r"\d+", Path(name).stem)
        key = (_kind(name), int(nums[-1])) if nums and _kind(name) else None
        if key in transcripts:
            entries.append((name, transcripts[key]))
        else:
            unmatched.append(name)
    return entries, unmatched


def write_esd_list(entries: list[tuple[str, str]], speaker: str, out_path: Path) -> None:
    """Style-Bert-VITS2 esd.list: <wav file>|<speaker>|JP|<text>."""
    lines = [f"{wav}|{speaker}|JP|{text}\n" for wav, text in entries]
    Path(out_path).write_text("".join(lines), encoding="utf-8")


def neutralize_bert(dataset_dir: Path, mode: Literal["zeros", "mean"],
                    load: Callable | None = None, save: Callable | None = None,
                    zeros_like: Callable | None = None) -> int:
    """Overwrite every *.bert.pt with zeros (or the dataset mean vector), keeping shapes.

    Writes a marker so running the cell twice does nothing the second time.
    """
    d = Path(dataset_dir)
    marker = d / f".bert_neutralized_{mode}"
    if marker.exists():
        return 0
    if load is None or save is None or zeros_like is None:
        import torch
        load = load or (lambda p: torch.load(p, map_location="cpu"))
        save = save or (lambda o, p: torch.save(o, p))
        zeros_like = zeros_like or torch.zeros_like
    files = sorted(d.rglob("*.bert.pt"))
    mean = None
    if mode == "mean":
        tensors = [load(f) for f in files]
        total = sum(t.sum(axis=-1) if hasattr(t, "axis") else t.sum(-1) for t in tensors)
        count = sum(t.shape[-1] for t in tensors)
        mean = total / count
    for f in files:
        t = load(f)
        new = zeros_like(t)
        if mean is not None:
            new = new + mean[:, None]
        save(new, f)
    marker.write_text(str(len(files)))
    return len(files)


def _sentence_key(name: str):
    nums = re.findall(r"\d+", Path(name).stem)
    kind = _kind(name)
    return (kind, int(nums[-1])) if nums and kind else None


def dedupe_by_sentence(files: list[Path], prefer_sr: int = 44100) -> list[Path]:
    """Keep one recording per ITA sentence (zips often ship 48 kHz and 44.1 kHz copies).

    Prefers the file whose sample rate is ``prefer_sr``; files that don't look like
    ITA names are kept as they are (match_ita_files reports them as unmatched).
    """
    import soundfile as sf

    best: dict[tuple, Path] = {}
    others: list[Path] = []
    for f in sorted(files, key=lambda p: str(p)):
        key = _sentence_key(Path(f).name)
        if key is None:
            others.append(f)
            continue
        cur = best.get(key)
        if cur is None or (sf.info(str(cur)).samplerate != prefer_sr
                           and sf.info(str(f)).samplerate == prefer_sr):
            best[key] = f
    return sorted(best.values(), key=lambda p: str(p)) + others


def find_dataset_zip(datasets_dir: Path, name: str) -> Path | None:
    """Exact name first, else a renamed copy like 'name (1).zip' or 'name-2.zip'."""
    d = Path(datasets_dir)
    exact = d / name
    if exact.exists():
        return exact
    stem = Path(name).stem
    hits = sorted(p for p in d.glob("*.zip") if p.stem.startswith(stem))
    return hits[0] if hits else None
