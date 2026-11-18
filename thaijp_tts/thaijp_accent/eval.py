"""Evaluation: Thai CER, ASR / UTMOS wrappers, blind listening sheet, summary table."""
from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path

M4_CER, M4_INTEL = 0.25, 3.5
_STRIP = re.compile(r"[\s\.,!?ฯๆ'\"“”…-]+")


def thai_cer(ref: str, hyp: str) -> float:
    import jiwer

    r, h = _STRIP.sub("", ref), _STRIP.sub("", hyp)
    if not r:
        return 0.0 if not h else 1.0
    return float(jiwer.cer(r, h))


def transcribe(wavs: list[Path], model_id: str = "openai/whisper-large-v3",
               device: str | None = None) -> list[str]:
    """Thai ASR. ``device``: "cuda" / "mps" / "cpu" (default: cuda if available)."""
    import torch
    from transformers import pipeline

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    dev = 0 if device == "cuda" else device
    try:
        asr = pipeline("automatic-speech-recognition", model=model_id, device=dev, dtype=dtype)
    except TypeError:  # transformers < 4.56 only knows torch_dtype (Mac setup pins torch<2.4)
        asr = pipeline("automatic-speech-recognition", model=model_id, device=dev, torch_dtype=dtype)
    return [asr(str(w), generate_kwargs={"language": "th", "task": "transcribe"})["text"] for w in wavs]


def utmos(wavs: list[Path]) -> list[float]:
    import librosa
    import torch

    # skip_validation avoids the rate-limited GitHub API call torch.hub makes by default
    predictor = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True,
                               skip_validation=True)
    scores = []
    for w in wavs:
        y, sr = librosa.load(str(w), sr=16000)
        scores.append(float(predictor(torch.from_numpy(y).unsqueeze(0), sr).item()))
    return scores


SHEET_COLS = ["clip_id", "file", "intelligibility_1to5", "accentedness_1to5", "transcript"]


def make_listening_sheet(rows: list[dict], out_csv: Path, seed: int = 0) -> None:
    """Blind sheet: anonymous ids (c001…) in random order; the mapping goes to <name>_key.csv.

    Refuses to overwrite an existing sheet, so ratings already typed in are never lost.
    """
    out_csv = Path(out_csv)
    if out_csv.exists():
        raise FileExistsError(f"{out_csv} already exists — keep it (it may hold ratings) or rename it first")
    rows = list(rows)
    random.Random(seed).shuffle(rows)
    key_path = out_csv.with_name(out_csv.stem + "_key.csv")
    with open(out_csv, "w", encoding="utf-8", newline="") as f, \
            open(key_path, "w", encoding="utf-8", newline="") as k:
        w = csv.DictWriter(f, SHEET_COLS)
        w.writeheader()
        kw = csv.writer(k)
        kw.writerow(["anon_id", "clip_id", "file"])
        for i, r in enumerate(rows, 1):
            anon = f"c{i:03d}"
            w.writerow({"clip_id": anon, "file": f"{anon}.wav", "intelligibility_1to5": "",
                        "accentedness_1to5": "", "transcript": ""})
            kw.writerow([anon, r["clip_id"], r["file"]])


def summarize(results_csv: Path) -> list[dict]:
    """Mean CER / UTMOS / intelligibility per (level, bert_mode, tone_map) + m4_pass."""
    groups: dict[tuple, list[dict]] = defaultdict(list)
    with open(results_csv, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            groups[(r["level"], r["bert_mode"], r["tone_map"])].append(r)

    def mean(rs, k):
        vals = [float(r[k]) for r in rs if r.get(k) not in (None, "")]
        return sum(vals) / len(vals) if vals else None

    out = []
    for (level, bert, tm), rs in sorted(groups.items()):
        cer, intel = mean(rs, "cer"), mean(rs, "intelligibility")
        out.append({"level": level, "bert_mode": bert, "tone_map": tm, "n": len(rs), "cer": cer,
                    "utmos": mean(rs, "utmos"), "intelligibility": intel,
                    "m4_pass": bool(level == "light" and cer is not None and cer <= M4_CER
                                    and intel is not None and intel >= M4_INTEL)})
    return out
