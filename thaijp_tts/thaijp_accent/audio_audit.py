"""Audit training clips: loudness, noise floor, bandwidth, DC offset, edge silence.

Only numpy + soundfile. Loudness is gated speech RMS in dBFS (not ITU-R BS.1770 LUFS):
good enough to spot clips that are several dB off from the rest of the corpus.
"""
from __future__ import annotations

import csv
import io
from pathlib import Path

import numpy as np

FRAME_S = 0.02
N_FFT = 2048
HOP = 1024
GATE_DB = 30.0           # speech frames: within 30 dB of the loudest frame
BW_MARGIN_DB = 6.0       # speech must exceed the noise-frame spectrum by this to count
LOUD_OUTLIER_DB = 6.0
NARROW_RATIO = 0.7       # bandwidth below 70 % of the corpus median
MIN_SNR_DB = 25.0
MAX_DC = 0.005
MAX_EDGE_S = 0.5

FIELDS = ["name", "sr", "dur_s", "peak_db", "dc", "rms_db", "speech_rms_db", "noise_floor_db",
          "snr_db", "bw_hz", "lead_s", "trail_s", "flags"]


def _db(p: np.ndarray | float) -> np.ndarray | float:
    return 10 * np.log10(np.maximum(p, 1e-12))


def _frame_db(x: np.ndarray, n: int) -> np.ndarray:
    k = len(x) // n
    if k == 0:
        return np.array([-120.0])
    return _db(np.mean(x[: k * n].reshape(k, n) ** 2, axis=1))


def _bandwidth(x: np.ndarray, sr: int) -> float:
    """Highest frequency where speech frames exceed the quiet-frame spectrum by BW_MARGIN_DB."""
    if len(x) < N_FFT * 2:
        return 0.0
    win = np.hanning(N_FFT)
    starts = range(0, len(x) - N_FFT + 1, HOP)
    spec = np.array([np.abs(np.fft.rfft(x[s:s + N_FFT] * win)) ** 2 for s in starts])
    energy = _db(spec.sum(axis=1))
    order = np.argsort(energy)
    noise = spec[order[: max(1, len(order) // 6)]].mean(axis=0)
    speech_rows = spec[energy > energy.max() - GATE_DB]
    diff = _db(speech_rows.mean(axis=0)) - _db(noise)
    diff = np.convolve(diff, np.ones(9) / 9, mode="same")
    above = np.nonzero(diff >= BW_MARGIN_DB)[0]
    return float(np.fft.rfftfreq(N_FFT, 1 / sr)[above[-1]]) if len(above) else 0.0


def analyze(x: np.ndarray, sr: int) -> dict:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        return {"sr": sr, "dur_s": 0.0}
    n = max(1, int(FRAME_S * sr))
    fdb = _frame_db(x, n)
    active = fdb > fdb.max() - GATE_DB
    speech_rms = float(_db(np.mean(10 ** (fdb[active] / 10))))
    floor = float(np.percentile(fdb, 10))
    quiet = fdb < speech_rms - 35
    voiced = np.nonzero(~quiet)[0]
    lead = float(voiced[0] * FRAME_S) if len(voiced) else len(x) / sr
    trail = float((len(fdb) - 1 - voiced[-1]) * FRAME_S) if len(voiced) else len(x) / sr
    return {"sr": sr, "dur_s": len(x) / sr, "peak_db": float(20 * np.log10(max(np.abs(x).max(), 1e-9))),
            "dc": float(x.mean()), "rms_db": float(_db(np.mean(x ** 2))),
            "speech_rms_db": speech_rms, "noise_floor_db": floor, "snr_db": speech_rms - floor,
            "bw_hz": _bandwidth(x, sr), "lead_s": lead, "trail_s": trail}


def flag_rows(rows: list[dict]) -> dict[str, list[str]]:
    """Map clip name → problems. Loudness and bandwidth are judged against the corpus median."""
    rows = [r for r in rows if r.get("dur_s")]
    if not rows:
        return {}
    med_loud = float(np.median([r["speech_rms_db"] for r in rows]))
    med_bw = float(np.median([r["bw_hz"] for r in rows]))
    med_sr = int(np.median([r["sr"] for r in rows]))
    out: dict[str, list[str]] = {}
    for r in rows:
        f = []
        if abs(r["speech_rms_db"] - med_loud) > LOUD_OUTLIER_DB:
            f.append("loudness_outlier")
        if med_bw and r["bw_hz"] < NARROW_RATIO * med_bw:
            f.append("narrowband")
        if r["snr_db"] < MIN_SNR_DB:
            f.append("noisy")
        if abs(r["dc"]) > MAX_DC:
            f.append("dc_offset")
        if max(r["lead_s"], r["trail_s"]) > MAX_EDGE_S:
            f.append("long_silence")
        if r["sr"] != med_sr:
            f.append("sample_rate")
        if f:
            out[r["name"]] = f
    return out


def audit_dir(path: Path) -> tuple[list[dict], dict[str, list[str]]]:
    import soundfile as sf

    rows = []
    for p in sorted(Path(path).glob("*.wav")):
        x, sr = sf.read(str(p), always_2d=False)
        rows.append({"name": p.name, **analyze(x, sr)})
    return rows, flag_rows(rows)


def to_csv(rows: list[dict], flags: dict[str, list[str]]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, FIELDS, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({**{k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()},
                    "flags": " ".join(flags.get(r["name"], []))})
    return buf.getvalue()
