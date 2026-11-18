"""Audit a folder of wav clips: python scripts/audio_audit.py DIR [--csv out.csv]."""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from thaijp_accent import audio_audit as aa  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dir", type=Path)
    ap.add_argument("--csv", type=Path)
    a = ap.parse_args()
    rows, flags = aa.audit_dir(a.dir)
    if not rows:
        print("ไม่พบไฟล์ .wav ใน", a.dir)
        return 1
    if a.csv:
        a.csv.write_text(aa.to_csv(rows, flags), encoding="utf-8")
        print("เขียน", a.csv)
    ok = [r for r in rows if r.get("dur_s")]
    print(f"ไฟล์ทั้งหมด {len(rows)} · sample rate: {dict(Counter(r['sr'] for r in ok))}")
    for k, label in [("speech_rms_db", "ความดังเสียงพูด dBFS"), ("noise_floor_db", "noise floor dBFS"),
                     ("snr_db", "SNR dB"), ("bw_hz", "แบนด์วิดท์ Hz"), ("lead_s", "เงียบต้น s"),
                     ("trail_s", "เงียบท้าย s")]:
        v = np.array([r[k] for r in ok])
        print(f"  {label:22s} min {v.min():8.1f}  p10 {np.percentile(v, 10):8.1f}  "
              f"median {np.median(v):8.1f}  p90 {np.percentile(v, 90):8.1f}  max {v.max():8.1f}")
    print("ปัญหาที่พบ:", dict(Counter(f for fl in flags.values() for f in fl)) or "ไม่มี")
    for name, fl in sorted(flags.items())[:30]:
        print(" ", name, fl)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
