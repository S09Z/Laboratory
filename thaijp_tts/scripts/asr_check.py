"""Whisper-transcribe outputs_mac/long_*.wav and print CER vs the script (scripts/synth_long.py TEXTS).

Run:  ../thaijp_lab/.venv/bin/python scripts/asr_check.py [device]   (default cpu)
Writes <Laboratory>/outputs_mac/long_asr.tsv
"""
from __future__ import annotations

import re
import sys
from collections import defaultdict
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(PROJ), str(PROJ / "scripts")]
from synth_long import TEXTS  # noqa: E402

from thaijp_accent.eval import thai_cer, transcribe  # noqa: E402

out = PROJ.parent / "outputs_mac"
wavs = sorted(out.glob("long_L??_*.wav"))
hyps = transcribe(wavs, device=sys.argv[1] if len(sys.argv) > 1 else "cpu")
by_level: dict[str, list[float]] = defaultdict(list)
rows = []
for w, h in zip(wavs, hyps):
    tid, level = re.match(r"long_(L\d\d)_(.+)\.wav", w.name).groups()
    cer = thai_cer(TEXTS[tid], h)
    by_level[level].append(cer)
    rows.append((tid, level, f"{cer:.2f}", h.strip()))
    print(tid, level, f"{cer:.2f}", h.strip(), flush=True)
(out / "long_asr.tsv").write_text("id\tlevel\tcer\thyp\n" + "\n".join("\t".join(r) for r in rows), encoding="utf-8")
print()
for lv, v in sorted(by_level.items()):
    print(f"{lv:12s} mean CER {sum(v) / len(v):.2f} (n={len(v)})")
