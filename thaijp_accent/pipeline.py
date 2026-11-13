"""End-to-end front-end: Thai text → Sbv2Input.

CLI:  python -m thaijp_accent.pipeline "ไป เที่ยว ภูเก็ต" --level strong [--fixtures tests/data/ipa_fixtures.tsv]
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

from .g2p import ThaiG2P, load_lexicon, thai_to_syllables
from .mora import syllables_to_morae
from .rules import apply_rules
from .sbv2_io import to_sbv2_input
from .types import AccentLevel, Sbv2Input, Syllable

DEFAULT_LEXICON = Path(__file__).with_name("lexicon.tsv")


class ThaiJPFrontend:
    def __init__(self, g2p: ThaiG2P, lexicon_path: Path | None = DEFAULT_LEXICON):
        self.g2p = g2p
        self.lexicon = load_lexicon(lexicon_path) if lexicon_path else {}

    def syllables(self, text: str) -> tuple[list[Syllable], list[str]]:
        return thai_to_syllables(text, self.g2p, self.lexicon)

    def convert(self, text: str, level: AccentLevel, tone_map: str = "default"
                ) -> tuple[Sbv2Input, dict]:
        sylls, warnings = self.syllables(text)
        if not sylls:
            raise ValueError("no Thai syllables")
        sylls, log = apply_rules(sylls, AccentLevel(level), text)
        morae = syllables_to_morae(sylls, AccentLevel(level), tone_map)
        return to_sbv2_input(morae), {"warnings": warnings, "rule_log": log, "syllables": sylls}


def load_sentences(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    for r in rows:
        r["tags"] = (r.get("tags") or "").split()
    return rows


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("--level", choices=["light", "medium", "strong"], default="light")
    ap.add_argument("--tone-map", default="default")
    ap.add_argument("--fixtures", type=Path, help="offline G2P table instead of umt5")
    a = ap.parse_args(argv)
    if a.fixtures:
        from .g2p import FixtureG2P
        g2p = FixtureG2P(a.fixtures)
    else:
        from .g2p import UmT5G2P
        g2p = UmT5G2P()
    inp, info = ThaiJPFrontend(g2p).convert(a.text, AccentLevel[a.level.upper()], a.tone_map)
    print("kana  :", inp.kana)
    print("phones:", " ".join(inp.phones))
    print("tones :", "".join(map(str, inp.tones)))
    print("rules :", sorted({r for r, _ in info["rule_log"]}))
    for w in info["warnings"]:
        print("WARN  :", w)


if __name__ == "__main__":
    main()
