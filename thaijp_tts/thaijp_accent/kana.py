"""Mora → katakana (used as the readable text / BERT input for Style-Bert-VITS2)."""
from __future__ import annotations

from .types import Mora

_V = {"a": 0, "i": 1, "u": 2, "e": 3, "o": 4}
_ROWS = {
    "": "アイウエオ", "k": "カキクケコ", "g": "ガギグゲゴ", "s": ["サ", "スィ", "ス", "セ", "ソ"],
    "sh": ["シャ", "シ", "シュ", "シェ", "ショ"], "t": ["タ", "ティ", "トゥ", "テ", "ト"],
    "d": ["ダ", "ディ", "ドゥ", "デ", "ド"], "ch": ["チャ", "チ", "チュ", "チェ", "チョ"],
    "ts": ["ツァ", "ツィ", "ツ", "ツェ", "ツォ"], "n": "ナニヌネノ", "h": "ハヒフヘホ",
    "f": ["ファ", "フィ", "フ", "フェ", "フォ"], "b": "バビブベボ", "p": "パピプペポ",
    "m": "マミムメモ", "y": ["ヤ", "イ", "ユ", "イェ", "ヨ"], "r": "ラリルレロ",
    "w": ["ワ", "ウィ", "ウ", "ウェ", "ウォ"], "z": "ザジズゼゾ", "j": ["ジャ", "ジ", "ジュ", "ジェ", "ジョ"],
}
_SPECIAL = {"N": "ン", "q": "ッ", ",": "、", ".": "。", "!": "！", "?": "？"}


def mora_to_kana(m: Mora, prev: Mora | None = None) -> str:
    ph = m.phones
    if len(ph) == 1 and ph[0] in _SPECIAL:
        return _SPECIAL[ph[0]]
    if len(ph) == 1 and ph[0] in _V:
        if prev is not None and prev.syl_idx == m.syl_idx and not prev.is_pause \
                and prev.phones and prev.phones[-1] == ph[0]:
            return "ー"
        return _ROWS[""][_V[ph[0]]]
    if len(ph) == 2 and ph[0] in _ROWS and ph[1] in _V:
        return _ROWS[ph[0]][_V[ph[1]]]
    raise ValueError(f"no katakana for mora {ph}")
