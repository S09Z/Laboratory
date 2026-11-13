"""Morae → Style-Bert-VITS2 input (phones/tones with "_" pads, word2ph per kana char)."""
from __future__ import annotations

from .kana import mora_to_kana
from .sbv2_symbols import JA_SYMBOLS, PAD, PUNCT
from .types import Mora, Sbv2Input


def to_sbv2_input(morae: list[Mora]) -> Sbv2Input:
    phones, tones, word2ph, kana = [PAD], [0], [1], []
    prev = None
    for m in morae:
        for p in m.phones:
            if p not in JA_SYMBOLS and p not in PUNCT:
                raise ValueError(f"phone {p!r} is not in the SBV2 Japanese symbol set")
        k = mora_to_kana(m, prev)
        phones += m.phones
        tones += [m.pitch] * len(m.phones)
        if len(k) == 1:
            word2ph.append(len(m.phones))
        else:  # e.g. ティ / ファ: one kana char per phone
            assert len(k) == len(m.phones), (k, m.phones)
            word2ph += [1] * len(k)
        kana.append(k)
        prev = m
    phones.append(PAD)
    tones.append(0)
    word2ph.append(1)
    return Sbv2Input(phones, tones, word2ph, "".join(kana))
