"""Core data types shared by every stage of the Thai → Japanese-accent pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class AccentLevel(IntEnum):
    LIGHT = 1
    MEDIUM = 2
    STRONG = 3


@dataclass
class Syllable:
    """One Thai syllable in IPA terms.

    tone: 0=สามัญ 1=เอก 2=โท 3=ตรี 4=จัตวา. onset "" means vowel-initial (ʔ).
    hints: instructions from the rule engine to the mora mapper.
    """

    onset: str
    cluster: str | None
    vowel: str
    long: bool
    coda: str | None
    tone: int
    word_idx: int
    hints: dict[str, str] = field(default_factory=dict)


@dataclass
class Mora:
    phones: list[str]
    pitch: int
    word_idx: int
    syl_idx: int
    is_pause: bool = False


@dataclass
class Sbv2Input:
    phones: list[str]
    tones: list[int]
    word2ph: list[int]
    kana: str
