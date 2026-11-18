"""Syllable → Japanese mora mapping, tone → mora pitch, and word pauses (rule 31)."""
from __future__ import annotations

from pathlib import Path

import yaml

from .types import AccentLevel, Mora, Syllable

TONE_MAPS_PATH = Path(__file__).with_name("tone_maps.yaml")

ONSET = {"p": "p", "pʰ": "p", "t": "t", "tʰ": "t", "k": "k", "kʰ": "k", "b": "b",
         "d": "d", "m": "m", "n": "n", "ŋ": "g", "f": "f", "s": "s", "h": "h",
         "tɕ": "ch", "tɕʰ": "ch", "j": "y", "w": "w", "r": "r", "l": "r", "": ""}
CLUSTER = {"r": "r", "l": "r", "w": "w"}
VOWEL = {"a": "a", "i": "i", "u": "u", "ɯ": "u", "e": "e", "ɛ": "e", "o": "o", "ɔ": "o"}
STOP_EPENTHESIS = {"p": ["p", "u"], "t": ["t", "o"], "k": ["k", "u"]}


def load_tone_maps(path: Path = TONE_MAPS_PATH) -> dict[str, dict[int, object]]:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return {name: {int(k): v for k, v in table.items()} for name, table in raw.items()}


def _epenthetic_vowel(c: str) -> str:
    return "o" if c in ("t", "d") else "u"


def _nucleus(s: Syllable) -> list[str]:
    """Vowel(s) of the syllable, one entry per mora (before length/coda)."""
    if s.vowel == "ia":
        return ["e"] if s.hints.get("ia_mode") == "e" else ["i", "a"]
    if s.vowel in ("ɯa", "ua"):
        return ["u", "a"]
    if s.vowel == "ɤ":
        return [s.hints.get("er_vowel", "o")]
    return [VOWEL[s.vowel]]


def _syllable_phones(s: Syllable) -> list[list[str]]:
    """Phones of each mora of one syllable."""
    morae: list[list[str]] = []
    c = ONSET[s.onset]
    if s.cluster:
        morae.append([c, _epenthetic_vowel(c)])
        c = CLUSTER[s.cluster]
    vowels = _nucleus(s)
    morae.append(([c] if c else []) + [vowels[0]])
    morae.extend([v] for v in vowels[1:])
    if s.long and len(vowels) == 1:
        morae.append([vowels[0]])
    mode = s.hints.get("coda_mode")
    if s.coda in ("m", "n", "ŋ"):
        morae.append(["m", "u"] if mode == "mu" else ["N"])
    elif s.coda in STOP_EPENTHESIS:
        morae.append(["q"])
        if mode == "epenthetic":
            morae.append(list(STOP_EPENTHESIS[s.coda]))
    elif s.coda == "w":
        morae.append(["u"])
    elif s.coda == "j":
        morae.append(["i"])
    return morae


def _contour(tone: int, n_morae: int, prev: int, table: dict[int, object]) -> list[int]:
    spec = table[tone]
    if spec == "carry":
        return [prev] * n_morae
    contour = list(spec)
    if n_morae == 1:
        return [1] if tone == 2 else [contour[-1]]
    return [contour[min(j, len(contour) - 1)] for j in range(n_morae)]


def _pause_after_word(level: AccentLevel, n_sylls_in_word: int) -> bool:
    return level >= AccentLevel.STRONG or n_sylls_in_word >= 2


def syllables_to_morae(sylls: list[Syllable], level: AccentLevel,
                       tone_map: str = "default") -> list[Mora]:
    table = load_tone_maps()[tone_map]
    out: list[Mora] = []
    prev_pitch = 0
    word_len: dict[int, int] = {}
    for s in sylls:
        word_len[s.word_idx] = word_len.get(s.word_idx, 0) + 1
    for i, s in enumerate(sylls):
        if i > 0 and sylls[i - 1].word_idx != s.word_idx and \
                _pause_after_word(level, word_len[sylls[i - 1].word_idx]):
            out.append(Mora([","], 0, sylls[i - 1].word_idx, -1, is_pause=True))
            prev_pitch = 0
        phones = _syllable_phones(s)
        pitches = _contour(s.tone, len(phones), prev_pitch, table)
        for ph, pi in zip(phones, pitches, strict=False):
            out.append(Mora(ph, pi, s.word_idx, i))
        prev_pitch = pitches[-1]
    return out
