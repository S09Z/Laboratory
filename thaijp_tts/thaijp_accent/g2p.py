"""Thai front-end: text normalization, G2P adapters and IPA → Syllable parsing.

``parse_ipa`` accepts the notations Thai G2P tools commonly emit, so the
real model output can be plugged in without rewriting the parser:

* syllables separated by ``.``, ``-`` or spaces
* tone as a trailing digit (0 mid, 1 low, 2 falling, 3 high, 4 rising),
  as Chao tone letters (``˧ ˨˩ ˥˩ ˦˥ ˩˩˦`` and close variants),
  or as a combining diacritic on the vowel (``à á â ǎ ā``)
* vowel length with ``ː``; a leading ``ʔ`` means a vowel-initial syllable

Notebook section 0 prints the real ``B-K/umt5-thai-g2p`` output for sample
words and runs it through this parser, so a format mismatch fails loudly.
"""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Protocol

from .types import Syllable

ONSETS = ["tɕʰ", "pʰ", "tʰ", "kʰ", "tɕ", "p", "t", "k", "b", "d", "m", "n", "ŋ",
          "f", "s", "h", "j", "w", "r", "l", "ʔ", "c", "y"]
CLUSTER_BASES = {"p", "pʰ", "t", "tʰ", "k", "kʰ", "b", "f", "d"}
CLUSTER_SECOND = {"r", "l", "w"}
# (written form, base vowel, long)
VOWELS = [
    ("iːa", "ia", True), ("iːə", "ia", True), ("ia", "ia", True), ("iə", "ia", True),
    ("ɯːa", "ɯa", True), ("ɯːə", "ɯa", True), ("ɯa", "ɯa", True), ("ɯə", "ɯa", True),
    ("uːa", "ua", True), ("uːə", "ua", True), ("ua", "ua", True), ("uə", "ua", True),
    ("aː", "a", True), ("iː", "i", True), ("ɯː", "ɯ", True), ("uː", "u", True),
    ("eː", "e", True), ("ɛː", "ɛ", True), ("oː", "o", True), ("ɔː", "ɔ", True),
    ("ɤː", "ɤ", True), ("əː", "ɤ", True),
    ("a", "a", False), ("i", "i", False), ("ɯ", "ɯ", False), ("u", "u", False),
    ("e", "e", False), ("ɛ", "ɛ", False), ("o", "o", False), ("ɔ", "ɔ", False),
    ("ɤ", "ɤ", False), ("ə", "ɤ", False),
]
CODAS = {"p": "p", "t": "t", "k": "k", "m": "m", "n": "n", "ŋ": "ŋ", "w": "w",
         "j": "j", "ʔ": None, "": None}

CHAO = {  # longest first when matching
    "˩˩˦": 4, "˨˩˦": 4, "˩˦": 4, "˨˦": 4,
    "˥˩": 2, "˦˩": 2, "˥˧": 2,
    "˦˥": 3, "˥˥": 3, "˦˦": 3, "˥": 3, "˦": 3,
    "˨˩": 1, "˩˩": 1, "˨˨": 1, "˩": 1, "˨": 1,
    "˧˧": 0, "˧": 0,
}
DIACRITICS = {"̀": 1, "́": 3, "̂": 2, "̌": 4, "̄": 0}


class ThaiG2P(Protocol):
    def word_to_ipa(self, word: str) -> str: ...


class FixtureG2P:
    """Lookup-table G2P (tests, offline runs). TSV: word<TAB>ipa; '#' comments."""

    def __init__(self, path: Path):
        self.table: dict[str, str] = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip() and not line.startswith("#"):
                word, ipa = line.split("\t")[:2]
                self.table[word.strip()] = ipa.strip()

    def word_to_ipa(self, word: str) -> str:
        if word not in self.table:
            raise KeyError(word)
        return self.table[word]


class UmT5G2P:
    """``B-K/umt5-thai-g2p`` (Apache-2.0), loaded on first use."""

    def __init__(self, model_id: str = "B-K/umt5-thai-g2p", device: str | None = None):
        self.model_id, self.device = model_id, device
        self._tok = self._model = None

    def _load(self):
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(self.model_id)
        self.device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self.device).eval()

    def word_to_ipa(self, word: str) -> str:
        if self._model is None:
            self._load()
        inputs = self._tok(word, return_tensors="pt").to(self.device)
        out = self._model.generate(**inputs, num_beams=3, max_new_tokens=64)
        return clean_umt5_output(self._tok.decode(out[0], skip_special_tokens=True), word)


_TONE_CHARS = set("˩˨˧˦˥")
_MARKS = {"\u0361": "", "\u035c": "", "\u032f": "", "\u031a": "", "ˀ": ""}


def _strip_marks(s: str) -> str:
    for k, v in _MARKS.items():
        s = s.replace(k, v)
    return s


def _orthographic_syllables(word: str) -> int | None:
    try:
        from pythainlp.tokenize import syllable_tokenize
        return len(syllable_tokenize(word))
    except Exception:
        return None


def clean_umt5_output(raw: str, word: str) -> str:
    """Normalize real ``B-K/umt5-thai-g2p`` output into ``parse_ipa`` notation.

    Observed on Colab: phones separated by spaces, syllables by " . ", Chao tone
    letters sometimes repeated ("˧ ˧"), tie bars / non-syllabic marks, trailing
    junk after the tone ("ua̯j ˥˩"), and whole syllables repeated ("maː˩˩˦.maː˩˩˦")
    or a stray split before a cluster ("pa˨˩.plaː˧" for ปลา).
    """
    sylls: list[tuple[str, str]] = []
    for chunk in raw.split("."):
        phones, tone = [], None
        for tok in chunk.split():
            if set(tok) <= _TONE_CHARS:
                tone = tok
                break          # anything after the first tone is junk
            phones.append(_strip_marks(tok))
        if phones and tone:
            sylls.append(("".join(phones), tone))
    n = _orthographic_syllables(word)
    if n is not None and len(sylls) > n:
        dedup = [s for i, s in enumerate(sylls) if i == 0 or s != sylls[i - 1]]
        sylls = dedup
    if n is not None and len(sylls) > n:
        # drop "Ca" artifacts right before a syllable starting with C + cluster consonant
        kept = []
        for i, (ph, tone) in enumerate(sylls):
            nxt = sylls[i + 1][0] if i + 1 < len(sylls) else ""
            if len(ph) == 2 and ph[1] == "a" and nxt[:1] == ph[0] and nxt[1:2] in ("r", "l", "w"):
                continue
            kept.append((ph, tone))
        sylls = kept
    return ".".join(ph + tone for ph, tone in sylls)


_LATIN = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_DIGITS = re.compile(r"\d+(?:[.,]\d+)?")


def normalize_text(text: str) -> tuple[str, list[str]]:
    """Digits → Thai words; Latin words dropped (with a warning); collapse spaces."""
    from pythainlp.util import num_to_thaiword

    warnings: list[str] = []

    def num(m: re.Match) -> str:
        s = m.group(0).replace(",", "")
        try:
            return num_to_thaiword(int(float(s)))
        except Exception:
            warnings.append(f"could not read number {m.group(0)!r}; dropped")
            return " "

    text = _DIGITS.sub(num, text)
    for w in _LATIN.findall(text):
        warnings.append(f"dropped non-Thai word {w!r}")
    text = _LATIN.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text, warnings


def _split_tone(syl: str) -> tuple[str, int]:
    m = re.search(r"([0-4])$", syl)
    if m:
        return syl[: m.start()], int(m.group(1))
    for letters in CHAO:
        if syl.endswith(letters):
            return syl[: -len(letters)], CHAO[letters]
    decomposed = unicodedata.normalize("NFD", syl)
    for mark, tone in DIACRITICS.items():
        if mark in decomposed:
            return unicodedata.normalize("NFC", decomposed.replace(mark, "")), tone
    raise ValueError(f"no tone marker in syllable {syl!r}")


def _parse_syllable(raw: str, word_idx: int) -> Syllable:
    body, tone = _split_tone(raw.strip())
    # unreleased-stop mark (p̚) and glottalisation (ˀ) carry no contrast for us
    body = unicodedata.normalize("NFC", _strip_marks(body))
    onset = next((o for o in ONSETS if body.startswith(o)), None)
    if onset is None:  # vowel-initial written without ʔ
        onset = ""
    rest = body[len(onset):]
    onset = {"ʔ": "", "c": "tɕ", "y": "j"}.get(onset, onset)
    cluster = None
    if (onset in CLUSTER_BASES and rest[:1] in CLUSTER_SECOND and len(rest) > 1
            and any(rest[1:].startswith(v[0]) for v in VOWELS)):
        cluster, rest = rest[0], rest[1:]
    vowel = next((v for v in VOWELS if rest.startswith(v[0])), None)
    if vowel is None:
        raise ValueError(f"no vowel in syllable {raw!r}")
    written, base, long = vowel
    rest = rest[len(written):]
    if rest not in CODAS:
        raise ValueError(f"unknown coda {rest!r} in syllable {raw!r}")
    return Syllable(onset, cluster, base, long, CODAS[rest], tone, word_idx)


def parse_ipa(ipa: str, word_idx: int) -> list[Syllable]:
    parts = [p for p in re.split(r"[.\-\s]+", ipa.strip()) if p]
    if not parts:
        raise ValueError("empty IPA")
    return [_parse_syllable(p, word_idx) for p in parts]


def load_lexicon(path: Path) -> dict[str, str]:
    if not Path(path).exists():
        return {}
    return FixtureG2P(path).table


def thai_to_syllables(text: str, g2p: ThaiG2P, lexicon: dict[str, str] | None = None
                      ) -> tuple[list[Syllable], list[str]]:
    from pythainlp.tokenize import word_tokenize

    lexicon = lexicon or {}
    text, warnings = normalize_text(text)
    out: list[Syllable] = []
    word_idx = 0
    for word in word_tokenize(text, engine="newmm", keep_whitespace=False):
        if not re.search(r"[฀-๿]", word):
            continue  # punctuation
        try:
            ipa = lexicon[word] if word in lexicon else g2p.word_to_ipa(word)
            sylls = parse_ipa(ipa, word_idx)
        except Exception as e:
            warnings.append(f"skipped word {word!r}: {e}")
            continue
        out.extend(sylls)
        word_idx += 1
    return out, warnings
