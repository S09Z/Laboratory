"""Accent rule engine: rewrites Syllables (or sets mapper hints) by accent level.

Rules run in id order; each rule is applied only when its level from
``rules.yaml`` is <= the requested level. "Occasional" rules (8, 19) use a
RNG seeded from the input text, so the same sentence always gives the same
output.
"""
from __future__ import annotations

import copy
import hashlib
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import yaml

from .types import AccentLevel, Syllable

RULES_PATH = Path(__file__).with_name("rules.yaml")
OCCASIONAL_P = 0.3
STOPS = {"p", "t", "k"}
NASALS = {"m", "n", "ŋ"}

_LEVELS = {"light": AccentLevel.LIGHT, "medium": AccentLevel.MEDIUM, "strong": AccentLevel.STRONG}


@dataclass(frozen=True)
class RuleMeta:
    id: int
    level: AccentLevel
    evidence: str
    impl: str
    desc: str = ""


def load_rules(path: Path = RULES_PATH) -> list[RuleMeta]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return [RuleMeta(d["id"], _LEVELS[d["level"]], d["evidence"], d["impl"], d.get("desc", ""))
            for d in data]


@dataclass
class Ctx:
    sylls: list[Syllable]
    level: AccentLevel
    rng: random.Random

    def next(self, i: int) -> Syllable | None:
        return self.sylls[i + 1] if i + 1 < len(self.sylls) else None

    def prev(self, i: int) -> Syllable | None:
        return self.sylls[i - 1] if i > 0 else None

    def word_final(self, i: int) -> bool:
        n = self.next(i)
        return n is None or n.word_idx != self.sylls[i].word_idx


RuleFn = Callable[[Ctx, int], bool]
REGISTRY: dict[int, RuleFn] = {}


def rule(rule_id: int):
    def deco(fn: RuleFn) -> RuleFn:
        REGISTRY[rule_id] = fn
        return fn
    return deco


@rule(1)
def _r1(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.cluster == "l":
        s.cluster = "r"
        return True
    return False


@rule(2)
def _r2(c: Ctx, i: int) -> bool:
    s, p = c.sylls[i], c.prev(i)
    redup = p is not None and p.word_idx != s.word_idx and \
        (p.onset, p.vowel, p.coda, p.tone) == (s.onset, s.vowel, s.coda, s.tone)
    if s.onset == "l" and (s.tone == 4 or redup):
        s.onset = "r"
        return True
    return False


@rule(3)
def _r3(c: Ctx, i: int) -> bool:
    s, hit = c.sylls[i], False
    if s.onset == "l":
        s.onset, hit = "r", True
    if s.cluster == "l":
        s.cluster, hit = "r", True
    return hit


@rule(4)
def _r4(c: Ctx, i: int) -> bool:
    s, n = c.sylls[i], c.next(i)
    if s.coda == "n" and n is not None and n.onset in ("k", "kʰ"):
        s.coda = "ŋ"
        return True
    return False


@rule(5)
def _r5(c: Ctx, i: int) -> bool:
    s, n = c.sylls[i], c.next(i)
    if s.coda == "ŋ" and n is not None and n.onset in ("t", "tʰ", "n", "l"):
        s.coda = "n"
        return True
    return False


@rule(6)
def _r6(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.coda in NASALS and "coda_mode" not in s.hints:
        s.hints["coda_mode"] = "N"
        return True
    return False


@rule(7)
def _r7(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.coda == "m":
        s.hints["coda_mode"] = "mu"
        return True
    return False


@rule(8)
def _r8(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.onset == "t" and c.rng.random() < OCCASIONAL_P:
        s.onset = "tʰ"
        return True
    return False


@rule(15)
def _r15(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.coda in STOPS:
        s.hints.setdefault("coda_mode", "q")
        return True
    return False


@rule(16)
def _r16(c: Ctx, i: int) -> bool:
    # At strong level every word boundary gets a pause (rule 31), so every
    # word-final stop is phrase-final.
    s = c.sylls[i]
    if s.coda in STOPS and c.word_final(i):
        s.hints["coda_mode"] = "epenthetic"
        return True
    return False


@rule(18)
def _r18(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.long and s.coda is not None and s.vowel not in ("ia", "ɯa", "ua"):
        s.long = False
        return True
    return False


@rule(19)
def _r19(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if not s.long and s.coda is None and c.rng.random() < OCCASIONAL_P:
        s.long = True
        return True
    return False


@rule(21)
def _r21(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.vowel == "ɤ":
        s.hints["er_vowel"] = "u"
        return True
    return False


@rule(23)
def _r23(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.vowel not in ("ia", "ɯa", "ua"):
        return False
    s.hints["ia_mode"] = "e" if (c.level >= AccentLevel.STRONG and s.vowel == "ia"
                                 and s.coda in ("ŋ", "n")) else "split"
    return True


@rule(25)
def _r25(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.tone == 3:
        s.tone = 4 if s.long else 0
        return True
    return False


@rule(26)
def _r26(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.tone == 1 and (s.long or s.vowel in ("ia", "ɯa", "ua")):
        s.tone = 0
        return True
    return False


@rule(27)
def _r27(c: Ctx, i: int) -> bool:
    s = c.sylls[i]
    if s.tone == 3:
        s.tone = 4
        return True
    return False


def _seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def apply_rules(sylls: list[Syllable], level: AccentLevel, seed_text: str
                ) -> tuple[list[Syllable], list[tuple[int, int]]]:
    ctx = Ctx(copy.deepcopy(sylls), AccentLevel(level), random.Random(_seed(seed_text)))
    log: list[tuple[int, int]] = []
    for meta in load_rules():
        fn = REGISTRY.get(meta.id)
        if fn is None or meta.level > ctx.level:
            continue
        for i in range(len(ctx.sylls)):
            if fn(ctx, i):
                log.append((meta.id, i))
    return ctx.sylls, log
