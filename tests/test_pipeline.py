import csv
from pathlib import Path

import pytest

from thaijp_accent.g2p import FixtureG2P
from thaijp_accent.pipeline import ThaiJPFrontend, load_sentences
from thaijp_accent.types import AccentLevel as L

DATA = Path(__file__).parent / "data"
FE = ThaiJPFrontend(FixtureG2P(DATA / "ipa_fixtures.tsv"))
SENTS = load_sentences(DATA / "test_sentences.tsv")


def test_thirty_sentences_five_m0():
    assert len(SENTS) == 30
    assert sum("m0" in s["tags"] for s in SENTS) == 5


@pytest.mark.parametrize("level", list(L))
@pytest.mark.parametrize("tone_map", ["default", "learner"])
def test_invariants(level, tone_map):
    for s in SENTS:
        inp, info = FE.convert(s["text"], level, tone_map)
        assert info["warnings"] == [], (s["id"], info["warnings"])
        assert len(inp.phones) == len(inp.tones)
        assert sum(inp.word2ph) == len(inp.phones)
        assert len(inp.word2ph) == len(inp.kana) + 2
        assert set(inp.tones) <= {0, 1}
        assert inp.phones[0] == inp.phones[-1] == "_"


def test_sentence_coverage():
    seen = set()
    for s in SENTS:
        for syl in FE.syllables(s["text"])[0]:
            seen.add(f"tone{syl.tone}")
            seen.add("long" if syl.long else "short")
            if syl.cluster:
                seen.add("cluster")
            if syl.vowel in ("ia", "ɯa", "ua"):
                seen.add("diphthong")
            if syl.coda:
                seen.add(f"coda_{syl.coda}")
    need = {f"tone{i}" for i in range(5)} | {"long", "short", "cluster", "diphthong"} | \
        {f"coda_{c}" for c in ["p", "t", "k", "m", "n", "ŋ", "w", "j"]}
    assert need <= seen, need - seen


def test_empty_input_raises():
    with pytest.raises(ValueError, match="no Thai syllables"):
        FE.convert("  ...  ", L.LIGHT)


def test_phuket_kana_strong():
    inp, _ = FE.convert("ภูเก็ต", L.STRONG)
    assert inp.kana == "プーケット"


def test_mixed_digits_and_latin():
    inp, info = FE.convert("ราคา 50 บาท OK", L.LIGHT)
    assert any("OK" in w for w in info["warnings"])
    assert len(inp.phones) > 2


def test_deterministic_end_to_end():
    a = FE.convert("ไป เที่ยว ภูเก็ต", L.LIGHT)
    b = FE.convert("ไป เที่ยว ภูเก็ต", L.LIGHT)
    assert a[0] == b[0] and a[1]["rule_log"] == b[1]["rule_log"]
