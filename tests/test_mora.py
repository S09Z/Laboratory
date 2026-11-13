from pathlib import Path

import pytest

from thaijp_accent.g2p import FixtureG2P, parse_ipa
from thaijp_accent.mora import syllables_to_morae
from thaijp_accent.rules import apply_rules
from thaijp_accent.sbv2_symbols import JA_SYMBOLS, PUNCT
from thaijp_accent.types import AccentLevel as L, Syllable

FIX = FixtureG2P(Path(__file__).parent / "data" / "ipa_fixtures.tsv").table


def phones(morae):
    return [p for m in morae for p in m.phones]


def run(ipa, level, tone_map="default"):
    s, _ = apply_rules(parse_ipa(ipa, 0), level, ipa)
    return syllables_to_morae(s, level, tone_map)


@pytest.mark.parametrize("level", list(L))
def test_all_phones_in_symbol_set(level):
    for word, ipa in FIX.items():
        for p in phones(run(ipa, level)):
            assert p in JA_SYMBOLS | PUNCT, (word, p)


def test_phuket_strong():
    assert phones(run(FIX["ภูเก็ต"], L.STRONG)) == ["p", "u", "u", "k", "e", "q", "t", "o"]


def test_phuket_light():
    assert phones(run(FIX["ภูเก็ต"], L.LIGHT)) == ["p", "u", "u", "k", "e", "q"]


def test_cluster_epenthesis():
    m = syllables_to_morae(parse_ipa("plaː0", 0), L.LIGHT)
    assert [x.phones for x in m] == [["p", "u"], ["r", "a"], ["a"]]


def test_coda_nasal_N():
    m = syllables_to_morae(parse_ipa("ŋaːn0", 0), L.LIGHT)
    assert m[0].phones[0] == "g" and m[-1].phones == ["N"]


def test_tone_default_rising_two_mora():
    m = syllables_to_morae(parse_ipa("maː4", 0), L.LIGHT)
    assert [x.pitch for x in m] == [0, 1]


def test_tone_learner_high_equals_rising():
    hi = syllables_to_morae(parse_ipa("maː3", 0), L.LIGHT, "learner")
    ri = syllables_to_morae(parse_ipa("maː4", 0), L.LIGHT, "learner")
    assert [x.pitch for x in hi] == [x.pitch for x in ri]


def _words(n, sylls_per_word=1):
    return [Syllable("m", None, "a", False, None, 0, w) for w in range(n) for _ in range(sylls_per_word)]


def test_pause_strong_every_boundary():
    m = syllables_to_morae(_words(3), L.STRONG)
    assert sum(x.is_pause for x in m) == 2


def test_pause_light_after_multisyllable_only():
    s = [Syllable("m", None, "a", False, None, 0, 0), Syllable("k", None, "a", False, None, 0, 0),
         Syllable("t", None, "a", False, None, 0, 1), Syllable("s", None, "a", False, None, 0, 2)]
    m = syllables_to_morae(s, L.LIGHT)
    pauses = [i for i, x in enumerate(m) if x.is_pause]
    assert len(pauses) == 1 and m[pauses[0] - 1].word_idx == 0
