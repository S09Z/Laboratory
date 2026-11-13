from pathlib import Path

import pytest

from thaijp_accent.g2p import FixtureG2P, normalize_text, parse_ipa, thai_to_syllables

FIX = Path(__file__).parent / "data" / "ipa_fixtures.tsv"


def fields(s):
    return (s.onset, s.cluster, s.vowel, s.long, s.coda, s.tone)


def test_parse_ipa_kla():
    (s,) = parse_ipa("plaː0", 0)
    assert fields(s) == ("p", "l", "a", True, None, 0)


def test_parse_ipa_phuket():
    a, b = parse_ipa("pʰuː0.ket1", 0)
    assert fields(a) == ("pʰ", None, "u", True, None, 0)
    assert fields(b) == ("k", None, "e", False, "t", 1)


def test_parse_ipa_diphthong():
    s = parse_ipa("tɕʰiaŋ0.maj1", 0)[0]
    assert (s.onset, s.vowel, s.coda) == ("tɕʰ", "ia", "ŋ")


@pytest.mark.parametrize("ipa,tone", [("paː˩˩˦", 4), ("maː˦˥", 3), ("kʰaːw˥˩", 2), ("kaj˨˩", 1), ("diː˧", 0), ("mǎː", 4)])
def test_parse_ipa_tone_notations(ipa, tone):
    assert parse_ipa(ipa, 0)[0].tone == tone


def test_parse_ipa_glottal_onset_is_empty():
    assert parse_ipa("ʔɔːk1", 0)[0].onset == ""


def test_parse_ipa_rejects_garbage():
    with pytest.raises(ValueError):
        parse_ipa("???", 0)


def test_normalize_digits():
    assert normalize_text("ราคา 50 บาท")[0] == "ราคา ห้าสิบ บาท"


def test_normalize_latin_dropped():
    text, warnings = normalize_text("ดี OK")
    assert text == "ดี"
    assert any("OK" in w for w in warnings)


class _Garbage:
    def word_to_ipa(self, word):
        return "???"


def test_unparseable_word_skipped():
    sylls, warnings = thai_to_syllables("ปลา", _Garbage())
    assert sylls == []
    assert any("ปลา" in w for w in warnings)


def test_lexicon_overrides_g2p():
    sylls, _ = thai_to_syllables("ปลา", FixtureG2P(FIX), lexicon={"ปลา": "paː4"})
    assert sylls[0].tone == 4


def test_word_idx_increments():
    sylls, _ = thai_to_syllables("ปลา กิน", FixtureG2P(FIX))
    assert [s.word_idx for s in sylls] == [0, 1]


def test_parse_ipa_unreleased_stop_mark():
    a, b = parse_ipa("kruŋ˧.tʰeːp̚˥˩", 0)
    assert b.coda == "p" and b.tone == 2 and a.cluster == "r"
