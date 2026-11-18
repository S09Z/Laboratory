"""Real B-K/umt5-thai-g2p outputs captured on Colab (2026-11-13)."""
import pytest

from thaijp_accent.g2p import clean_umt5_output, parse_ipa

REAL = {
    "ปลา": ("p a ˨˩ .  p l  aː ˧ ˧", [("p", "l", "a", True, None, 0)]),
    "กรุงเทพ": ("k r  u  ŋ ˧ .  tʰ e ː p̚ ˥˩", [("k", "r", "u", False, "ŋ", 0), ("tʰ", None, "e", True, "p", 2)]),
    "ภูเก็ต": ("pʰ uː ˧ .  k e  t̚ ˨˩  .", [("pʰ", None, "u", True, None, 0), ("k", None, "e", False, "t", 1)]),
    "เชียงใหม่": ("t͡ɕʰ ia̯  ŋ ˧ .  m aj  ˨˩", [("tɕʰ", None, "ia", True, "ŋ", 0), ("m", None, "a", False, "j", 1)]),
    "ข้าว": ("kʰ aːw ˥˩ .  kʰ aːw  ˥˩  ua̯j  ˥˩", [("kʰ", None, "a", True, "w", 2)]),
    "ส้มตำ": ("s o  m ˥˩ .  t a m ˧", [("s", None, "o", False, "m", 2), ("t", None, "a", False, "m", 0)]),
    "เลือก": ("lɯ a̯  k̚  ˥˩ .  lɯ a̯  k̚  ˥˩", [("l", None, "ɯa", True, "k", 2)]),
    "หมา": ("m aː  ˩˩˦ .  m aː  ˩˩˦", [("m", None, "a", True, None, 4)]),
    "ม้า": ("m aː  ˦˥  ˨  ˦˥", [("m", None, "a", True, None, 3)]),
    "สวัสดี": ("s a ˨˩ .  w a t̚ ˨˩ .  d iː ˧  ua̯j  ˩˩˦",
               [("s", None, "a", False, None, 1), ("w", None, "a", False, "t", 1), ("d", None, "i", True, None, 0)]),
}


@pytest.mark.parametrize("word", list(REAL))
def test_real_umt5_output(word):
    raw, expected = REAL[word]
    sylls = parse_ipa(clean_umt5_output(raw, word), 0)
    assert [(s.onset, s.cluster, s.vowel, s.long, s.coda, s.tone) for s in sylls] == expected


def test_legit_reduplication_kept():
    # นานา really has two identical syllables; orthography says 2, so nothing is collapsed
    assert clean_umt5_output("n aː ˧ .  n aː ˧", "นานา").count(".") == 1


def test_no_tone_anywhere_raises():
    with pytest.raises(ValueError):
        parse_ipa(clean_umt5_output("p a .", "ปะ"), 0)
