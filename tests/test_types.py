from thaijp_accent.types import AccentLevel, Syllable


def test_accent_level_ordering():
    assert AccentLevel.LIGHT < AccentLevel.MEDIUM < AccentLevel.STRONG


def test_syllable_hints_default_independent():
    a = Syllable("p", None, "a", True, None, 0, 0)
    b = Syllable("p", None, "a", True, None, 0, 0)
    a.hints["x"] = "1"
    assert b.hints == {}
