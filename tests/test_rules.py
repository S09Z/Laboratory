import copy

from thaijp_accent.rules import REGISTRY, apply_rules, load_rules
from thaijp_accent.types import AccentLevel as L, Syllable


def syl(onset, vowel, long=False, coda=None, tone=0, cluster=None, w=0):
    return Syllable(onset, cluster, vowel, long, coda, tone, w)


def test_yaml_covers_33_rules():
    rules = load_rules()
    assert {r.id for r in rules} == set(range(1, 34))
    for r in rules:
        if r.impl in ("syllable", "mapper"):
            assert r.id in REGISTRY, r.id


def test_levels_match_report():
    lv = {r.id: r.level for r in load_rules()}
    assert (lv[18], lv[15], lv[9], lv[23]) == (L.LIGHT, L.MEDIUM, L.STRONG, L.MEDIUM)


def test_rule1_l_to_r_in_cluster_light():
    out, log = apply_rules([syl("p", "a", True, cluster="l")], L.LIGHT, "ปลา")
    assert out[0].cluster == "r"
    assert (1, 0) in log


def test_rule4_n_before_k():
    out, _ = apply_rules([syl("k", "i", coda="n"), syl("k", "a")], L.LIGHT, "x")
    assert out[0].coda == "ŋ"


def test_rule5_ng_before_t():
    out, _ = apply_rules([syl("k", "a", coda="ŋ"), syl("t", "a")], L.LIGHT, "x")
    assert out[0].coda == "n"


def test_rule18_shorten_closed_long_light():
    out, _ = apply_rules([syl("t", "a", True, coda="ŋ"), syl("m", "a", True, w=1)], L.LIGHT, "x")
    assert out[0].long is False
    assert out[1].long is True


def test_rule25_high_tone_flattened():
    out, _ = apply_rules([syl("m", "a", True, tone=3)], L.LIGHT, "x")
    assert out[0].tone in (0, 4)


def test_rule26_low_to_mid():
    out, _ = apply_rules([syl("k", "a", True, tone=1)], L.LIGHT, "x")
    assert out[0].tone == 0


def test_rule7_only_at_strong():
    s = [syl("s", "o", coda="m")]
    assert apply_rules(s, L.STRONG, "x")[0][0].hints.get("coda_mode") == "mu"
    assert apply_rules(s, L.MEDIUM, "x")[0][0].hints.get("coda_mode") != "mu"


def test_rule16_phrase_final_epenthetic_strong():
    out, _ = apply_rules([syl("pʰ", "u", True), syl("k", "e", coda="t", tone=1)], L.STRONG, "x")
    assert out[-1].hints["coda_mode"] == "epenthetic"


def test_deterministic():
    s = [syl("t", "a"), syl("t", "i"), syl("t", "u", w=1)]
    assert apply_rules(s, L.LIGHT, "abc") == apply_rules(s, L.LIGHT, "abc")


def test_input_not_mutated():
    s = [syl("p", "a", True, cluster="l")]
    before = copy.deepcopy(s)
    apply_rules(s, L.STRONG, "x")
    assert s == before
