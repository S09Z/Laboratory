import types

import numpy as np

from thaijp_accent.synth import synthesize
from thaijp_accent.types import Sbv2Input

INP = Sbv2Input(["_", "k", "a", "_"], [0, 0, 1, 0], [1, 2, 1], "カ")


class FakeModel:
    def __init__(self, infer_mod):
        self.infer_mod, self.calls, self.bert = infer_mod, [], None

    def infer(self, **kw):
        self.calls.append(kw)
        self.bert = self.infer_mod.extract_bert_feature("t", [1, 2, 1], None, "cpu", None, 0.7)
        return 44100, np.zeros(10)


class HalfTensor:
    """Stands in for the float16 BERT output newer transformers produce."""
    dtype = "float16"

    def float(self):
        out = HalfTensor()
        out.dtype = "float32"
        return out


def fake_mod():
    return types.SimpleNamespace(extract_bert_feature=lambda *a, **k: HalfTensor())


def test_zeros_bert_shape():
    mod = fake_mod()
    m = FakeModel(mod)
    synthesize(m, INP, "zeros", infer_module=mod, zeros=np.zeros)
    assert m.bert.shape == (1024, 4) and not m.bert.any()
    assert mod.extract_bert_feature().dtype == "float16"  # patch is undone


def test_phones_passed_verbatim():
    mod = fake_mod()
    m = FakeModel(mod)
    synthesize(m, INP, "kana", infer_module=mod)
    assert m.calls[0]["given_phone"] == INP.phones
    assert m.calls[0]["given_tone"] == INP.tones
    assert m.calls[0]["line_split"] is False


def test_kana_mode_passes_text():
    mod = fake_mod()
    m = FakeModel(mod)
    synthesize(m, INP, "kana", infer_module=mod)
    assert m.calls[0]["text"] == "カ"


def test_kana_mode_casts_bert_to_float32():
    # Colab T4 crash: "Input type (c10::Half) and bias type (float) should be the same"
    mod = fake_mod()
    m = FakeModel(mod)
    synthesize(m, INP, "kana", infer_module=mod)
    assert m.bert.dtype == "float32"
    assert mod.extract_bert_feature().dtype == "float16"  # original restored
