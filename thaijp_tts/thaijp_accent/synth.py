"""Style-Bert-VITS2 2.7.0 inference with our own phones/tones.

Verified against the 2.7.0 source: ``TTSModel.infer`` takes ``given_phone`` and
``given_tone`` (pads "_" included, like ``clean_text`` output) and adjusts
word2ph itself, but only when ``line_split=False``. BERT features come from
``style_bert_vits2.models.infer.extract_bert_feature``; we always wrap that
function while synthesizing:

* "kana": real BERT features, cast to float32 — on current Colab the Japanese
  DeBERTa returns float16, which crashes the float32 voice model with
  "Input type (c10::Half) and bias type (float) should be the same".
* "zeros" / "mean" (experiment E2): neutral features of the right length.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Literal

from .types import Sbv2Input


@contextmanager
def _patched_bert(infer_module, mode, zeros=None, mean_bert=None):
    original = infer_module.extract_bert_feature

    def wrapped(norm_text, word2ph, *args, **kwargs):
        if mode == "kana":
            return original(norm_text, word2ph, *args, **kwargs).float()
        z = zeros((1024, sum(word2ph)))
        if mode == "mean":
            z = z + mean_bert[:, None]
        return z

    infer_module.extract_bert_feature = wrapped
    try:
        yield
    finally:
        infer_module.extract_bert_feature = original


def synthesize(model, inp: Sbv2Input, bert_mode: Literal["kana", "zeros", "mean"] = "kana",
               style: str = "Neutral", style_weight: float = 1.0, length: float = 1.0,
               intonation_scale: float = 1.0, mean_bert=None, infer_module=None, zeros=None):
    """Return (sampling_rate, int16 audio)."""
    kwargs = {"text": inp.kana, "given_phone": list(inp.phones),
              "given_tone": list(inp.tones), "line_split": False, "style": style,
              "style_weight": style_weight, "length": length,
              "intonation_scale": intonation_scale}
    if infer_module is None:
        from style_bert_vits2.models import infer as infer_module  # type: ignore
    if bert_mode != "kana" and zeros is None:
        import torch
        zeros = torch.zeros
    if bert_mode == "mean" and mean_bert is None:
        raise ValueError("bert_mode='mean' needs mean_bert (1024-dim)")
    with _patched_bert(infer_module, bert_mode, zeros, mean_bert):
        return model.infer(**kwargs)
