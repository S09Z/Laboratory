"""Small, idempotent patches to Style-Bert-VITS2 2.7.0 for the current Colab image.

style_gen.py loads pyannote.audio 3.x, which is older than the rest of Colab:

* torchaudio >= 2.9 removed AudioMetaData / list_audio_backends / info / load
  (I/O moved to torchcodec). What style_gen really needs (compliance.kaldi.fbank,
  functional.resample) still exists, so we add placeholders for the removed
  names and hand pyannote an in-memory waveform read with soundfile.
* huggingface_hub >= 1.0 dropped ``use_auth_token``; pyannote still passes it to
  ``hf_hub_download`` — we map it to ``token``.
* torch >= 2.6 defaults ``torch.load(weights_only=True)``, which rejects the
  official pyannote checkpoint; TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD restores the old
  default for this process only (the checkpoint comes from pyannote's own HF repo).
"""
from __future__ import annotations

from pathlib import Path

IMPORT_LINE = "from pyannote.audio import Inference, Model"
CALL_LINE = "    return inference(wav_path)  # type: ignore"

MARKER_V1 = "# --- thaijp_tts torchaudio>=2.9 compat ---"
MARKER = "# --- thaijp_tts compat v2 (torchaudio>=2.9, huggingface_hub>=1, torch>=2.6) ---"

TORCHAUDIO_COMPAT = '''import torchaudio as _ta
if not hasattr(_ta, "AudioMetaData"):
    class _AudioMetaData:  # only used in type hints by pyannote 3.x
        pass
    _ta.AudioMetaData = _AudioMetaData
if not hasattr(_ta, "list_audio_backends"):
    _ta.list_audio_backends = lambda: ["soundfile"]
'''

HF_COMPAT = '''import huggingface_hub as _hh
_hf_orig = _hh.hf_hub_download
def _hf_compat(*args, use_auth_token=None, **kwargs):
    if use_auth_token is not None and "token" not in kwargs:
        kwargs["token"] = use_auth_token
    return _hf_orig(*args, **kwargs)
_hh.hf_hub_download = _hf_compat
'''

TORCH_COMPAT = '''import os as _os
_os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
'''

SHIM_V1 = MARKER_V1 + "\n" + TORCHAUDIO_COMPAT
SHIM = MARKER + "\n" + TORCH_COMPAT + TORCHAUDIO_COMPAT + HF_COMPAT

CALL = '''    import soundfile as _sf
    _x, _sr = _sf.read(wav_path, dtype="float32", always_2d=True)
    return inference({"waveform": torch.from_numpy(_x.T.copy()), "sample_rate": _sr})  # type: ignore'''


def patch_style_gen(sbv2_dir: Path) -> bool:
    """Return True if the file was (re)patched now, False if already up to date."""
    path = Path(sbv2_dir) / "style_gen.py"
    src = path.read_text(encoding="utf-8")
    if MARKER in src:
        return False
    if MARKER_V1 in src:  # undo the first version before applying the current one
        src = src.replace(SHIM_V1, "", 1).replace(CALL, CALL_LINE, 1)
    if IMPORT_LINE not in src or CALL_LINE not in src or MARKER_V1 in src:
        raise RuntimeError("style_gen.py differs from SBV2 2.7.0; patch not applied")
    src = src.replace(IMPORT_LINE, SHIM + IMPORT_LINE, 1).replace(CALL_LINE, CALL, 1)
    path.write_text(src, encoding="utf-8")
    return True
