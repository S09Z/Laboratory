from colab.sbv2_patches import patch_style_gen

STYLE_GEN_270 = '''import argparse
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import torch
from numpy.typing import NDArray
from pyannote.audio import Inference, Model
from tqdm import tqdm


# 推論時にインポートするために短いが関数を書く
def get_style_vector(wav_path: str) -> NDArray[Any]:
    return inference(wav_path)  # type: ignore
'''


def test_patch_style_gen_adds_shim_before_pyannote_import(tmp_path):
    (tmp_path / "style_gen.py").write_text(STYLE_GEN_270, encoding="utf-8")
    assert patch_style_gen(tmp_path) is True
    src = (tmp_path / "style_gen.py").read_text(encoding="utf-8")
    assert src.index("AudioMetaData") < src.index("from pyannote.audio import")
    assert "list_audio_backends" in src
    assert "return inference(wav_path)  # type: ignore" not in src
    assert '"waveform"' in src and "soundfile" in src
    compile(src, "style_gen.py", "exec")


def test_patch_style_gen_idempotent(tmp_path):
    (tmp_path / "style_gen.py").write_text(STYLE_GEN_270, encoding="utf-8")
    patch_style_gen(tmp_path)
    once = (tmp_path / "style_gen.py").read_text(encoding="utf-8")
    assert patch_style_gen(tmp_path) is False
    assert (tmp_path / "style_gen.py").read_text(encoding="utf-8") == once


def test_patch_style_gen_refuses_unknown_source(tmp_path):
    (tmp_path / "style_gen.py").write_text("print('changed upstream')\n", encoding="utf-8")
    import pytest
    with pytest.raises(RuntimeError):
        patch_style_gen(tmp_path)


def test_patch_adds_hf_token_compat_before_pyannote(tmp_path):
    (tmp_path / "style_gen.py").write_text(STYLE_GEN_270, encoding="utf-8")
    patch_style_gen(tmp_path)
    src = (tmp_path / "style_gen.py").read_text(encoding="utf-8")
    assert src.index("use_auth_token") < src.index("from pyannote.audio import")
    assert "TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD" in src


def test_hf_wrapper_maps_use_auth_token_to_token(tmp_path, monkeypatch):
    import sys
    import types
    calls = {}
    fake_hh = types.SimpleNamespace(hf_hub_download=lambda *a, **k: calls.update(k) or "path")
    monkeypatch.setitem(sys.modules, "huggingface_hub", fake_hh)
    from colab.sbv2_patches import HF_COMPAT
    exec(HF_COMPAT, {})
    assert fake_hh.hf_hub_download("repo", "f.bin", use_auth_token="tok") == "path"
    assert calls == {"token": "tok"}
    fake_hh.hf_hub_download("repo", "f.bin", use_auth_token=None)
    assert "use_auth_token" not in calls


def test_upgrades_file_patched_by_previous_version(tmp_path):
    from colab import sbv2_patches as sp
    old = STYLE_GEN_270.replace(sp.IMPORT_LINE, sp.SHIM_V1 + sp.IMPORT_LINE).replace(sp.CALL_LINE, sp.CALL)
    (tmp_path / "style_gen.py").write_text(old, encoding="utf-8")
    assert patch_style_gen(tmp_path) is True
    src = (tmp_path / "style_gen.py").read_text(encoding="utf-8")
    assert src.count("from pyannote.audio import") == 1
    assert src.count("_AudioMetaData:") == 1 and "use_auth_token" in src
    compile(src, "style_gen.py", "exec")
