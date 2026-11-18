import importlib.util
from pathlib import Path

import pytest

_p = Path(__file__).resolve().parents[1] / "mac" / "train_mac.py"
_s = importlib.util.spec_from_file_location("train_mac", _p)
tm = importlib.util.module_from_spec(_s)
_s.loader.exec_module(tm)


def test_rewrite_paths_swaps_colab_prefix():
    t = f"{tm.COLAB_PREFIX}/sbv2/Data/x/wavs/a.wav|spk|JP|text"
    assert tm.rewrite_paths(t, "/mac/ds") == "/mac/ds/sbv2/Data/x/wavs/a.wav|spk|JP|text"


def test_rewrite_paths_leaves_other_text():
    assert tm.rewrite_paths("abc", "/m") == "abc"


def test_patch_source_rejects_unknown_trainer():
    with pytest.raises(ValueError):
        tm.patch_source("print('not sbv2')")
