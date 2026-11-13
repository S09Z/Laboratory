import pickle
from pathlib import Path

import numpy as np
import soundfile as sf

from thaijp_accent import data_prep as dp

SR = 24000


def wav(path, seconds=2.0, amp=0.3):
    t = np.arange(int(SR * seconds)) / SR
    sf.write(path, amp * np.sin(2 * np.pi * 220 * t), SR)
    return path


def test_qc_flags_short(tmp_path):
    assert "too_short" in dp.qc_clip(wav(tmp_path / "a.wav", 0.5))


def test_qc_flags_clipping(tmp_path):
    assert "clipping" in dp.qc_clip(wav(tmp_path / "a.wav", 2.0, amp=1.5))


def test_qc_clean_clip_ok(tmp_path):
    assert dp.qc_clip(wav(tmp_path / "a.wav")) == []


def test_parse_ita_transcript():
    t = dp.parse_ita_transcript("EMOTION100_001:えっ嘘でしょ。,エッウソデショ。\nRECITATION324_002:ツァツォに旅行した。,ツァツォニリョコーシタ。\n")
    assert t[("EMOTION", 1)] == "えっ嘘でしょ。"
    assert t[("RECITATION", 2)] == "ツァツォに旅行した。"


def test_match_ita_files_by_kind_and_number():
    t = {("EMOTION", 1): "a", ("RECITATION", 1): "b"}
    files = [Path("emoNormal001.wav"), Path("recitation001.wav"), Path("readme.wav")]
    entries, unmatched = dp.match_ita_files(files, t)
    assert entries == [("emoNormal001.wav", "a"), ("recitation001.wav", "b")]
    assert unmatched == ["readme.wav"]


def test_write_esd_list_format(tmp_path):
    out = tmp_path / "esd.list"
    dp.write_esd_list([("a.wav", "こんにちは")], "amitaro", out)
    assert out.read_text(encoding="utf-8") == "a.wav|amitaro|JP|こんにちは\n"


def _save(obj, path):
    Path(path).write_bytes(pickle.dumps(obj))


def _load(path):
    return pickle.loads(Path(path).read_bytes())


def test_neutralize_zeros_keeps_shape(tmp_path):
    _save(np.ones((1024, 7)), tmp_path / "x.bert.pt")
    n = dp.neutralize_bert(tmp_path, "zeros", load=_load, save=_save, zeros_like=np.zeros_like)
    arr = _load(tmp_path / "x.bert.pt")
    assert n == 1 and arr.shape == (1024, 7) and not arr.any()


def test_neutralize_idempotent(tmp_path):
    _save(np.ones((1024, 3)), tmp_path / "x.bert.pt")
    kw = dict(load=_load, save=_save, zeros_like=np.zeros_like)
    dp.neutralize_bert(tmp_path, "zeros", **kw)
    assert dp.neutralize_bert(tmp_path, "zeros", **kw) == 0


def test_dedupe_by_sentence_prefers_44k(tmp_path):
    a = wav(tmp_path / "EMOTION100_072.wav")             # 24 kHz
    d = tmp_path / "44k"
    d.mkdir()
    t = np.arange(SR * 2) / SR
    sf.write(d / "emoNormal072.wav", 0.3 * np.sin(2 * np.pi * 220 * t), 44100)
    r = wav(tmp_path / "recitation001.wav")
    kept = dp.dedupe_by_sentence([a, d / "emoNormal072.wav", r])
    assert sorted(p.name for p in kept) == ["emoNormal072.wav", "recitation001.wav"]


def test_find_dataset_zip_exact(tmp_path):
    (tmp_path / "ITAcorpus_amitaro_runrun.zip").write_bytes(b"x")
    assert dp.find_dataset_zip(tmp_path, "ITAcorpus_amitaro_runrun.zip").name == "ITAcorpus_amitaro_runrun.zip"


def test_find_dataset_zip_renamed_copy(tmp_path):
    (tmp_path / "ITAcorpus_amitaro_runrun (1).zip").write_bytes(b"x")
    assert dp.find_dataset_zip(tmp_path, "ITAcorpus_amitaro_runrun.zip").name == "ITAcorpus_amitaro_runrun (1).zip"


def test_find_dataset_zip_none(tmp_path):
    (tmp_path / "ITAcorpus_amitaro_2.2.zip").write_bytes(b"x")
    assert dp.find_dataset_zip(tmp_path, "ITAcorpus_amitaro_runrun.zip") is None
