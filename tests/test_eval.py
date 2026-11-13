import csv

from thaijp_accent.eval import make_listening_sheet, summarize, thai_cer


def test_cer_ignores_spaces():
    assert thai_cer("ปลา กิน", "ปลากิน") == 0


def test_cer_counts_errors():
    assert thai_cer("ปลา", "ปา") > 0


def test_listening_sheet_hides_condition(tmp_path):
    rows = [{"clip_id": f"c{i}", "file": f"f{i}.wav", "level": "light"} for i in range(4)]
    out = tmp_path / "sheet.csv"
    make_listening_sheet(rows, out)
    header = next(csv.reader(out.open(encoding="utf-8")))
    assert "level" not in header and header[:2] == ["clip_id", "file"]


def test_summarize_m4_pass_flag(tmp_path):
    p = tmp_path / "r.csv"
    with p.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, ["level", "bert_mode", "tone_map", "cer", "utmos", "intelligibility"])
        w.writeheader()
        w.writerow(dict(level="light", bert_mode="kana", tone_map="default", cer=0.2, utmos=3, intelligibility=4))
        w.writerow(dict(level="strong", bert_mode="kana", tone_map="default", cer=0.1, utmos=3, intelligibility=4))
    rows = {r["level"]: r for r in summarize(p)}
    assert rows["light"]["m4_pass"] is True and rows["strong"]["m4_pass"] is False


def test_transcribe_uses_dtype_not_deprecated_torch_dtype(monkeypatch, tmp_path):
    import sys, types
    seen = {}

    def fake_pipeline(task, **kw):
        seen.update(kw)
        return lambda path, generate_kwargs=None: {"text": "x"}

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=fake_pipeline))
    from thaijp_accent.eval import transcribe
    assert transcribe([tmp_path / "a.wav"]) == ["x"]
    assert "dtype" in seen and "torch_dtype" not in seen


def test_utmos_skips_github_api_validation(monkeypatch, tmp_path):
    import numpy as np, soundfile as sf, torch
    sf.write(tmp_path / "a.wav", np.zeros(16000, dtype="float32"), 16000)
    seen = {}

    def fake_load(repo, model, **kw):
        seen.update(kw)
        return lambda wav, sr: torch.tensor([3.0])

    monkeypatch.setattr(torch.hub, "load", fake_load)
    from thaijp_accent.eval import utmos
    assert utmos([tmp_path / "a.wav"]) == [3.0]
    assert seen.get("skip_validation") is True


def test_listening_sheet_is_blind_and_keeps_key(tmp_path):
    rows = [{"clip_id": f"s01_{lv}_default", "file": f"s01_{lv}_default.wav", "level": lv} for lv in ("light", "strong")]
    out = tmp_path / "sheet.csv"
    make_listening_sheet(rows, out)
    body = out.read_text(encoding="utf-8")
    assert "light" not in body and "strong" not in body
    key = (tmp_path / "sheet_key.csv").read_text(encoding="utf-8")
    assert "s01_light_default" in key


def test_listening_sheet_never_overwrites_filled_sheet(tmp_path):
    out = tmp_path / "sheet.csv"
    out.write_text("filled by user", encoding="utf-8")
    import pytest
    with pytest.raises(FileExistsError):
        make_listening_sheet([{"clip_id": "a", "file": "a.wav"}], out)


def test_transcribe_falls_back_to_torch_dtype_on_old_transformers(monkeypatch, tmp_path):
    import sys, types
    calls = []

    def old_pipeline(task, model, device, torch_dtype=None, **kw):
        if kw:
            raise TypeError(f"unexpected {list(kw)}")
        calls.append(torch_dtype)
        return lambda path, generate_kwargs=None: {"text": "y"}

    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(pipeline=old_pipeline))
    from thaijp_accent.eval import transcribe
    assert transcribe([tmp_path / "a.wav"]) == ["y"] and len(calls) == 1


def test_transcribe_uses_mps_device_string(monkeypatch, tmp_path):
    import sys, types
    seen = {}
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(
        pipeline=lambda task, **kw: seen.update(kw) or (lambda p, generate_kwargs=None: {"text": ""})))
    from thaijp_accent.eval import transcribe
    transcribe([tmp_path / "a.wav"], device="mps")
    assert seen["device"] == "mps"
