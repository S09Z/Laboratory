from colab import preflight as pf


def test_run_checks_collects_failures_without_stopping():
    def ok():
        return "fine"

    def boom():
        raise RuntimeError("broken thing")

    res = pf.run_checks([("a", ok), ("b", boom), ("c", ok)])
    assert [(r.name, r.ok) for r in res] == [("a", True), ("b", False), ("c", True)]
    assert "broken thing" in res[1].detail


def test_check_free_space_fails_below_min(monkeypatch):
    monkeypatch.setattr(pf.shutil, "disk_usage", lambda p: (100 * 2**30, 97 * 2**30, 3 * 2**30))
    import pytest
    with pytest.raises(RuntimeError, match="3.0 GB"):
        pf.check_free_space("/x", min_gb=6)


def test_check_free_space_ok(monkeypatch):
    monkeypatch.setattr(pf.shutil, "disk_usage", lambda p: (100 * 2**30, 10 * 2**30, 90 * 2**30))
    assert "90.0 GB" in pf.check_free_space("/x", min_gb=6)


def test_required_modules_listed():
    assert {"pkg_resources", "soxr", "tensorboard", "pycrfsuite"} <= set(pf.REQUIRED_MODULES)


def test_style_gen_probe_script_uses_patch_shim():
    from colab.sbv2_patches import SHIM
    script = pf.style_gen_probe_script("/tmp/a.wav")
    assert SHIM in script and script.index(SHIM) < script.index("from pyannote.audio import")
    compile(script, "probe", "exec")


def test_required_free_gb_scales_with_kept_checkpoints():
    # one more set than kept exists briefly (delete happens after the next save)
    assert pf.required_free_gb(keep=2) < 8.3 < pf.required_free_gb(keep=3) + 2
    assert pf.required_free_gb(keep=3) > pf.required_free_gb(keep=2)


def test_check_modules_flags_old_ml_dtypes(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "ml_dtypes", types.SimpleNamespace(__version__="0.4.1"))
    monkeypatch.setattr(pf, "REQUIRED_MODULES", {})
    import pytest
    with pytest.raises(RuntimeError, match="ml_dtypes"):
        pf.check_modules()


def test_ml_dtypes_ok_when_float4_present(monkeypatch):
    import sys
    import types
    monkeypatch.setitem(sys.modules, "ml_dtypes", types.SimpleNamespace(__version__="0.5.4", float4_e2m1fn=object()))
    monkeypatch.setattr(pf, "REQUIRED_MODULES", {})
    assert pf.check_modules() == "all present"
