import numpy as np
import pytest

from thaijp_accent import audio_audit as aa

SR = 44100


def _speechlike(seconds=2.0, sr=SR, amp=0.3, seed=0, lowpass_hz=None):
    """Noise bursts at ~4 Hz syllable rate: 'speech' on a quiet floor."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sr)
    x = rng.standard_normal(n)
    if lowpass_hz is not None:
        X = np.fft.rfft(x)
        f = np.fft.rfftfreq(n, 1 / sr)
        X[f > lowpass_hz] = 0
        x = np.fft.irfft(X, n)
    x = x / np.max(np.abs(x))
    env = np.maximum(np.sin(2 * np.pi * 4 * np.arange(n) / sr), 0) ** 2  # smooth: no broadband edge splatter
    return amp * x * env + 1e-4 * rng.standard_normal(n)


def test_bandwidth_detects_lowpassed_audio():
    full = aa.analyze(_speechlike(), SR)
    narrow = aa.analyze(_speechlike(lowpass_hz=8000), SR)
    assert full["bw_hz"] > 18000
    assert 7000 < narrow["bw_hz"] < 9000


def test_loudness_scales_with_amplitude():
    a = aa.analyze(_speechlike(amp=0.3), SR)
    b = aa.analyze(_speechlike(amp=0.15), SR)
    assert a["speech_rms_db"] - b["speech_rms_db"] == pytest.approx(6.0, abs=0.5)


def test_dc_offset_and_peak():
    x = _speechlike() + 0.02
    r = aa.analyze(x, SR)
    assert r["dc"] == pytest.approx(0.02, abs=0.002)
    assert r["peak_db"] < 0


def test_snr_and_noise_floor():
    r = aa.analyze(_speechlike(), SR)
    assert r["noise_floor_db"] < -60
    assert r["snr_db"] > 40
    noisy = _speechlike() + 0.02 * np.random.default_rng(1).standard_normal(int(2 * SR))
    assert aa.analyze(noisy, SR)["snr_db"] < r["snr_db"] - 15


def test_leading_trailing_silence():
    x = np.concatenate([np.zeros(SR), _speechlike(1.0), np.zeros(SR // 2)])
    r = aa.analyze(x, SR)
    assert r["lead_s"] == pytest.approx(1.0, abs=0.1)
    assert r["trail_s"] >= 0.4


def test_stereo_is_downmixed():
    mono = _speechlike()
    r = aa.analyze(np.stack([mono, mono], axis=1), SR)
    assert r["dur_s"] == pytest.approx(2.0, abs=0.01)


def test_empty_audio_does_not_crash():
    r = aa.analyze(np.zeros(0), SR)
    assert r["dur_s"] == 0


def test_flag_outliers():
    base = {"name": "ok", "sr": SR, "dur_s": 2.0, "speech_rms_db": -20.0, "bw_hz": 21000.0, "snr_db": 50.0,
            "dc": 0.0, "lead_s": 0.1, "trail_s": 0.1}
    rows = [dict(base, name=f"f{i}") for i in range(9)]
    rows.append(dict(base, name="quiet", speech_rms_db=-32.0))
    rows.append(dict(base, name="narrow", bw_hz=9000.0))
    rows.append(dict(base, name="noisy", snr_db=15.0))
    rows.append(dict(base, name="dc", dc=0.01))
    rows.append(dict(base, name="gap", lead_s=0.9))
    rows.append(dict(base, name="sr", sr=22050))
    flags = aa.flag_rows(rows)
    assert flags["quiet"] == ["loudness_outlier"]
    assert flags["narrow"] == ["narrowband"]
    assert flags["noisy"] == ["noisy"]
    assert flags["dc"] == ["dc_offset"]
    assert flags["gap"] == ["long_silence"]
    assert flags["sr"] == ["sample_rate"]
    assert "f0" not in flags


def test_audit_dir_end_to_end(tmp_path):
    import soundfile as sf

    for i in range(3):
        sf.write(tmp_path / f"a{i}.wav", _speechlike(seed=i), SR)
    sf.write(tmp_path / "narrow.wav", _speechlike(lowpass_hz=6000), SR)
    rows, flags = aa.audit_dir(tmp_path)
    assert len(rows) == 4
    assert flags["narrow.wav"] == ["narrowband"]
    csv_text = aa.to_csv(rows, flags)
    assert csv_text.splitlines()[0].startswith("name,")
    assert "narrowband" in csv_text
