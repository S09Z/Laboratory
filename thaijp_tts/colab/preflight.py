"""Fast checks (1–2 min) that run before any long step, so failures show up early.

Every check runs even if an earlier one fails; the notebook prints one table and
stops only after all results are known.
"""
from __future__ import annotations

import importlib
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .sbv2_patches import SHIM

# modules the pipeline imports at some later step (name → why)
REQUIRED_MODULES = {
    "pkg_resources": "pyopenjtalk / librosa (needs setuptools<81)",
    "soxr": "transformers 5 WavLM (training discriminator)",
    "tensorboard": "train_ms_jp_extra.py",
    "pycrfsuite": "pythainlp syllable count (G2P repeat fix)",
    "pyopenjtalk": "Japanese phonemes in SBV2",
}


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str


def run_checks(checks: list[tuple[str, Callable[[], str]]]) -> list[CheckResult]:
    out = []
    for name, fn in checks:
        try:
            out.append(CheckResult(name, True, fn() or "ok"))
        except Exception as e:
            out.append(CheckResult(name, False, f"{type(e).__name__}: {e}"[:400]))
    return out


GB_PER_SAVE = 1.7   # measured: G 0.88 + D 0.56 + WD 0.01 GB (.pth) + 0.25 GB inference .safetensors
MARGIN_GB = 1.0


def required_free_gb(keep: int) -> float:
    """Peak Drive use while training: keep+1 saves exist briefly (old ones are removed after the next save)."""
    return round((keep + 1) * GB_PER_SAVE + MARGIN_GB, 1)


def check_modules() -> str:
    missing = []
    try:
        import ml_dtypes  # onnx / tensorflow on Colab need float4 types (ml_dtypes >= 0.5)
        if not hasattr(ml_dtypes, "float4_e2m1fn"):
            missing.append(f"ml_dtypes>=0.5.1 (installed {getattr(ml_dtypes, '__version__', '?')}; trainer import fails)")
    except ImportError:
        pass
    for mod, why in REQUIRED_MODULES.items():
        try:
            importlib.import_module(mod)
        except Exception:
            missing.append(f"{mod} ({why})")
    if missing:
        raise RuntimeError("missing: " + ", ".join(missing))
    return "all present"


def check_free_space(path: str, min_gb: float = 6.0) -> str:
    _total, _used, free = shutil.disk_usage(path)
    free_gb = free / 2**30
    if free_gb < min_gb:
        raise RuntimeError(f"only {free_gb:.1f} GB free on Drive; need ≥ {min_gb:.0f} GB for checkpoints")
    return f"{free_gb:.1f} GB free"


def style_gen_probe_script(wav_path: str) -> str:
    """Same import chain and call as the patched style_gen.py, on one file."""
    return (SHIM + "from pyannote.audio import Inference, Model\n"
            "import numpy as np, soundfile as sf, torch\n"
            "model = Model.from_pretrained('pyannote/wespeaker-voxceleb-resnet34-LM')\n"
            "inference = Inference(model, window='whole')\n"
            f"x, sr = sf.read({wav_path!r}, dtype='float32', always_2d=True)\n"
            "v = inference({'waveform': torch.from_numpy(x.T.copy()), 'sample_rate': sr})\n"
            "assert v.shape == (256,) and not np.isnan(v).any(), v.shape\n"
            "print('style vector', v.shape)\n")


def _run_py(code: str, cwd: Path, timeout: int = 600) -> str:
    r = subprocess.run([sys.executable, "-c", code], cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stdout + r.stderr).strip().splitlines()[-1] if (r.stdout + r.stderr).strip() else "failed")
    return (r.stdout.strip().splitlines() or ["ok"])[-1]


def check_style_gen(sbv2_dir: Path, tmp_wav: Path) -> str:
    import numpy as np
    import soundfile as sf

    t = np.arange(44100 * 2) / 44100
    sf.write(tmp_wav, (0.2 * np.sin(2 * np.pi * 220 * t)).astype("float32"), 44100)
    return _run_py(style_gen_probe_script(str(tmp_wav)), sbv2_dir)


def check_trainer_imports(sbv2_dir: Path) -> str:
    code = ("import importlib.util, warnings; warnings.filterwarnings('ignore')\n"
            "s = importlib.util.spec_from_file_location('t', 'train_ms_jp_extra.py')\n"
            "m = importlib.util.module_from_spec(s); s.loader.exec_module(m); print('trainer imports ok')\n")
    return _run_py(code, sbv2_dir)


def check_wavlm() -> str:
    code = ("from transformers import AutoModel\n"
            "m = AutoModel.from_pretrained('microsoft/wavlm-base-plus'); print('wavlm', type(m).__name__)\n")
    return _run_py(code, Path("."))


def run_preflight(sbv2_dir: Path, drive_dir: Path, min_free_gb: float = 6.1) -> list[CheckResult]:
    sbv2_dir = Path(sbv2_dir)
    return run_checks([
        ("python modules", check_modules),
        ("Drive free space", lambda: check_free_space(str(drive_dir), min_free_gb)),
        ("style_gen (pyannote)", lambda: check_style_gen(sbv2_dir, Path("/tmp/preflight.wav"))),
        ("trainer imports", lambda: check_trainer_imports(sbv2_dir)),
        ("WavLM (training)", check_wavlm),
    ])
