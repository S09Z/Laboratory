import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
SRC = ROOT / "notebooks_src" / "ThaiJP_TTS_AllInOne.py"


def build(tmp_path):
    out = tmp_path / "nb.ipynb"
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", str(SRC), "-o", str(out)],
                   check=True, capture_output=True)
    return nbformat.read(out, as_version=4)


def test_notebook_builds_and_validates(tmp_path):
    nb = build(tmp_path)
    nbformat.validate(nb)


def test_drive_mounted_before_project_use(tmp_path):
    code = [c.source for c in build(tmp_path).cells if c.cell_type == "code"]
    first_import = next(i for i, s in enumerate(code) if "from thaijp_accent" in s or "from colab." in s)
    mount = next(i for i, s in enumerate(code) if "drive.mount" in s)
    assert mount < first_import


def test_every_code_cell_has_markdown_before(tmp_path):
    cells = build(tmp_path).cells
    for i, c in enumerate(cells):
        if c.cell_type == "code":
            assert i > 0 and cells[i - 1].cell_type == "markdown", c.source[:60]


def test_notebook_python_compiles(tmp_path):
    for c in build(tmp_path).cells:
        if c.cell_type == "code":
            src = "\n".join(ln for ln in c.source.splitlines() if not ln.lstrip().startswith(("%", "!")))
            compile(src, "cell", "exec")


def test_magics_restored(tmp_path):
    src = "\n".join(c.source for c in build(tmp_path).cells)
    assert "%load_ext tensorboard" in src


def _code(tmp_path):
    return "\n".join(c.source for c in build(tmp_path).cells if c.cell_type == "code")


def test_config_yml_written_before_training(tmp_path):
    src = _code(tmp_path)
    assert src.index("write_sbv2_config_yml(") < src.index("run_training(cmd")


def test_preprocess_success_checked_and_marked(tmp_path):
    src = _code(tmp_path)
    assert ".preprocess_ok" in src and "ok, msg = preprocess_all(" in src


def test_e2_copies_pretrained_init(tmp_path):
    # from SBV2/pretrained_jp_extra, not from E1's models/ (SBV2 deletes those after 3 saves)
    assert 'ensure_pretrained_init(SBV2 / "pretrained_jp_extra", E2 / "models")' in _code(tmp_path)


def test_lexicon_lives_on_drive(tmp_path):
    src = _code(tmp_path)
    assert 'LAB / "lexicon.tsv"' in src and "lexicon_path=LEX" in src


def test_gpu_checked_before_install(tmp_path):
    src = _code(tmp_path)
    assert src.index("nvidia-smi") < src.index("uv pip install")


def test_cmake_policy_set_as_env_var(tmp_path):
    src = _code(tmp_path)
    assert 'env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"' in src
    assert src.index("CMAKE_POLICY_VERSION_MINIMUM") < src.index("requirements-colab.txt --no-progress")
    assert "--no-deps" not in src   # a half-installed env must fail loudly, not continue


def test_dataset_zip_selected_and_deduped(tmp_path):
    src = _code(tmp_path)
    assert 'DATASET_ZIP = "ITAcorpus_amitaro_runrun.zip"' in src
    assert "dedupe_by_sentence(all_wavs)" in src
    assert src.index("patch_style_gen(SBV2)") < src.index("ok, msg = preprocess_all(")


def test_setup_purges_stale_project_modules(tmp_path):
    src = _code(tmp_path)
    i = src.index("z.extractall")
    assert 'startswith(("thaijp_accent", "colab."))' in src[i:i + 600] or \
           'startswith(("thaijp_accent", "colab"))' in src[i:i + 600]


def test_install_pins_setuptools_and_adds_runtime_deps(tmp_path):
    src = _code(tmp_path)
    assert '"setuptools<81"' in src and "--upgrade setuptools" not in src
    for dep in ("soxr", "tensorboard", "python-crfsuite"):
        assert dep in src


def test_preflight_runs_before_m0_and_data_prep(tmp_path):
    src = _code(tmp_path)
    assert src.index("run_preflight(") < src.index("snapshot_download(") < src.index("ok, msg = preprocess_all(")


def test_shared_helpers_defined_before_section1(tmp_path):
    cells = [c.source for c in build(tmp_path).cells if c.cell_type == "code"]
    m0 = next(i for i, s in enumerate(cells) if "snapshot_download(" in s and "M0_DIR" in s)
    before = "\n".join(cells[:m0])
    for name in ("def load_tts", "SENTS =", "from thaijp_accent.synth import synthesize", "import soundfile as sf",
                 "from IPython.display import Audio, display", "thai_cer", "transcribe"):
        assert name in before, name


def test_training_frees_gpu_and_ensures_pretrained_and_stops_on_failure(tmp_path):
    src = _code(tmp_path)
    t = src.index("run_training(cmd")
    pre = src[:t]
    assert "torch.cuda.empty_cache()" in pre[pre.rindex("## 3") if "## 3" in pre else 0:] or "free_gpu_memory(" in pre
    assert "ensure_pretrained_init(" in src
    after = src[t:]
    assert 'result in ("finished", "budget_stop")' in after


def test_e2_copy_skips_raw_and_backups(tmp_path):
    src = _code(tmp_path)
    assert 'ignore_patterns("models", "models_backup", "raw")' in src


def test_eval_dir_tied_to_model_checkpoint(tmp_path):
    src = _code(tmp_path)
    assert "EV = OUT / f\"eval_{TRAIN_NAME}_{" in src


def test_zip_validated_before_extract(tmp_path):
    src = _code(tmp_path)
    assert "zipfile.is_zipfile(" in src and ".extracted_ok" in src


def test_ml_dtypes_upgraded_and_keep_ckpts_setting(tmp_path):
    src = _code(tmp_path)
    assert '"ml_dtypes>=0.5.1"' in src and "float4_e2m1fn" in src
    assert "KEEP_CKPTS = 2" in src
    assert "keep_ckpts=KEEP_CKPTS" in src and "keep=KEEP_CKPTS" in src
    assert "required_free_gb(KEEP_CKPTS)" in src
