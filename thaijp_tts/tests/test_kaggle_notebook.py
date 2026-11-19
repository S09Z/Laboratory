import importlib.util
import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
_s = importlib.util.spec_from_file_location("make_kaggle", ROOT / "scripts" / "make_kaggle.py")
mk = importlib.util.module_from_spec(_s)
_s.loader.exec_module(mk)


def _nb(tmp_path):
    src = mk.build(mk.SRC.read_text(encoding="utf-8"))
    py = tmp_path / "k.py"
    py.write_text(src, encoding="utf-8")
    out = tmp_path / "k.ipynb"
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", str(py), "-o", str(out)],
                   check=True, capture_output=True)
    return nbformat.read(out, as_version=4)


def test_kaggle_notebook_valid_and_compiles(tmp_path):
    nb = _nb(tmp_path)
    nbformat.validate(nb)
    for c in nb.cells:
        if c.cell_type == "code":
            src = "\n".join(ln for ln in c.source.splitlines() if not ln.lstrip().startswith(("%", "!")))
            compile(src, "<cell>", "exec")


def test_no_colab_or_drive_dependencies(tmp_path):
    code = "\n".join(c.source for c in _nb(tmp_path).cells if c.cell_type == "code")
    for bad in ("google.colab", "drive.mount", "/content", "drive_trash", "MyDrive"):
        assert bad not in code, bad
    assert "/kaggle/working" in code and "/kaggle/input" in code


def test_committed_source_is_up_to_date():
    assert (ROOT / "notebooks_src" / "ThaiJP_TTS_Kaggle.py").read_text(encoding="utf-8") == \
        mk.build(mk.SRC.read_text(encoding="utf-8"))


def test_hf_backup_is_wired_in_before_resume_and_on_prune(tmp_path):
    code = "\n".join(c.source for c in _nb(tmp_path).cells if c.cell_type == "code")
    assert code.index("BACKUP.restore()") < code.index("find_resumable_checkpoint(MODELS)")
    assert "on_prune=_on_prune" in code and "_on_prune = BACKUP.push if BACKUP else None" in code
    assert 'HF_REPO = ""' in code
