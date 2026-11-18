import subprocess
import sys
from pathlib import Path

import nbformat

ROOT = Path(__file__).parents[1]
SRC = ROOT / "notebooks_src" / "ThaiJP_TTS_Mac.py"


def build(tmp_path):
    out = tmp_path / "mac.ipynb"
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", str(SRC), "-o", str(out)],
                   check=True, capture_output=True)
    return nbformat.read(out, as_version=4)


def test_mac_notebook_builds_and_has_no_colab_or_cuda(tmp_path):
    nb = build(tmp_path)
    nbformat.validate(nb)
    code = "\n".join(c.source for c in nb.cells if c.cell_type == "code")
    assert "google.colab" not in code and '"cuda"' not in code and "nvidia-smi" not in code
    assert "pick_device(" in code and "find_drive_lab(" in code


def test_mac_notebook_every_code_cell_has_markdown_before(tmp_path):
    cells = build(tmp_path).cells
    for i, c in enumerate(cells):
        if c.cell_type == "code":
            assert i > 0 and cells[i - 1].cell_type == "markdown", c.source[:60]


def test_mac_notebook_no_undefined_names(tmp_path):
    code = "\n".join(c.source for c in build(tmp_path).cells if c.cell_type == "code")
    f = tmp_path / "all.py"
    f.write_text(code, encoding="utf-8")
    r = subprocess.run([sys.executable, "-m", "pyflakes", str(f)], capture_output=True, text=True)
    bad = [ln for ln in r.stdout.splitlines() if "undefined name" in ln]
    assert not bad, bad


def test_setup_script_pins_python311_and_infer_requirements():
    s = (ROOT / "mac" / "setup_mac.sh").read_text(encoding="utf-8")
    for needle in ("uname -m", "arm64", "--python 3.11", "requirements-infer.txt", "--only_infer",
                   "CMAKE_POLICY_VERSION_MINIMUM", '"setuptools<81"', "ipykernel install", "set -euo pipefail"):
        assert needle in s, needle
    r = subprocess.run(["bash", "-n", str(ROOT / "mac" / "setup_mac.sh")], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
