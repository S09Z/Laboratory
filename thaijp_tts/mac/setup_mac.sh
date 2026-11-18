#!/usr/bin/env bash
# One-time setup for running thaijp_tts on an Apple Silicon Mac (M1/M2/M3, 16 GB RAM is enough).
# Inference / G2P / evaluation only — training stays on Colab (SBV2's trainer is CUDA-only).
#
# Usage (from the unzipped thaijp_tts folder):
#   bash mac/setup_mac.sh
# Re-running is safe: finished steps are skipped.
set -euo pipefail

PROJ="$(cd "$(dirname "$0")/.." && pwd)"
LAB_LOCAL="${THAIJP_HOME:-$(dirname "$PROJ")/thaijp_lab}"   # default: <workspace>/thaijp_lab (อยู่ข้าง thaijp_tts)
SBV2="$LAB_LOCAL/Style-Bert-VITS2"
VENV="$LAB_LOCAL/.venv"
SBV2_TAG="2.7.0"

step() { printf "\n\033[1m==> %s\033[0m\n" "$*"; }
die()  { printf "\n\033[31m✗ %s\033[0m\n" "$*" >&2; exit 1; }

step "1/7 ตรวจเครื่อง"
[ "$(uname -s)" = "Darwin" ] || die "สคริปต์นี้สำหรับ macOS"
[ "$(uname -m)" = "arm64" ] || die "ต้องเป็น Apple Silicon (arm64) — ตอนนี้เป็น $(uname -m)"
xcode-select -p >/dev/null 2>&1 || die "ยังไม่มี Xcode Command Line Tools — รัน: xcode-select --install แล้วรันสคริปต์นี้อีกครั้ง"
echo "macOS $(sw_vers -productVersion) · $(sysctl -n machdep.cpu.brand_string) · RAM $(( $(sysctl -n hw.memsize) / 1073741824 )) GB"

step "2/7 cmake (สำหรับ compile pyopenjtalk)"
if ! command -v cmake >/dev/null 2>&1; then
  command -v brew >/dev/null 2>&1 || die "ไม่พบ Homebrew — ติดตั้งจาก https://brew.sh แล้วรันใหม่ (หรือ: pip install cmake)"
  brew install cmake
fi
# CMake 4 ไม่รับ cmake_minimum_required < 3.5 ของ pyopenjtalk
export CMAKE_POLICY_VERSION_MINIMUM=3.5

step "3/7 uv + Python 3.11"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
mkdir -p "$LAB_LOCAL"
[ -d "$VENV" ] || uv venv --python 3.11 "$VENV"
export VIRTUAL_ENV="$VENV"
export PATH="$VENV/bin:$PATH"

step "4/7 Style-Bert-VITS2 $SBV2_TAG"
[ -d "$SBV2" ] || git clone --depth 1 --branch "$SBV2_TAG" https://github.com/litagin02/Style-Bert-VITS2.git "$SBV2"

step "5/7 ติดตั้ง library (pyopenjtalk จะ compile 3–5 นาที)"
uv pip install "setuptools<81" wheel
uv pip install -r "$SBV2/requirements-infer.txt"
uv pip install pythainlp python-crfsuite jiwer soundfile sentencepiece librosa pyyaml \
               jupyterlab ipykernel ipywidgets jupytext nbformat "ml_dtypes>=0.5.1"
# SBV2 2.7.0 pin torch<2.4 → transformers 5.x (ต้องการ torch>=2.6) ใช้ไม่ได้ ต้องคุมไว้ที่ 4.x
uv pip install "transformers>=4.40,<5" "huggingface_hub<1"

step "6/7 ดาวน์โหลด BERT ญี่ปุ่น (inference เท่านั้น)"
( cd "$SBV2" && python initialize.py --only_infer --skip_default_models )
# torch<2.4 ไม่ยอมโหลด .bin ผ่าน transformers ใหม่ → แปลง BERT เป็น safetensors
python "$PROJ/mac/bin_to_safetensors.py" "$SBV2/bert/deberta-v2-large-japanese-char-wwm"

step "7/7 ลงทะเบียน kernel และสร้าง notebook"
python -m ipykernel install --user --name thaijp-mac --display-name "thaijp (Mac M3)"
python -m jupytext --to ipynb "$PROJ/notebooks_src/ThaiJP_TTS_Mac.py" -o "$PROJ/notebooks/ThaiJP_TTS_Mac.ipynb"
python - <<'PY'
import torch, transformers, pyopenjtalk  # noqa: F401
print("torch", torch.__version__, "| MPS:", torch.backends.mps.is_available(), "| transformers", transformers.__version__)
PY

cat <<EOF

✅ ติดตั้งเสร็จ
เปิด notebook:
  source "$VENV/bin/activate" && jupyter lab "$PROJ/notebooks/ThaiJP_TTS_Mac.ipynb"
แล้วเลือก kernel "thaijp (Mac M3)"   (VS Code: เปิดไฟล์ .ipynb แล้วเลือก kernel เดียวกัน)
EOF
