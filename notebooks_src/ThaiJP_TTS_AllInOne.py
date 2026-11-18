# ---
# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#     provenance: []
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown]
# # 🎙️ Thai → สำเนียงญี่ปุ่น · เสียงสาว VTuber (Style-Bert-VITS2 JP-Extra)
#
# Notebook เดียวจบ: ติดตั้ง → ทดสอบแนวคิด (M0) → เตรียมข้อมูล → เทรน → สังเคราะห์ภาษาไทย → วัดผล
#
# **ก่อนเริ่ม**
# 1. `Runtime > Change runtime type > T4 GPU`
# 2. ใน Google Drive ต้องมีโฟลเดอร์ `Laboratory/` ที่มี `thaijp_tts.zip` (ไฟล์โปรเจกต์)
# 3. ทุก cell **รันซ้ำได้** — ถ้า session หลุดหรือ GPU ถูกตัด ให้กด `Runtime > Run all` ใหม่
#    งานที่ทำเสร็จแล้วจะถูกข้าม และการเทรนจะต่อจาก checkpoint ล่าสุดใน Drive เอง
#
# | ส่วน | ใช้ GPU | เวลาโดยประมาณ |
# |---|---|---|
# | 0 ติดตั้ง | – | 5–8 นาที (ครั้งแรก) |
# | 1 M0 ทดสอบแนวคิด | ✓ | 5 นาที |
# | 2 เตรียมข้อมูล | ✓ | 10 นาที |
# | 3 เทรน | ✓ | 1–4 ชม. ต่อรอบ |
# | 4 สังเคราะห์ | ✓ | ทันที |
# | 5 วัดผล | ✓ | 20–30 นาที |

# %% [markdown]
# ## ⚙️ ตั้งค่าหลัก
# **ทำอะไร:** กำหนดชื่อ run, ระดับงบเวลา และการทดลอง · **ควรเห็น:** ไม่มี output

# %%
RUN_ID = "amitaro_runrun"      # ชื่อโมเดล/การทดลอง (เปลี่ยนเมื่ออยากเริ่มใหม่)
DATASET_ZIP = "ITAcorpus_amitaro_runrun.zip"   # สไตล์るんるん (おしゃべり向け) — เหมาะกับ VTuber
EXPERIMENT = "E1"              # "E1" = BERT ปกติ · "E2" = BERT เป็นศูนย์ (neutral-BERT)
TRAIN_BUDGET_HOURS = 3.5       # Colab ฟรีมักถูกตัดราว 4–12 ชม. — หยุดอย่างปลอดภัยก่อน
EPOCHS = 100
SAVE_EVERY_STEPS = 500
KEEP_CKPTS = 2                 # จำนวน checkpoint ที่เก็บไว้ (≈1.7 GB ต่อชุด) — 2 = ล่าสุด + สำรอง 1 ชุด
SPEAKER = "amitaro"
RUN_M0 = False                 # True = รันส่วน 1 (ทดสอบแนวคิดด้วยโมเดลつくよみちゃん) — ทำครั้งเดียวพอ
RUN_EVAL = False               # True = รันส่วน 5 (วัดผล ~20–30 นาที) — เปิดเมื่อเทรนเสร็จแล้ว
PURGE_TRASHED_CHECKPOINTS = True  # ลบ checkpoint เก่าที่ถูกย้ายไปถังขยะ Drive ออกถาวร (เฉพาะไฟล์ checkpoint ของ run นี้)
SBV2_TAG = "2.7.0"
LAB = "/content/drive/MyDrive/Laboratory"

# %% [markdown]
# ## 0. ติดตั้ง
# **ทำอะไร:** mount Drive, แตกไฟล์โปรเจกต์, ติดตั้ง Style-Bert-VITS2 เวอร์ชัน pin `2.7.0`
#
# **ทำไม:** ตรึงเวอร์ชันเพื่อให้ผลทำซ้ำได้ · ครั้งแรก runtime จะ **restart เองหนึ่งครั้ง** (เหมือน notebook ทางการของ SBV2)
# หลัง restart ให้กด `Run all` อีกครั้ง — cell นี้จะข้ามการติดตั้ง
#
# **ควรเห็น:** `✅ setup ready`

# %%
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

from google.colab import drive

# ตรวจ GPU ก่อนเสียเวลาติดตั้ง (ไม่เทรนบน CPU)
assert shutil.which("nvidia-smi"), "No GPU: Runtime > Change runtime type > T4 GPU แล้วรันใหม่"

drive.mount("/content/drive")
LAB = Path(LAB)
assert (LAB / "thaijp_tts.zip").exists(), f"ไม่พบ {LAB/'thaijp_tts.zip'} — อัปโหลดไฟล์โปรเจกต์ไว้ในโฟลเดอร์ Laboratory ก่อน"

PROJ = Path("/content/thaijp_tts")
# แตก zip ใหม่ทุกครั้ง เพื่อให้ได้โค้ดล่าสุดที่อัปโหลดไว้ (ไฟล์ที่คุณแก้เอง เช่น lexicon อยู่บน Drive ไม่หาย)
shutil.rmtree(PROJ, ignore_errors=True)
with zipfile.ZipFile(LAB / "thaijp_tts.zip") as z:
    z.extractall("/content")
# ล้าง module ของโปรเจกต์ที่ import ค้างไว้ใน kernel นี้ — ไม่งั้นจะยังใช้โค้ดเก่าแม้แตก zip ใหม่แล้ว
for _m in [m for m in sys.modules if m == "colab" or m.startswith(("thaijp_accent", "colab."))]:
    del sys.modules[_m]
if str(PROJ) not in sys.path:
    sys.path.insert(0, str(PROJ))

SBV2 = Path("/content/Style-Bert-VITS2")
MARK = Path("/content/.sbv2_installed")

def sh(cmd, env):
    """รันคำสั่ง ถ้าล้มเหลวให้แสดง 40 บรรทัดสุดท้ายแล้วหยุด (ไม่ข้ามไปเงียบ ๆ)"""
    r = subprocess.run(cmd, shell=True, env=env, capture_output=True, text=True)
    if r.returncode != 0:
        print("\n".join((r.stdout + r.stderr).splitlines()[-40:]))
        raise RuntimeError(f"ติดตั้งไม่สำเร็จ: {cmd}")
    return r

if not MARK.exists():
    env = os.environ.copy()
    env["PATH"] = "/root/.local/bin:/root/.cargo/bin:" + env.get("PATH", "")
    # CMake 4 ไม่รับ cmake_minimum_required < 3.5 (pyopenjtalk) — ตั้งเป็น env var เพราะ
    # setup.py ของ pyopenjtalk เรียก cmake เองโดยตรง จึงไม่อ่าน CMAKE_ARGS
    env["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
    os.environ["CMAKE_POLICY_VERSION_MINIMUM"] = "3.5"
    print("1/5 build tools…")
    sh("apt-get -qq update && apt-get -qq install -y cmake build-essential", env)
    sh("curl -LsSf https://astral.sh/uv/install.sh | sh", env)
    print("2/5 clone Style-Bert-VITS2", SBV2_TAG, "…")
    if not SBV2.exists():
        sh(f"git clone --depth 1 --branch {SBV2_TAG} https://github.com/litagin02/Style-Bert-VITS2.git {SBV2}", env)
    # Colab ปัจจุบันเป็น Python 3.13 — pin เก่าบางตัวไม่มี wheel สำหรับ 3.13 จึงผ่อนเป็น >=
    req = SBV2 / "requirements-colab.txt"
    req.write_text(re.sub(r"^(numba|numpy|onnxruntime)==", r"\1>=", req.read_text(), flags=re.M))
    print("3/5 requirements (pyopenjtalk จะ compile เอง ใช้เวลา 3–5 นาที)…")
    # setuptools>=81 ไม่มี pkg_resources ซึ่ง pyopenjtalk และ librosa ยังใช้อยู่
    sh('uv pip install --system "setuptools<81" wheel', env)
    sh(f"cd {SBV2} && uv pip install --system -r requirements-colab.txt --no-progress", env)
    print("4/5 thaijp deps…")
    # soxr: transformers 5 ต้องใช้ตอนโหลด WavLM (training) · tensorboard: train_ms_jp_extra.py import
    sh('uv pip install --system pythainlp python-crfsuite jiwer soundfile sentencepiece soxr tensorboard "ml_dtypes>=0.5.1" --no-progress', env)
    print("5/5 initialize (ดาวน์โหลด pretrained JP-Extra)…")
    sh(f"cd {SBV2} && python initialize.py --skip_default_models", env)
    MARK.write_text(time.ctime())
    print("🔁 ติดตั้งเสร็จ กำลัง restart runtime — จากนั้นกด Runtime > Run all อีกครั้ง")
    os.kill(os.getpid(), 9)

# runtime ที่ติดตั้งไว้ก่อนหน้า (จาก notebook รุ่นเก่า) อาจขาด dependency เหล่านี้ — ติดตั้งเพิ่มเฉพาะที่ขาด
import importlib.util

_need = {"pkg_resources": '"setuptools<81"', "soxr": "soxr", "tensorboard": "tensorboard", "pycrfsuite": "python-crfsuite"}
_missing = [pkg for mod, pkg in _need.items() if importlib.util.find_spec(mod) is None]
try:   # SBV2 requirements ลด ml_dtypes ลงเป็นรุ่นที่ไม่มี float4 → onnx/tensorflow import ไม่ได้ ตัวเทรนพัง
    import ml_dtypes
    if not hasattr(ml_dtypes, "float4_e2m1fn"):
        _missing.append('"ml_dtypes>=0.5.1"')
except ImportError:
    pass
if _missing:
    print("ติดตั้งเพิ่ม:", _missing)
    _env = {**os.environ, "PATH": "/root/.local/bin:/root/.cargo/bin:" + os.environ["PATH"]}
    _uv = shutil.which("uv", path=_env["PATH"])
    _cmd = (f"{_uv} pip install --system " if _uv else f"{sys.executable} -m pip install -q ") + " ".join(_missing)
    subprocess.run(_cmd, shell=True, check=True, env=_env)

os.chdir(SBV2)
if str(SBV2) not in sys.path:
    sys.path.insert(0, str(SBV2))
DATASET_ROOT = LAB / "sbv2" / "Data"
ASSETS_ROOT = LAB / "sbv2" / "model_assets"
OUT = LAB / "outputs"
for p in (DATASET_ROOT, ASSETS_ROOT, OUT, LAB / "datasets"):
    p.mkdir(parents=True, exist_ok=True)
import yaml

with open(SBV2 / "configs" / "paths.yml", "w", encoding="utf-8") as f:
    yaml.dump({"dataset_root": str(DATASET_ROOT), "assets_root": str(ASSETS_ROOT)}, f)
print("✅ setup ready")

# %% [markdown]
# ### 0.1 ตรวจ GPU
# **ทำอะไร:** ตรวจว่าได้ GPU จริง แล้วเลือก batch size ตาม VRAM · **ควรเห็น:** ชื่อ GPU (เช่น Tesla T4) และ batch size
# ถ้าไม่ได้ GPU จะหยุดพร้อมบอกวิธีแก้ — ไม่แอบเทรนบน CPU

# %%
from colab.runtime import data_hash, ensure_gpu, pick_batch_size, write_manifest

GPU = ensure_gpu()
BATCH_SIZE = pick_batch_size(GPU["vram_gb"])
print(GPU, "→ batch_size", BATCH_SIZE)

# %% [markdown]
# ### 0.2 ตรวจ G2P ภาษาไทย (umt5-thai-g2p)
# **ทำอะไร:** แปลงคำตัวอย่างเป็น IPA แล้วให้ parser แยกพยางค์ · **ทำไม:** ถ้ารูปแบบ IPA ของโมเดลไม่ตรงกับที่ parser รู้จัก
# จะเห็นทันทีตรงนี้ ก่อนเสียเวลาเทรน · **ควรเห็น:** ทุกบรรทัดขึ้น ✅
# ถ้ามี ❌ ให้เพิ่มคำนั้นใน `Laboratory/lexicon.tsv` บน Drive (รูปแบบ `คำ<TAB>IPA` เช่น `ปลา	plaː0`)
# ไฟล์นี้อยู่บน Drive จึงไม่หายเมื่อเปลี่ยนเครื่อง · ถ้า ❌ เกือบทุกคำ แปลว่ารูปแบบ IPA ของโมเดลต่างจากที่ parser รองรับ —
# ส่ง output ของ cell นี้มาให้ผู้พัฒนาปรับ parser

# %%
try:
    import pycrfsuite  # noqa: F401  # pythainlp ใช้นับพยางค์ เพื่อตัดพยางค์ที่โมเดลพูดซ้ำ
except ImportError:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "python-crfsuite"], check=True)
from thaijp_accent.g2p import UmT5G2P, parse_ipa
from thaijp_accent.pipeline import ThaiJPFrontend, load_sentences
from thaijp_accent.types import AccentLevel

G2P = UmT5G2P()
LEX = LAB / "lexicon.tsv"
if not LEX.exists():
    shutil.copy2(PROJ / "thaijp_accent" / "lexicon.tsv", LEX)
bad = 0
for w in ["ปลา", "กรุงเทพ", "ภูเก็ต", "เชียงใหม่", "ข้าว", "ส้มตำ", "เลือก", "หมา", "ม้า", "สวัสดี"]:
    ipa = G2P.word_to_ipa(w)   # ผ่าน clean_umt5_output แล้ว (รวม phone, เลือกวรรณยุกต์แรก, ตัดพยางค์ซ้ำ)
    try:
        s = parse_ipa(ipa, 0)
        print("✅", w, ipa, [(x.onset, x.cluster, x.vowel, x.long, x.coda, x.tone) for x in s])
    except Exception as e:
        bad += 1
        print("❌", w, repr(ipa), e)
FE = ThaiJPFrontend(G2P, lexicon_path=LEX)
assert bad <= 2, f"parser อ่าน IPA ไม่ได้ {bad}/10 คำ — ดูหมายเหตุด้านบน"
print("ปัญหา", bad, "คำ (เพิ่มใน lexicon.tsv ได้)" if bad else "— parser อ่านรูปแบบ IPA ได้")

# %% [markdown]
# ### 0.3 Preflight — ตรวจทุกขั้นที่ใช้เวลานานล่วงหน้า (1–2 นาที)
# **ทำอะไร:** ทดสอบสิ่งที่ส่วน 2–5 ต้องใช้ *ก่อน* เริ่มงานยาว ๆ: library ที่ต้องมี · พื้นที่ Drive ·
# style_gen (pyannote) กับไฟล์เสียง 1 ไฟล์ · import ตัวเทรน · โหลด WavLM
#
# **ทำไม:** ไม่ให้ไปเจอ error หลังรอ preprocess หรือเทรนไปแล้ว · **ควรเห็น:** ✅ ครบ 5 บรรทัด
# ถ้ามี ❌ notebook จะหยุดตรงนี้ — ส่งตารางนี้มาให้ผู้พัฒนาดูได้เลย

# %%
from colab.preflight import run_preflight
from colab.sbv2_patches import patch_style_gen

print("patch style_gen:", patch_style_gen(SBV2))
from colab.preflight import required_free_gb

_pf = run_preflight(SBV2, LAB, min_free_gb=required_free_gb(KEEP_CKPTS))   # ช่วงเทรนมี checkpoint สูงสุด KEEP_CKPTS+1 ชุด
for r in _pf:
    print("✅" if r.ok else "❌", f"{r.name:24s}", r.detail)
assert all(r.ok for r in _pf), "preflight ไม่ผ่าน — ดูบรรทัด ❌ ด้านบน"

# %% [markdown]
# ### 0.4 เครื่องมือที่ทุกส่วนใช้ร่วมกัน
# **ทำอะไร:** import และฟังก์ชันที่ส่วน 1, 4, 5 ใช้ — อยู่ตรงนี้เพื่อให้ข้ามส่วน 1 ได้โดยไม่เกิด NameError

# %%
import gc

import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from IPython.display import Audio, display
from style_bert_vits2.tts_model import TTSModel

from thaijp_accent.eval import thai_cer, transcribe
from thaijp_accent.synth import synthesize

SENTS = load_sentences(PROJ / "tests" / "data" / "test_sentences.tsv")

def load_tts(model_dir: Path) -> TTSModel:
    sts = sorted(model_dir.rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)
    assert sts, f"ยังไม่มีโมเดลใน {model_dir} — เทรน (ส่วน 3) ให้ได้อย่างน้อย 1 checkpoint ก่อน"
    cfg = next(model_dir.rglob("config.json"))
    sv = next(model_dir.rglob("style_vectors.npy"))
    print("model:", sts[-1].name)
    return TTSModel(model_path=sts[-1], config_path=cfg, style_vec_path=sv, device="cuda")

def free_gpu_memory(*names):
    """ปล่อย GPU ที่ notebook ถือไว้ (โมเดล G2P / つくよみちゃん / Whisper cache) ก่อนเริ่มเทรน"""
    for n in names:
        globals().pop(n, None)
    if "G2P" in globals():
        G2P._model = None
    gc.collect()
    torch.cuda.empty_cache()
    print(f"GPU ว่าง {torch.cuda.mem_get_info()[0] / 2**30:.1f} GB")

# %% [markdown]
# ## 1. M0 — ทดสอบแนวคิดโดยยังไม่เทรน
# **ทำอะไร:** ใช้โมเดล つくよみちゃん ที่มีคนเทรนไว้แล้ว อ่านประโยคไทย 5 ประโยค ทั้งแบบ kana และแบบส่ง phone ตรง
#
# **ทำไม:** เช็กว่า "ภาษาไทยที่ถอดเป็นเสียงญี่ปุ่น" ฟังรู้เรื่องจริง ก่อนลงแรงเทรน
#
# **เกณฑ์ผ่าน:** คุณ (และคนไทยอีก 1–2 คน) ฟังแล้วจับใจความได้ **≥ 3 ใน 5 ประโยค** และ CER เฉลี่ย **≤ 40%**
#
# เครดิต: フリー素材キャラクター「つくよみちゃん」(© Rei Yumesaki) — โมเดล `ayousanz/tsukuyomi-chan-style-bert-vits2-model`

# %%
cers = []
if RUN_M0:
    M0_DIR = LAB / "m0_tsukuyomi"
    if not any(M0_DIR.rglob("*.safetensors")):
        snapshot_download("ayousanz/tsukuyomi-chan-style-bert-vits2-model", local_dir=M0_DIR)
    m0_model = load_tts(M0_DIR)
    m0 = [s for s in SENTS if "m0" in s["tags"]]
    (OUT / "m0").mkdir(exist_ok=True)
    m0_wavs = []
    for s in m0:
        inp, info = FE.convert(s["text"], AccentLevel.LIGHT)
        sr, audio = synthesize(m0_model, inp, "kana")
        path = OUT / "m0" / f"{s['id']}_light.wav"
        sf.write(path, audio, sr)
        m0_wavs.append(path)
        print(s["id"], s["text"], "→", inp.kana)
        display(Audio(str(path)))
    hyps = transcribe(m0_wavs)
    cers = [thai_cer(s["text"], h) for s, h in zip(m0, hyps)]
    for s, h, c in zip(m0, hyps, cers):
        print(f"{s['id']}  ASR: {h!r}  CER={c:.2f}")
    print(f"CER เฉลี่ย = {sum(cers)/len(cers):.2f}")
else:
    print("ข้ามส่วน 1 (RUN_M0 = False)")

# %% [markdown]
# **ให้คะแนน M0:** ฟังโดยไม่ดูข้อความ แล้วใส่จำนวนประโยคที่เข้าใจ (0–5)

# %%
M0_UNDERSTOOD = 0   # ← แก้ตัวเลขนี้หลังฟัง
if cers:
    m0_pass = M0_UNDERSTOOD >= 3 and sum(cers) / len(cers) <= 0.40
    print("M0", "✅ PASS — ไปต่อส่วน 2" if m0_pass else
          "❌ ยังไม่ผ่าน — ลองระดับ/tone map อื่นในส่วน 4 กับโมเดลนี้ก่อน ถ้ายังไม่ผ่าน กลับมาคุยแนวทางกันก่อนเทรน")

# %% [markdown]
# ## 2. เตรียมข้อมูลเสียง (あみたろ ITA corpus)
# **ทำอะไร:** อ่านไฟล์เสียงจาก Drive → จับคู่กับประโยค ITA → ตรวจคุณภาพ → สร้าง `esd.list` → preprocess ของ SBV2
#
# **ต้องทำเอง 1 ครั้ง:**
# 1. เปิด https://amitaro.net/ หน้า "ITAコーパス読み上げ音声" อ่าน**ข้อกำหนดการใช้งาน**
# 2. ตั้ง `I_ACCEPT_AMITARO_TERMS = True` ด้านล่าง — cell จะดาวน์โหลด `DATASET_ZIP` จาก amitaro.net มาไว้ที่
#    `Laboratory/datasets/` ให้เอง (หรือวางไฟล์ไว้เองก่อนก็ได้)
#
# **ข้อกำหนดสำคัญ:** ใส่เครดิต あみたろの声素材工房 (https://amitaro.net/) ทุกครั้งที่เผยแพร่เสียง · ห้ามใช้กับงานจำกัดอายุ ·
# ห้ามทำให้เข้าใจว่าเป็นเสียงของあみたろตัวจริง
#
# **ควรเห็น:** จำนวนไฟล์ที่จับคู่ได้ (ITA มี 424 ประโยค), ตารางไฟล์ที่มีปัญหา, และ `✅ preprocess done`

# %%
I_ACCEPT_AMITARO_TERMS = False   # ← เปลี่ยนเป็น True หลังอ่านข้อกำหนด
assert I_ACCEPT_AMITARO_TERMS, "อ่านข้อกำหนดที่ amitaro.net แล้วตั้ง I_ACCEPT_AMITARO_TERMS = True"

import urllib.parse
import urllib.request

from colab.sbv2_patches import patch_style_gen
from thaijp_accent.data_prep import (
    dedupe_by_sentence,
    find_dataset_zip,
    match_ita_files,
    parse_ita_transcript,
    qc_clip,
    write_esd_list,
)

MODEL_NAME = RUN_ID
DS = DATASET_ROOT / MODEL_NAME
RAW = DS / "raw"
if not (DS / ".preprocess_ok").exists():
    print("ไฟล์ใน Laboratory/datasets/:", sorted(p.name for p in (LAB / "datasets").iterdir()))
    zp = find_dataset_zip(LAB / "datasets", DATASET_ZIP)   # รับชื่อที่ถูกเปลี่ยน เช่น "... (1).zip"
    if zp is not None and not zipfile.is_zipfile(zp):
        print("⚠️ ไฟล์", zp.name, "ไม่ใช่ zip ที่สมบูรณ์ (ดาวน์โหลดไม่ครบ?) — ลบแล้วดาวน์โหลดใหม่")
        zp.unlink()
        zp = None
    if zp is None:
        zp = LAB / "datasets" / DATASET_ZIP
        url = "https://amitaro.net/download/corpus/" + urllib.parse.quote(DATASET_ZIP)
        print("ไม่พบ", DATASET_ZIP, "→ ดาวน์โหลด", url)
        tmp = Path("/content") / (DATASET_ZIP + ".part")
        urllib.request.urlretrieve(url, tmp)
        assert zipfile.is_zipfile(tmp), "ไฟล์ที่ดาวน์โหลดไม่ใช่ zip — เปิด amitaro.net ตรวจว่าลิงก์ยังใช้ได้"
        shutil.move(str(tmp), zp)
    else:
        print("ใช้ไฟล์ที่มีอยู่:", zp.name)
    src = LAB / "datasets" / Path(DATASET_ZIP).stem      # แยกโฟลเดอร์ต่อ zip — ไม่ปนสไตล์
    if not (src / ".extracted_ok").exists():               # แตกไม่ครบรอบก่อน → แตกใหม่
        shutil.rmtree(src, ignore_errors=True)
        with zipfile.ZipFile(zp) as z:
            z.extractall(src)
        (src / ".extracted_ok").write_text(time.ctime())
    all_wavs = sorted(src.rglob("*.wav"))
    wavs = dedupe_by_sentence(all_wavs)                  # zip มีทั้ง 48k และ 44.1k → เก็บ 44.1k ประโยคละไฟล์
    print(f"wav ทั้งหมด {len(all_wavs)} → หลังตัดซ้ำ {len(wavs)}")
    tr = {}
    for kind in ("emotion", "recitation"):
        url = f"https://raw.githubusercontent.com/mmorise/ita-corpus/main/{kind}_transcript_utf8.txt"
        tr.update(parse_ita_transcript(urllib.request.urlopen(url).read().decode("utf-8")))
    entries, unmatched = match_ita_files(wavs, tr)
    print(f"จับคู่ได้ {len(entries)} ไฟล์ · ไม่ตรง {len(unmatched)}:", unmatched[:10])
    by_name = {w.name: w for w in wavs}
    RAW.mkdir(parents=True, exist_ok=True)
    kept, problems = [], []
    for name, text in entries:
        issues = qc_clip(by_name[name])
        if issues:
            problems.append((name, issues))
            continue
        shutil.copy2(by_name[name], RAW / name)
        kept.append((name, text))
    print(f"ผ่าน QC {len(kept)} · ตัดออก {len(problems)}")
    for n, i in problems[:20]:
        print("  ⚠️", n, i)
    write_esd_list(kept, SPEAKER, DS / "esd.list")
    (DS / "CREDITS.txt").write_text(
        "音声: あみたろの声素材工房 (https://amitaro.net/) の ITA コーパス読み上げ音声を使用。\n"
        "ITA corpus: https://github.com/mmorise/ita-corpus\n", encoding="utf-8")

    print("patch style_gen (torchaudio>=2.9):", patch_style_gen(SBV2))
    from gradio_tabs.train import preprocess_all
    from style_bert_vits2.nlp.japanese import pyopenjtalk_worker
    pyopenjtalk_worker.initialize_worker()
    ok, msg = preprocess_all(model_name=MODEL_NAME, batch_size=BATCH_SIZE, epochs=EPOCHS,
                   save_every_steps=SAVE_EVERY_STEPS, num_processes=2, normalize=False, trim=True,
                   freeze_EN_bert=False, freeze_JP_bert=False, freeze_ZH_bert=False,
                   freeze_style=False, freeze_decoder=False, use_jp_extra=True,
                   val_per_lang=4, log_interval=200, yomi_error="skip")
    assert ok, f"preprocess ล้มเหลว: {msg} — รัน cell นี้ใหม่ได้เลย"
    (DS / ".preprocess_ok").write_text(time.ctime())
print("✅ preprocess done:", DS, "· data hash", data_hash(DS / "wavs"))

# %% [markdown]
# ### 2.1 (เฉพาะ E2) สร้างชุดข้อมูลแบบ neutral-BERT
# **ทำอะไร:** คัดลอกชุดข้อมูลเป็น `<RUN_ID>_E2` แล้วแทน BERT feature ทุกไฟล์ด้วยศูนย์ ·
# **ทำไม:** ตอนอ่านภาษาไทย ข้อความ kana ไม่มีความหมายภาษาญี่ปุ่น ถ้าเทรนด้วย BERT ศูนย์ ตอนใช้งานก็ส่งศูนย์ให้ตรงกัน ·
# **ควรเห็น:** จำนวนไฟล์ที่ถูกแทน (รันซ้ำจะเป็น 0)

# %%
from colab.runtime import ensure_pretrained_init
from thaijp_accent.data_prep import neutralize_bert

TRAIN_NAME = MODEL_NAME if EXPERIMENT == "E1" else f"{MODEL_NAME}_E2"
if EXPERIMENT == "E2":
    E2 = DATASET_ROOT / TRAIN_NAME
    if not E2.exists():
        shutil.copytree(DS, E2, ignore=shutil.ignore_patterns("models", "models_backup", "raw"))
        for lst in ("train.list", "val.list"):
            p = E2 / lst
            p.write_text(p.read_text(encoding="utf-8").replace(str(DS), str(E2)), encoding="utf-8")
        cfg = json.loads((E2 / "config.json").read_text(encoding="utf-8"))
        for k in ("training_files", "validation_files"):
            cfg["data"][k] = cfg["data"][k].replace(str(DS), str(E2))
        cfg["model_name"] = TRAIN_NAME
        (E2 / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    # น้ำหนักเริ่มต้น JP-Extra จาก SBV2/pretrained_jp_extra (ไม่ใช่จาก E1 — SBV2 ลบของ E1 ทิ้งหลังเซฟครบ 3 ชุด)
    print("pretrained init files:", ensure_pretrained_init(SBV2 / "pretrained_jp_extra", E2 / "models"))
    print("neutralized:", neutralize_bert(E2, "zeros"))
print("train dataset:", TRAIN_NAME)

# %% [markdown]
# ## 3. เทรน (fine-tune JP-Extra) — ทนต่อ session หลุด
# **ทำอะไร:**
# - ตรวจ checkpoint ล่าสุดใน Drive — ไฟล์ที่เขียนไม่เสร็จ/เสียจากการถูกตัดกลางคันจะถูกย้ายไป `models/corrupt/`
# - เทรนต่อจากจุดนั้นเอง, เก็บ checkpoint ทุก 500 step และเก็บไว้ `KEEP_CKPTS` ชุดล่าสุด (รวมไฟล์ .safetensors เพื่อไม่ให้ Drive เต็ม)
# - เมื่อครบ `TRAIN_BUDGET_HOURS` จะ**รอ checkpoint ถัดไปก่อน**แล้วค่อยหยุด — ไม่เสียงานที่ทำไป
# - ถ้า VRAM ไม่พอ (OOM) จะลด batch size ลงครึ่งหนึ่งแล้วลองใหม่อัตโนมัติ
#
# **ควรเห็น:** `finished` (ครบ epoch) หรือ `budget_stop` (หยุดอย่างปลอดภัย — รัน notebook ใหม่เพื่อเทรนต่อ)
# ดูความคืบหน้าได้ใน TensorBoard cell ถัดไป และ log ที่ `models/train.log`

# %%
from colab.ckpt import find_resumable_checkpoint
from colab.drive_trash import drive_service, folder_id, purge_trashed_checkpoints
from colab.runtime import ensure_pretrained_init, write_sbv2_config_yml
from colab.train_runner import run_training

TDS = DATASET_ROOT / TRAIN_NAME
MODELS = TDS / "models"
MODELS.mkdir(exist_ok=True)
resume = find_resumable_checkpoint(MODELS)
if resume is None:   # เริ่มใหม่: ต้องมีน้ำหนัก JP-Extra เสมอ ไม่งั้น SBV2 จะเทรนจากศูนย์แบบเงียบ ๆ
    ensure_pretrained_init(SBV2 / "pretrained_jp_extra", MODELS)
print("resume from step:", resume or "pretrained JP-Extra base")

free_gpu_memory("m0_model", "my_model")   # ให้ตัวเทรนได้ GPU เต็มที่

_on_prune = None
if PURGE_TRASHED_CHECKPOINTS:
    # ไฟล์ที่ลบผ่าน /content/drive จะไปอยู่ถังขยะและยังกินโควตา — ลบเฉพาะ checkpoint ของ run นี้ออกถาวร
    _svc = drive_service()   # จะมีหน้าต่างขอสิทธิ์ Google Drive ครั้งแรก
    def _rel(p):
        return list(Path(p).relative_to("/content/drive/MyDrive").parts)

    _ids = {folder_id(_svc, _rel(MODELS))}
    (ASSETS_ROOT / TRAIN_NAME).mkdir(parents=True, exist_ok=True)
    _ids.add(folder_id(_svc, _rel(ASSETS_ROOT / TRAIN_NAME)))
    def _on_prune():
        print("ลบ checkpoint เก่าในถังขยะถาวร:", purge_trashed_checkpoints(_svc, _ids), "ไฟล์")

cfg_path = TDS / "config.json"
bs = BATCH_SIZE
while True:
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    cfg["train"]["batch_size"] = bs
    cfg["train"]["bf16_run"] = False          # T4 ไม่รองรับ bf16
    cfg["train"]["eval_interval"] = SAVE_EVERY_STEPS
    cfg_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
    write_manifest(TDS / "manifest.json", run_id=TRAIN_NAME, experiment=EXPERIMENT, sbv2_tag=SBV2_TAG,
                   gpu=GPU, batch_size=bs, resume_step=resume, data_hash=data_hash(TDS / "wavs"), config=cfg["train"])
    write_sbv2_config_yml(SBV2, TRAIN_NAME, TDS, keep_ckpts=KEEP_CKPTS)   # trainer อ่าน model_name จากไฟล์นี้
    cmd = [sys.executable, "train_ms_jp_extra.py", "--config", str(cfg_path),
           "--model", str(TDS), "--assets_root", str(ASSETS_ROOT)]
    result = run_training(cmd, MODELS, budget_s=TRAIN_BUDGET_HOURS * 3600, poll_s=30,
                          assets_dir=ASSETS_ROOT / TRAIN_NAME, keep=KEEP_CKPTS, on_prune=_on_prune)
    print("result:", result)
    if result == "oom" and bs > 1:
        bs //= 2
        print("OOM → ลด batch size เป็น", bs)
        continue
    break
if result not in ("finished", "budget_stop"):
    print(open(MODELS / "train.log", encoding="utf-8", errors="ignore").read()[-3000:])
assert result in ("finished", "budget_stop"), f"การเทรนล้มเหลว ({result}) — ดู log ด้านบน แล้วส่งมาให้ผู้พัฒนาได้"

# %% [markdown]
# ### 3.1 TensorBoard
# **ทำอะไร:** แสดงกราฟ loss ของการเทรน · **ควรเห็น:** loss ลดลงเรื่อย ๆ (เปิดดูระหว่างหรือหลังเทรน)

# %%
# %load_ext tensorboard
# %tensorboard --logdir "{MODELS}"

# %% [markdown]
# ## 4. อ่านภาษาไทยด้วยโมเดลที่เทรนเอง
# **ทำอะไร:** พิมพ์ข้อความไทย เลือกระดับสำเนียง / ตาราง tone / style แล้วฟัง ·
# **ควรเห็น:** kana, phone, pitch, กฎที่ถูกใช้ และเสียง (บันทึกลง `Laboratory/outputs/`)
#
# - `BERT_MODE` ต้องตรงกับการทดลองที่เทรน: E1 → `"kana"`, E2 → `"zeros"`
# - `INTONATION < 1` = ช่วงเสียงแคบลงแบบผู้เรียน (กฎ 29) · `LENGTH > 1` = พูดช้าลง

# %%
TEXT = "สวัสดี ครับ วันนี้ ไป เที่ยว ภูเก็ต"  # @param {type:"string"}
LEVEL = "light"        # @param ["light", "medium", "strong"]
TONE_MAP = "default"   # @param ["default", "learner"]
BERT_MODE = "kana" if EXPERIMENT == "E1" else "zeros"
STYLE = "Neutral"
STYLE_WEIGHT = 1.0     # @param {type:"slider", min:0, max:5, step:0.1}
LENGTH = 1.0           # @param {type:"slider", min:0.6, max:1.6, step:0.05}
INTONATION = 0.9       # @param {type:"slider", min:0.5, max:1.5, step:0.05}

my_model = load_tts(ASSETS_ROOT / TRAIN_NAME)
inp, info = FE.convert(TEXT, AccentLevel[LEVEL.upper()], TONE_MAP)
print("kana  :", inp.kana)
print("phones:", " ".join(inp.phones))
print("pitch :", "".join(map(str, inp.tones)))
print("rules :", sorted({r for r, _ in info["rule_log"]}))
for w in info["warnings"]:
    print("⚠️", w)
sr, audio = synthesize(my_model, inp, BERT_MODE, style=STYLE, style_weight=STYLE_WEIGHT,
                       length=LENGTH, intonation_scale=INTONATION)
path = OUT / f"{TRAIN_NAME}_{LEVEL}_{TONE_MAP}_{int(time.time())}.wav"
sf.write(path, audio, sr)
print("saved:", path)
display(Audio(str(path)))

# %% [markdown]
# ## 5. วัดผล
# **ทำอะไร:** สังเคราะห์ 30 ประโยค × 3 ระดับ × 2 ตาราง tone แล้ววัด CER (Whisper) และ UTMOS
# จากนั้นสร้างแบบฟอร์มฟังแบบปิดเงื่อนไข (`listening_sheet.csv`) ให้คุณและคนไทยกรอกคะแนน
#
# **เกณฑ์จบ (M4):** ระดับ light มี CER ≤ 25% และคะแนนฟังรู้เรื่องเฉลี่ย ≥ 3.5/5 — หรือครบ 3 รอบปรับ-เทรนซ้ำ
#
# **ควรเห็น:** ตารางผลลัพธ์ และไฟล์ใน `Laboratory/outputs/eval_<RUN>/`

# %%
import csv

from thaijp_accent.eval import make_listening_sheet, summarize, utmos

if not RUN_EVAL:
    print("ข้ามส่วน 5 (RUN_EVAL = False) — ตั้งเป็น True ในส่วนตั้งค่าเมื่อเทรนเสร็จแล้ว")
else:
    if "my_model" not in globals():
        my_model = load_tts(ASSETS_ROOT / TRAIN_NAME)
    _ckpt = sorted((ASSETS_ROOT / TRAIN_NAME).rglob("*.safetensors"), key=lambda p: p.stat().st_mtime)[-1].stem
    EV = OUT / f"eval_{TRAIN_NAME}_{_ckpt}_{INTONATION}"     # โฟลเดอร์ต่อ checkpoint — ไม่วัดเสียงเก่าซ้ำ
    EV.mkdir(exist_ok=True)
    rows = []
    for level in ("light", "medium", "strong"):
        for tm in ("default", "learner"):
            for s in SENTS:
                clip = f"{s['id']}_{level}_{tm}"
                p = EV / f"{clip}.wav"
                if not p.exists():
                    inp, _ = FE.convert(s["text"], AccentLevel[level.upper()], tm)
                    sr, audio = synthesize(my_model, inp, BERT_MODE, intonation_scale=INTONATION)
                    sf.write(p, audio, sr)
                rows.append({"clip_id": clip, "file": p.name, "text": s["text"], "level": level,
                             "tone_map": tm, "bert_mode": BERT_MODE})
    if not (EV / "results.csv").exists():
        hyps = transcribe([EV / r["file"] for r in rows])
        mos = utmos([EV / r["file"] for r in rows])
        for r, h, m in zip(rows, hyps, mos):
            r.update(asr=h, cer=thai_cer(r["text"], h), utmos=m, intelligibility="")
        with open(EV / "results.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # แบบฟอร์มฟังแบบปิดเงื่อนไข: ไฟล์เสียงชื่อ c001.wav… ในโฟลเดอร์ listening/ (ไม่บอกระดับ/ตาราง)
    if not (EV / "listening_sheet.csv").exists():   # ไม่เขียนทับ — อาจกรอกคะแนนไว้แล้ว
        make_listening_sheet(rows, EV / "listening_sheet.csv")
        (EV / "listening").mkdir(exist_ok=True)
        for k in csv.DictReader(open(EV / "listening_sheet_key.csv", encoding="utf-8")):
            shutil.copy2(EV / k["file"], EV / "listening" / f"{k['anon_id']}.wav")
    for r in summarize(EV / "results.csv"):
        print(r)
    print("📝 ฟังไฟล์ใน", EV / "listening", "แล้วกรอก listening_sheet.csv จากนั้นรัน cell ถัดไป")

# %% [markdown]
# ### 5.1 รวมคะแนนฟังจริง
# **ทำอะไร:** อ่าน `listening_sheet.csv` ที่กรอกแล้ว รวมเข้ากับผลอัตโนมัติ และบอกว่า M4 ผ่านหรือยัง

# %%
if RUN_EVAL:
    key = {k["anon_id"]: k["clip_id"] for k in csv.DictReader(open(EV / "listening_sheet_key.csv", encoding="utf-8"))}
    scores = {key[r["clip_id"]]: r for r in csv.DictReader(open(EV / "listening_sheet.csv", encoding="utf-8"))}
    rows = list(csv.DictReader(open(EV / "results.csv", encoding="utf-8")))
    for r in rows:
        r["intelligibility"] = scores.get(r["clip_id"], {}).get("intelligibility_1to5", "")
    with open(EV / "results_scored.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    summary = summarize(EV / "results_scored.csv")
    for r in summary:
        print(r)
    print("M4:", "✅ ผ่าน" if any(r["m4_pass"] for r in summary) else "ยังไม่ผ่าน — ปรับกฎ/tone map แล้วเทรนซ้ำ (สูงสุด 3 รอบ)")
