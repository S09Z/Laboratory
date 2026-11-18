# ---
# jupyter:
#   kernelspec:
#     display_name: thaijp (Mac M3)
#     language: python
#     name: thaijp-mac
# ---

# %% [markdown]
# # 🎙️ Thai → สำเนียงญี่ปุ่น · เวอร์ชัน Mac (Apple Silicon, RAM 16 GB)
#
# ใช้บน Mac สำหรับ **ทุกอย่างยกเว้นการเทรน** (การเทรนทำใน Colab — ตัวเทรนของ SBV2 ใช้ได้เฉพาะ NVIDIA CUDA)
#
# | ส่วน | ทำอะไร | ใช้เวลา |
# |---|---|---|
# | 0 | ตรวจเครื่อง, หาโฟลเดอร์ `Laboratory` จาก Google Drive for desktop | ทันที |
# | 1 | ตรวจ G2P ภาษาไทย | ~1 นาที (ครั้งแรกดาวน์โหลดโมเดล) |
# | 2 | M0: ลองกับโมเดลつくよみちゃん (ไม่ต้องเทรน) | ~2 นาที |
# | 3 | พิมพ์ภาษาไทยแล้วฟัง ด้วยโมเดลที่เทรนใน Colab | ไม่กี่วินาที |
# | 4 | ฟังเทียบ 3 ระดับสำเนียง × 2 ตาราง tone | ~1 นาที |
# | 5 | วัดผล (Whisper + UTMOS) | ~20–40 นาที |
#
# **ก่อนเริ่ม:** รัน `bash mac/setup_mac.sh` หนึ่งครั้ง แล้วเลือก kernel **thaijp (Mac M3)**

# %% [markdown]
# ## ⚙️ ตั้งค่า
# **ทำอะไร:** ชื่อโมเดลที่เทรนใน Colab และตำแหน่งไฟล์ · ปกติไม่ต้องแก้ ถ้าใช้ชื่อเดียวกับ notebook Colab

# %%
TRAIN_NAME = "amitaro_runrun"   # ชื่อเดียวกับ RUN_ID (หรือ RUN_ID + "_E2") ใน notebook Colab
BERT_MODE = "kana"              # E1 → "kana" · E2 → "zeros"
LAB_OVERRIDE = ""               # เว้นว่าง = ใช้ workspace (โฟลเดอร์แม่ของ thaijp_tts ถ้ามี sbv2/) · ไม่งั้นหา MyDrive/Laboratory · หรือใส่ path เอง
RUN_EVAL = False                # True = รันส่วน 5

# %% [markdown]
# ## 0. ตรวจเครื่องและหาไฟล์
# **ทำอะไร:** ตั้ง path ของโปรเจกต์ / SBV2 / โฟลเดอร์ `Laboratory` และเลือก device (MPS ถ้าใช้ได้ ไม่งั้น CPU)
#
# **ควรเห็น:** path ทั้ง 3 และ `✅ ready` · ถ้าหา `Laboratory` ไม่เจอ: ติดตั้ง Google Drive for desktop
# แล้วตั้งให้โฟลเดอร์ `Laboratory` เป็น **Available offline** (คลิกขวา → Offline access) หรือใส่ `LAB_OVERRIDE`

# %%
import os
import sys
import time
from pathlib import Path

assert "thaijp_lab" in sys.executable or os.environ.get("THAIJP_HOME", "") in sys.executable, (
    f"kernel ผิดตัว: {sys.executable}\nเลือก kernel \"thaijp (Mac M3)\" (thaijp_lab/.venv) แล้ว Restart ก่อน")
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")   # op ที่ MPS ยังไม่รองรับ → ไปทำบน CPU แทนการ error

PROJ = Path.cwd().resolve().parent if Path.cwd().name == "notebooks" else Path.cwd().resolve()
assert (PROJ / "thaijp_accent").is_dir(), f"เปิด notebook จากโฟลเดอร์ thaijp_tts/notebooks (ตอนนี้อยู่ที่ {Path.cwd()})"
SBV2 = Path(os.environ.get("THAIJP_HOME", PROJ.parent / "thaijp_lab")) / "Style-Bert-VITS2"
assert SBV2.is_dir(), f"ไม่พบ {SBV2} — รัน: bash mac/setup_mac.sh"
for p in (str(PROJ), str(SBV2)):
    if p not in sys.path:
        sys.path.insert(0, p)
for _m in [m for m in sys.modules if m.startswith("thaijp_accent")]:   # ใช้โค้ดล่าสุดเสมอ
    del sys.modules[_m]

from thaijp_accent.devices import find_drive_lab, pick_device

_ws = PROJ.parent if (PROJ.parent / "sbv2").is_dir() else None
LAB = Path(LAB_OVERRIDE).expanduser() if LAB_OVERRIDE else (_ws or find_drive_lab())
assert LAB and LAB.is_dir(), "หาโฟลเดอร์ Laboratory ใน Google Drive for desktop ไม่เจอ — ใส่ LAB_OVERRIDE"
ASSETS_ROOT = LAB / "sbv2" / "model_assets"
OUT = LAB / "outputs_mac"
OUT.mkdir(parents=True, exist_ok=True)
os.chdir(SBV2)   # SBV2 อ่าน configs/ และ bert/ จาก working directory
print("PROJ :", PROJ)
print("SBV2 :", SBV2)
print("LAB  :", LAB)
print("✅ ready")

# %% [markdown]
# ## 1. ตรวจ G2P ภาษาไทย
# **ทำอะไร:** แปลงคำตัวอย่างเป็น IPA ด้วย `umt5-thai-g2p` แล้วแยกพยางค์ · **ควรเห็น:** ✅ ทุกคำ
# คำที่อ่านผิดให้เพิ่มใน `Laboratory/lexicon.tsv` (ไฟล์เดียวกับที่ Colab ใช้ — sync ผ่าน Drive)

# %%
import shutil

from thaijp_accent.g2p import UmT5G2P, parse_ipa
from thaijp_accent.pipeline import ThaiJPFrontend, load_sentences
from thaijp_accent.types import AccentLevel

G2P = UmT5G2P(device="cpu")   # โมเดลเล็ก — CPU เร็วพอและเลี่ยงปัญหา MPS
LEX = LAB / "lexicon.tsv"
if not LEX.exists():
    shutil.copy2(PROJ / "thaijp_accent" / "lexicon.tsv", LEX)
bad = 0
for w in ["ปลา", "กรุงเทพ", "ภูเก็ต", "เชียงใหม่", "ข้าว", "ส้มตำ", "เลือก", "หมา", "ม้า", "สวัสดี"]:
    ipa = G2P.word_to_ipa(w)
    try:
        s = parse_ipa(ipa, 0)
        print("✅", w, ipa, [(x.onset, x.cluster, x.vowel, x.long, x.coda, x.tone) for x in s])
    except Exception as e:
        bad += 1
        print("❌", w, repr(ipa), e)
FE = ThaiJPFrontend(G2P, lexicon_path=LEX)
SENTS = load_sentences(PROJ / "tests" / "data" / "test_sentences.tsv")
assert bad <= 2, f"parser อ่าน IPA ไม่ได้ {bad}/10 คำ"

# %% [markdown]
# ### 1.1 โหลดโมเดลเสียง + เลือก device
# **ทำอะไร:** ฟังก์ชันโหลดโมเดล SBV2 และทดลองสังเคราะห์สั้น ๆ บน MPS — ถ้า MPS ทำไม่ได้จะใช้ CPU อัตโนมัติ
# (M3 ใช้ CPU ก็สังเคราะห์ได้ประโยคละ ~1–2 วินาที)

# %%
import gc
import re

import numpy as np
import soundfile as sf
import torch
from huggingface_hub import snapshot_download
from IPython.display import Audio, display
from style_bert_vits2.tts_model import TTSModel

from thaijp_accent.synth import synthesize

_DEVICE_CACHE = {}

def list_ckpts(model_dir: Path) -> list[Path]:
    """checkpoint สำหรับ inference เท่านั้น (ไม่รวม G_/D_/WD_/DUR_ ที่เป็นของ trainer) เรียงตามเวลา"""
    skip = re.compile(r"^(G|D|WD|DUR)_\d+")
    return sorted((p for p in model_dir.rglob("*.safetensors") if not skip.match(p.name)),
                  key=lambda p: p.stat().st_mtime)

def load_tts(model_dir: Path) -> TTSModel:
    sts = list_ckpts(model_dir)
    assert sts, f"ยังไม่มีโมเดล inference (*.safetensors) ใน {model_dir} — เทรนใน Colab ก่อน แล้วรอ Drive sync"
    cfg = next(model_dir.rglob("config.json"))
    sv = next(model_dir.rglob("style_vectors.npy"), None)
    if sv is None:   # โมเดลสาธารณะบางตัว (เช่น M0) ไม่แนบ style_vectors.npy → ใช้เวกเตอร์ neutral ตัวเดียว
        sv = model_dir / "style_vectors.npy"
        np.save(sv, np.zeros((1, 256), dtype=np.float32))
        print("⚠️ ไม่พบ style_vectors.npy — สร้าง neutral (zeros) ที่", sv)
    probe_inp, _ = FE.convert("สวัสดี", AccentLevel.LIGHT)

    def probe(device):
        m = TTSModel(model_path=sts[-1], config_path=cfg, style_vec_path=sv, device=device)
        synthesize(m, probe_inp, BERT_MODE)
        _DEVICE_CACHE[device] = m

    key = str(sts[-1])
    if _DEVICE_CACHE.get("key") != key:
        _DEVICE_CACHE.clear()
        _DEVICE_CACHE["key"] = key
        dev = pick_device(probe=probe)
        if dev not in _DEVICE_CACHE:
            probe(dev)
        _DEVICE_CACHE["device"] = dev
    dev = _DEVICE_CACHE["device"]
    print("model:", sts[-1].name, "| device:", dev)
    return _DEVICE_CACHE[dev]

def free_memory():
    _DEVICE_CACHE.clear()
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def say(model, text, level="light", tone_map="default", **kw):
    inp, info = FE.convert(text, AccentLevel[level.upper()], tone_map)
    sr, audio = synthesize(model, inp, BERT_MODE, **kw)
    path = OUT / f"{TRAIN_NAME}_{level}_{tone_map}_{int(time.time() * 1000)}.wav"
    sf.write(path, audio, sr)
    return path, inp, info

# %% [markdown]
# ## 2. M0 — ลองกับโมเดลつくよみちゃん (ไม่ต้องเทรน)
# **ทำอะไร:** อ่าน 5 ประโยคทดสอบด้วยโมเดลที่มีคนเทรนไว้แล้ว · ใช้ไฟล์ใน `Laboratory/m0_tsukuyomi` ถ้ามี (จาก Colab)
#
# เครดิต: フリー素材キャラクター「つくよみちゃん」(© Rei Yumesaki)

# %%
M0_DIR = LAB / "m0_tsukuyomi"
if not any(M0_DIR.rglob("*.safetensors")):
    snapshot_download("ayousanz/tsukuyomi-chan-style-bert-vits2-model", local_dir=M0_DIR)
_saved_mode, BERT_MODE = BERT_MODE, "kana"
m0_model = load_tts(M0_DIR)
for s in [s for s in SENTS if "m0" in s["tags"]]:
    path, inp, _ = say(m0_model, s["text"])
    print(s["id"], s["text"], "→", inp.kana)
    display(Audio(str(path)))
BERT_MODE = _saved_mode
del m0_model
free_memory()

# %% [markdown]
# ## 3. พิมพ์ภาษาไทยแล้วฟัง (โมเดลที่เทรนใน Colab)
# **ทำอะไร:** โหลด checkpoint ล่าสุดของ `TRAIN_NAME` จาก Drive แล้วอ่านข้อความ ·
# **ควรเห็น:** kana, phone, pitch, กฎที่ใช้ และเสียง (บันทึกใน `Laboratory/outputs_mac/`)
#
# แก้ `lexicon.tsv` หรือโค้ดกฎแล้ว รัน cell ส่วน 1 ใหม่ จากนั้นรัน cell นี้ — ฟังผลได้ทันทีโดยไม่ต้องเทรนใหม่

# %%
TEXT = "สวัสดี ครับ วันนี้ ไป เที่ยว ภูเก็ต"
LEVEL = "light"        # "light" | "medium" | "strong"
TONE_MAP = "default"   # "default" | "learner"
STYLE_WEIGHT = 1.0
LENGTH = 1.0
INTONATION = 0.9

my_model = load_tts(ASSETS_ROOT / TRAIN_NAME)
path, inp, info = say(my_model, TEXT, LEVEL, TONE_MAP, style_weight=STYLE_WEIGHT, length=LENGTH,
                      intonation_scale=INTONATION)
print("kana  :", inp.kana)
print("phones:", " ".join(inp.phones))
print("pitch :", "".join(map(str, inp.tones)))
print("rules :", sorted({r for r, _ in info["rule_log"]}))
for w in info["warnings"]:
    print("⚠️", w)
display(Audio(str(path)))

# %% [markdown]
# ## 4. ฟังเทียบระดับสำเนียง × ตาราง tone
# **ทำอะไร:** ประโยคเดียว 6 แบบ เรียงให้ฟังต่อกัน — ใช้เลือกค่าที่ชอบก่อนวัดผลจริง

# %%
COMPARE_TEXT = "ฉัน ชอบ กิน ผัดไทย"
for level in ("light", "medium", "strong"):
    for tm in ("default", "learner"):
        path, inp, _ = say(my_model, COMPARE_TEXT, level, tm, intonation_scale=INTONATION)
        print(f"{level:6s} · {tm:7s} · {inp.kana}")
        display(Audio(str(path)))

# %% [markdown]
# ## 5. วัดผล (ตั้ง `RUN_EVAL = True`)
# **ทำอะไร:** 30 ประโยค × 3 ระดับ × 2 ตาราง → CER (Whisper large-v3 บน MPS/CPU) และ UTMOS
# พร้อมแบบฟอร์มฟังแบบปิดเงื่อนไข (ไฟล์ `c001.wav…` ในโฟลเดอร์ `listening/`)
#
# RAM 16 GB: โมเดลเสียงถูกปล่อยก่อนโหลด Whisper เพื่อไม่ให้หน่วยความจำเต็ม

# %%
import csv

from thaijp_accent.eval import make_listening_sheet, summarize, thai_cer, transcribe, utmos

if not RUN_EVAL:
    print("ข้ามส่วน 5 (RUN_EVAL = False)")
else:
    _ckpt = list_ckpts(ASSETS_ROOT / TRAIN_NAME)[-1].stem
    EV = OUT / f"eval_{TRAIN_NAME}_{_ckpt}_{INTONATION}"
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
        del my_model
        free_memory()
        asr_device = "mps" if torch.backends.mps.is_available() else "cpu"
        hyps = transcribe([EV / r["file"] for r in rows], device=asr_device)
        mos = utmos([EV / r["file"] for r in rows])
        for r, h, m in zip(rows, hyps, mos):
            r.update(asr=h, cer=thai_cer(r["text"], h), utmos=m, intelligibility="")
        with open(EV / "results.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    if not (EV / "listening_sheet.csv").exists():
        make_listening_sheet(rows, EV / "listening_sheet.csv")
        (EV / "listening").mkdir(exist_ok=True)
        for k in csv.DictReader(open(EV / "listening_sheet_key.csv", encoding="utf-8")):
            shutil.copy2(EV / k["file"], EV / "listening" / f"{k['anon_id']}.wav")
    for r in summarize(EV / "results.csv"):
        print(r)
    print("📝 ฟังไฟล์ใน", EV / "listening", "แล้วกรอก listening_sheet.csv · รวมคะแนนได้ใน notebook Colab ส่วน 5.1")
