"""Derive the Kaggle notebook source from the Colab AllInOne source.

Single source of truth: edit notebooks_src/ThaiJP_TTS_AllInOne.py, then `make kaggle`.
Every replacement must match exactly once — if the Colab source drifts, this fails loudly.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "notebooks_src" / "ThaiJP_TTS_AllInOne.py"
DST = ROOT / "notebooks_src" / "ThaiJP_TTS_Kaggle.py"

HEADER_OLD = """# jupyter:
#   accelerator: GPU
#   colab:
#     gpuType: T4
#     provenance: []
#   kernelspec:"""
HEADER_NEW = """# jupyter:
#   kernelspec:"""

INTRO_OLD = """# **ก่อนเริ่ม**
# 1. `Runtime > Change runtime type > T4 GPU`
# 2. ใน Google Drive ต้องมีโฟลเดอร์ `Laboratory/` ที่มี `thaijp_tts.zip` (ไฟล์โปรเจกต์)
# 3. ทุก cell **รันซ้ำได้** — ถ้า session หลุดหรือ GPU ถูกตัด ให้กด `Runtime > Run all` ใหม่
#    งานที่ทำเสร็จแล้วจะถูกข้าม และการเทรนจะต่อจาก checkpoint ล่าสุดใน Drive เอง
"""
INTRO_NEW = """# **เวอร์ชัน Kaggle** (สร้างจาก notebook Colab ด้วย `make kaggle` — อย่าแก้ไฟล์นี้ตรง ๆ)
#
# **ก่อนเริ่ม**
# 1. Settings > **Accelerator = GPU T4 x2** (แนะนำ — P100 ใหม่ ๆ มักใช้กับ PyTorch ล่าสุดไม่ได้; เทรนใช้ 1 GPU) และ **Internet = On**
#    (ต้องยืนยันเบอร์โทรในบัญชี Kaggle ก่อน ถึงจะเปิด Internet/GPU ได้)
# 2. อัปโหลด `thaijp_tts.zip` เป็น **Dataset** ของคุณ (Add Data > Upload) แล้ว attach เข้า notebook นี้
# 3. เทรนใช้ได้สูงสุด ~12 ชม./session และโควตา GPU ฟรี ~30 ชม./สัปดาห์ · ไฟล์ใน `/kaggle/working` **หายเมื่อ session จบ**
#    ถ้าไม่กด **Save Version** — เมื่อเทรนเสร็จหรือหยุดตามงบเวลา ให้กด Save Version (Save & Run All) หรือดาวน์โหลด
#    `Laboratory/sbv2/model_assets/<RUN_ID>/` เก็บไว้ · จะเทรนต่อรอบถัดไป ให้ attach output ของรอบก่อนเป็น Input (cell ติดตั้งจะคัดลอกกลับให้เอง)
# 4. **กันงานหาย (แนะนำ):** สร้าง token แบบ *write* ที่ huggingface.co/settings/tokens แล้วใส่ใน **Add-ons > Secrets** ชื่อ `HF_TOKEN`
#    (เปิดสวิตช์ให้ notebook นี้ใช้) → checkpoint ทุกชุดจะถูกสำรองขึ้น repo private `<ชื่อ HF>/thaijp-tts-ckpt` และ session ใหม่จะกู้กลับมาเทรนต่อเอง
"""

REPL: list[tuple[str, str]] = [
    (HEADER_OLD, HEADER_NEW),
    (INTRO_OLD, INTRO_NEW),
    ("TRAIN_BUDGET_HOURS = 3.5       # Colab ฟรีมักถูกตัดราว 4–12 ชม. — หยุดอย่างปลอดภัยก่อน",
     "TRAIN_BUDGET_HOURS = 8.0       # Kaggle จำกัด ~12 ชม./session — เผื่อเวลาติดตั้ง/เตรียมข้อมูล/บันทึกผล\n"
     'HF_REPO = ""                   # repo Hugging Face (private) สำรอง checkpoint · ว่าง = "<ชื่อ HF ของคุณ>/thaijp-tts-ckpt" · ต้องมี HF_TOKEN ใน Secrets'),
    ("PURGE_TRASHED_CHECKPOINTS = True  # ลบ checkpoint เก่าที่ถูกย้ายไปถังขยะ Drive ออกถาวร (เฉพาะไฟล์ checkpoint ของ run นี้)\n", ""),
    ('LAB = "/content/drive/MyDrive/Laboratory"', 'LAB = "/kaggle/working/Laboratory"   # โฟลเดอร์ที่ถูกเก็บเป็น output (จำกัด 20 GB)'),
    ("# **ทำอะไร:** mount Drive, แตกไฟล์โปรเจกต์, ติดตั้ง", "# **ทำอะไร:** หา `thaijp_tts.zip` ใน Input, แตกไฟล์โปรเจกต์, ติดตั้ง"),
    ("from google.colab import drive\n\n", ""),
    ('assert shutil.which("nvidia-smi"), "No GPU: Runtime > Change runtime type > T4 GPU แล้วรันใหม่"',
     'assert shutil.which("nvidia-smi"), "No GPU: Settings > Accelerator > GPU T4 x2 แล้วรันใหม่"'),
    ('''drive.mount("/content/drive")
LAB = Path(LAB)
assert (LAB / "thaijp_tts.zip").exists(), f"ไม่พบ {LAB/'thaijp_tts.zip'} — อัปโหลดไฟล์โปรเจกต์ไว้ในโฟลเดอร์ Laboratory ก่อน"

PROJ = Path("/content/thaijp_tts")''',
     '''LAB = Path(LAB)
# Kaggle แตก .zip ให้เองตอนสร้าง Dataset → รองรับทั้งไฟล์ zip และโฟลเดอร์ที่แตกแล้ว
_zips = sorted(Path("/kaggle/input").glob("**/thaijp_tts.zip"))
_dirs = sorted(Path("/kaggle/input").glob("**/thaijp_accent/lexicon.tsv"))
assert _zips or _dirs, "ไม่พบโปรเจกต์ใน /kaggle/input — อัปโหลด thaijp_tts.zip เป็น Dataset แล้ว Add Data เข้า notebook นี้"
# รอบก่อนหน้า (attach output ของ notebook นี้เป็น Input) → คัดลอก Laboratory/ กลับมาเพื่อเทรนต่อ
if not (LAB / "sbv2").exists():
    for _prev in sorted(Path("/kaggle/input").glob("**/Laboratory/sbv2")):
        print("คัดลอกงานรอบก่อนจาก", _prev.parent)
        shutil.copytree(_prev.parent, LAB, dirs_exist_ok=True)
        break

PROJ = Path("/kaggle/working/thaijp_tts")'''),
    ("# แตก zip ใหม่ทุกครั้ง เพื่อให้ได้โค้ดล่าสุดที่อัปโหลดไว้ (ไฟล์ที่คุณแก้เอง เช่น lexicon อยู่บน Drive ไม่หาย)",
     "# แตก zip ใหม่ทุกครั้ง เพื่อให้ได้โค้ดล่าสุดที่อัปโหลดไว้ (ไฟล์ที่คุณแก้เอง เช่น lexicon อยู่ใน Laboratory/ ไม่หาย)"),
    ('with zipfile.ZipFile(LAB / "thaijp_tts.zip") as z:\n    z.extractall("/content")',
     'if _zips:\n    with zipfile.ZipFile(_zips[0]) as z:\n        z.extractall("/kaggle/working")\nelse:\n    shutil.copytree(_dirs[0].parent.parent, PROJ)'),
    ('SBV2 = Path("/content/Style-Bert-VITS2")\nMARK = Path("/content/.sbv2_installed")',
     'Path("/kaggle/temp").mkdir(exist_ok=True)   # ไม่ถูกนับเป็น output — ใส่ของใหญ่ที่สร้างใหม่ได้\n'
     'SBV2 = Path("/kaggle/temp/Style-Bert-VITS2")\nMARK = Path("/kaggle/temp/.sbv2_installed")'),
    ("    # Colab ปัจจุบันเป็น Python 3.13", "    # Python ของ runtime ปัจจุบันใหม่"),
    ('''    print("🔁 ติดตั้งเสร็จ กำลัง restart runtime — จากนั้นกด Runtime > Run all อีกครั้ง")
    os.kill(os.getpid(), 9)''',
     '''    if "numpy" in sys.modules or "torch" in sys.modules:   # kernel โหลด lib เก่าไปแล้ว → ต้อง restart (ใช้ไม่ได้ใน Save & Run All)
        print("🔁 ติดตั้งเสร็จ กำลัง restart kernel — จากนั้นกด Run All อีกครั้ง")
        os.kill(os.getpid(), 9)'''),
    ('    sh("apt-get -qq update && apt-get -qq install -y cmake build-essential", env)',
     '    sh("apt-get -qq update && apt-get -qq install -y cmake build-essential", env)   # ต้องเปิด Internet'),
    ('tmp = Path("/content") / (DATASET_ZIP + ".part")', 'tmp = Path("/kaggle/temp") / (DATASET_ZIP + ".part")'),
    ("from colab.runtime import ensure_pretrained_init, write_sbv2_config_yml\nfrom colab.train_runner import run_training\n",
     "from colab.hf_backup import make_backup\nfrom colab.runtime import ensure_pretrained_init, write_sbv2_config_yml\nfrom colab.train_runner import run_training\n"),
    ("resume = find_resumable_checkpoint(MODELS)\n",
     "BACKUP = make_backup(HF_REPO, MODELS, ASSETS_ROOT / TRAIN_NAME)   # สำรอง/กู้ checkpoint ผ่าน Hugging Face (ต้องมี HF_TOKEN)\n"
     "if BACKUP:\n    BACKUP.restore()   # session ใหม่ → ดึง checkpoint ล่าสุดกลับมาเทรนต่อ (ถ้ามีไฟล์ในเครื่องอยู่แล้วจะใช้ไฟล์นั้น)\n"
     "resume = find_resumable_checkpoint(MODELS)\n"),
    ("from colab.drive_trash import drive_service, folder_id, purge_trashed_checkpoints\n", ""),
    ('''_on_prune = None
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
''', '''_on_prune = BACKUP.push if BACKUP else None   # ทุกครั้งที่บันทึก checkpoint ใหม่ → สำรองขึ้น Hugging Face (Kaggle ไม่มีถังขยะ — ลบไฟล์แล้วคืนพื้นที่ทันที)
(ASSETS_ROOT / TRAIN_NAME).mkdir(parents=True, exist_ok=True)
'''),
    ("cfg[\"train\"][\"bf16_run\"] = False          # T4 ไม่รองรับ bf16", "cfg[\"train\"][\"bf16_run\"] = False          # T4/P100 ไม่รองรับ bf16"),
    ('soxr tensorboard "ml_dtypes>=0.5.1" --no-progress', 'soxr tensorboard "ml_dtypes>=0.5.1" "numpy<2.3" --no-progress'),   # numba (librosa) รองรับ NumPy ≤ 2.2
    ('if _missing:\n    print("ติดตั้งเพิ่ม:", _missing)',
     '\nimport numpy as _np\n\n_RESTART = False\nif tuple(int(x) for x in _np.__version__.split(".")[:2]) >= (2, 3):   # numba รองรับ NumPy ≤ 2.2\n'
     '    _missing.append(\'"numpy<2.3"\')\n    _RESTART = True\nif _missing:\n    print("ติดตั้งเพิ่ม:", _missing)'),
    ("    subprocess.run(_cmd, shell=True, check=True, env=_env)\n",
     "    subprocess.run(_cmd, shell=True, check=True, env=_env)\n    if _RESTART:   # NumPy ถูกโหลดไปแล้ว → restart kernel แล้วกด Run All อีกครั้ง\n        print(\"🔁 ปรับ NumPy แล้ว กำลัง restart kernel — กด Run All อีกครั้ง\")\n        os.kill(os.getpid(), 9)\n"),
    ("`Laboratory/lexicon.tsv` บน Drive", "`Laboratory/lexicon.tsv`"),
    ("ไฟล์นี้อยู่บน Drive จึงไม่หายเมื่อเปลี่ยนเครื่อง", "ไฟล์นี้อยู่ใน Laboratory/ (output) จึงต่อรอบถัดไปได้"),
    ("library ที่ต้องมี · พื้นที่ Drive ·", "library ที่ต้องมี · พื้นที่ดิสก์ ·"),
    ("อ่านไฟล์เสียงจาก Drive →", "อ่านไฟล์เสียงจาก Laboratory/datasets →"),
    ("ตรวจ checkpoint ล่าสุดใน Drive —", "ตรวจ checkpoint ล่าสุดใน Laboratory/ —"),
    ("เพื่อไม่ให้ Drive เต็ม)", "เพื่อไม่ให้ดิสก์เต็ม — output จำกัด 20 GB)"),
]


def build(text: str) -> str:
    for old, new in REPL:
        n = text.count(old)
        if n != 1:
            raise SystemExit(f"replacement matched {n}x (need 1): {old[:70]!r}")
        text = text.replace(old, new)
    return text


def main() -> int:
    DST.write_text(build(SRC.read_text(encoding="utf-8")), encoding="utf-8")
    print("wrote", DST)
    return 0


if __name__ == "__main__":
    sys.exit(main())
