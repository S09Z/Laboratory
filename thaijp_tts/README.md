# thaijp_tts — ภาษาไทยสำเนียงญี่ปุ่น · เสียงสาว VTuber

ข้อความไทย → ตัวแปลงตามกฎ 33 ข้อ (3 ระดับ) → phone + pitch ญี่ปุ่น → **Style-Bert-VITS2 JP-Extra** ที่ fine-tune เสียงสาวเอง

## วิธีรัน (Google Colab)

1. ใน Google Drive ต้องมี `MyDrive/Laboratory/thaijp_tts.zip` และ `ThaiJP_TTS_AllInOne.ipynb`
2. เปิด `ThaiJP_TTS_AllInOne.ipynb` ด้วย Colab → `Runtime > Change runtime type > T4 GPU`
3. `Runtime > Run all` — ครั้งแรก runtime จะ restart เองหนึ่งครั้งหลังติดตั้ง ให้กด `Run all` อีกครั้ง
4. ทำตามคำแนะนำในแต่ละส่วน:
   - **ส่วน 1 (M0):** ฟัง 5 ประโยค แล้วใส่ `M0_UNDERSTOOD` เกณฑ์ผ่านคือเข้าใจ ≥ 3/5 และ CER ≤ 40%
   - **ส่วน 2:** ดาวน์โหลดเสียง ITA corpus ของあみたろจาก https://amitaro.net/ ไปไว้ที่ `Laboratory/datasets/` แล้วตั้ง `I_ACCEPT_AMITARO_TERMS = True`
   - **ส่วน 3:** เทรน ถ้า session ถูกตัดหรือขึ้น `budget_stop` ให้กด `Run all` ใหม่ ระบบจะเทรนต่อจาก checkpoint ล่าสุดเอง
   - **ส่วน 4:** ลองพิมพ์ภาษาไทยแล้วฟังได้เลย
   - **ส่วน 5:** วัดผล กรอก `listening_sheet.csv` เกณฑ์จบ M4 คือ light CER ≤ 25% และคะแนนฟังรู้เรื่อง ≥ 3.5/5

ทุกอย่าง (ข้อมูล, checkpoint, เสียง) เก็บใน `Laboratory/` บน Drive ต้องมีที่ว่างอย่างน้อย 15 GB

## ใช้บน Mac (Apple Silicon M1/M2/M3, RAM 16 GB)

Mac ใช้ได้กับ**ทุกขั้นยกเว้นการเทรน** ได้แก่ ตรวจ G2P, M0, พิมพ์ไทยแล้วฟัง, ฟังเทียบระดับ และวัดผล
การเทรนยังต้องทำใน Colab เพราะตัวเทรนของ SBV2 ใช้ได้เฉพาะ CUDA

1. ติดตั้ง **Google Drive for desktop** แล้วคลิกขวาที่โฟลเดอร์ `Laboratory` → *Available offline*
2. แตก `Laboratory/thaijp_tts.zip` ไปไว้ที่ไหนก็ได้ เช่น `~/thaijp_tts`
3. ใน Terminal: `bash ~/thaijp_tts/mac/setup_mac.sh`
   - สร้าง Python 3.11 + Style-Bert-VITS2 ไว้ที่ `thaijp_lab/` (โฟลเดอร์ข้าง `thaijp_tts/` ใน workspace; เปลี่ยนได้ด้วย `THAIJP_HOME`)
   - ใช้เวลาครั้งแรก 10–15 นาที
   - ต้องมี Xcode Command Line Tools และ Homebrew
4. `source <workspace>/thaijp_lab/.venv/bin/activate && jupyter lab thaijp_tts/notebooks/ThaiJP_TTS_Mac.ipynb`
   แล้วเลือก kernel **thaijp (Mac M3)** (ใช้กับ VS Code ได้เหมือนกัน)

โมเดลที่เทรนใน Colab (`Laboratory/sbv2/model_assets/<TRAIN_NAME>`) จะ sync มาที่ Mac ให้อัตโนมัติ
notebook จะลองใช้ GPU ของ Mac (MPS) ก่อน ถ้าใช้ไม่ได้จะสลับไปใช้ CPU ให้เอง

## ทดลอง E1 / E2

- `EXPERIMENT = "E1"` (ค่าเริ่มต้น): BERT ปกติ · ตอนใช้งานส่ง kana
- `EXPERIMENT = "E2"`: เทรนด้วย BERT เป็นศูนย์ · ตอนใช้งานส่งศูนย์ (สร้างชุดข้อมูลแยก `<RUN_ID>_E2`)

## โครงสร้าง

| โฟลเดอร์ | หน้าที่ |
|---|---|
| `thaijp_accent/` | front-end (G2P, กฎ, mora, kana), ตัวเชื่อม SBV2, เตรียมข้อมูล, วัดผล |
| `colab/` | ตรวจ GPU, checkpoint ปลอดภัย, รันเทรนแบบมีงบเวลา, ลด batch เมื่อ OOM |
| `notebooks/` | notebook รวมทุกขั้น |
| `tests/` | `make test` (192 tests, ไม่ต้องใช้อินเทอร์เน็ต) |
| `docs/` | แบบระบบ, แผน, รายงานวิจัย |

ลองบนเครื่องโดยไม่ใช้ GPU:
`python -m thaijp_accent.pipeline "ไป เที่ยว ภูเก็ต" --level strong --fixtures tests/data/ipa_fixtures.tsv`

ถ้า G2P อ่านคำผิด ให้เพิ่มคำนั้นใน `Laboratory/lexicon.tsv` บน Drive (notebook สร้างให้ครั้งแรก) รูปแบบคือ `คำ<TAB>IPA` เช่น `ปลา	plaː0` (ตัวเลขท้ายคือวรรณยุกต์: 0 สามัญ, 1 เอก, 2 โท, 3 ตรี, 4 จัตวา)

## เครดิตและสัญญาอนุญาต

- **Style-Bert-VITS2:** AGPL-3.0 (https://github.com/litagin02/Style-Bert-VITS2)
- **เสียงฝึก:** あみたろの声素材工房 (https://amitaro.net/)
  - ต้องใส่เครดิตทุกครั้งที่เผยแพร่เสียง
  - ห้ามใช้กับงานจำกัดอายุ
  - ห้ามอ้างว่าเป็นเสียงของあみたろตัวจริง
- **M0:** フリー素材キャラクター「つくよみちゃん」(© Rei Yumesaki)
- **ITA corpus:** https://github.com/mmorise/ita-corpus
- **G2P:** `B-K/umt5-thai-g2p` (Apache-2.0)

## Kaggle (ทางเลือกเมื่อโควตา Colab หมด)

`make kaggle` สร้าง `notebooks/ThaiJP_TTS_Kaggle.ipynb` จาก source ของ Colab (`scripts/make_kaggle.py`) —
อัปโหลด `thaijp_tts.zip` เป็น Kaggle Dataset, เปิด GPU T4 x2 + Internet แล้ว Run All
(ไฟล์ใน `/kaggle/working` หายเมื่อ session จบถ้าไม่ Save Version)

**กันงานหาย (Kaggle):** ใส่ token แบบ *write* ของ Hugging Face ที่ Add-ons > Secrets ชื่อ `HF_TOKEN` (เปิดสวิตช์ให้ notebook) —
ทุก checkpoint ใหม่จะถูกสำรองขึ้น repo private `<ชื่อ HF>/thaijp-tts-ckpt` (เก็บ 2 ชุดล่าสุด, `colab/hf_backup.py`)
และ session ใหม่จะกู้กลับมาเทรนต่อเอง · ไม่มี token = เตือนแล้วเทรนต่อโดยไม่สำรอง (session หลุดงานหาย)
ตั้ง `HF_REPO` ใน cell ตั้งค่าได้ถ้าอยากใช้ชื่อ repo อื่น

## ดูความคืบหน้าและตรวจข้อมูล

- ระหว่างเทรน cell จะพิมพ์ทุก ~2 นาที เช่น `step 1566/10868 (14%) · epoch 14/100 · 1.42 s/it · ETA 3:40:47 · g/total 37.38 d/total 2.20 g/mel 22.18 ...`
  (step/ETA อ่านจากแถบ tqdm ใน `train.log`, loss อ่านจากไฟล์ TensorBoard) · ค่า `d/total` ที่ลู่เข้าศูนย์ = discriminator ชนะ เสี่ยงเสียงมีอาร์ติแฟกต์
- ตรวจคุณภาพไฟล์เสียงก่อนเทรน: `python scripts/audio_audit.py <โฟลเดอร์ wav> --csv audit.csv`
  (ความดัง, SNR, แบนด์วิดท์, DC offset, ความเงียบหัวท้าย) — ชุด あみたろ 424 ไฟล์ผ่านหมด: ความดังต่างกัน ±1–3 dB, SNR 48–63 dB, 44.1 kHz ครบ

## ข้อเสนอแนะเพื่อพัฒนาต่อ (จากการรีวิว — ยังไม่ได้วัดผลจริง)

ลำดับตามผลที่คาดเทียบกับแรงที่ใช้ · ข้อที่ทำแล้วมีเครื่องหมาย ✅ · ข้ออื่นต้องทดลองเทียบก่อนเชื่อ

| # | ข้อเสนอ | สถานะ |
|---|---|---|
| 1 | แสดง step/loss ระหว่างเทรน | ✅ |
| 2 | สำรอง checkpoint ออกนอก Kaggle อัตโนมัติ (Hugging Face) | ✅ (ทดสอบด้วย API จำลอง — ยังไม่ยืนยันอัปโหลดจริง) |
| 3 | ตรวจสเปกตรัม/ความดังรายไฟล์ | ✅ |
| 4 | **ฟังเสียงตัวอย่างทุก ~1000 step** — loss ของ GAN ไม่บอกคุณภาพเสียงตรง ๆ | ยังไม่ทำ |
| 5 | **เลือก checkpoint ที่ดีที่สุดจากการประเมิน** ไม่ใช่อันสุดท้าย (ข้อมูล ~424 ไฟล์ overfit ได้ง่าย) + early stopping | ยังไม่ทำ |
| 6 | ลอง **fp16** (T4 ไม่รองรับ bf16): เร็วขึ้นราว 1.5–2 เท่าและเพิ่ม batch ได้ แต่ VITS อาจไม่เสถียร (loss NaN) — ทดลองสั้น ๆ เทียบ fp32 ก่อน | ยังไม่ทำ |
| 7 | **freeze** บางส่วน (BERT/encoder) เพื่อลด overfit กับข้อมูลน้อย — ตอนนี้ `freeze_*` เป็น False หมด | ยังไม่ทำ |
| 8 | เฉลี่ยน้ำหนัก (EMA / 3–5 checkpoint สุดท้าย) — ต้นทุนต่ำ | ยังไม่ทำ |
| 9 | ใช้ GPU ตัวที่สอง (ตอนนี้ว่าง) — รัน eval/Whisper ขนานกับเทรน | ยังไม่ทำ |
| 10 | กรองคู่เสียง–ข้อความด้วย ASR (Whisper) นอกเหนือจาก QC เดิม | ยังไม่ทำ |
| 11 | ตัวชี้วัดเพิ่ม: speaker similarity, ความเสถียรของ F0 · ชุดทดสอบแยกจากชุดที่ใช้ปรับกฎ | ยังไม่ทำ |
| 12 | ใช้ style vector จากชุดอารมณ์ (EMOTION) ให้ได้สไตล์ VTuber โดยไม่ต้องเทรนเพิ่ม (ตอนนี้ใช้แค่ `Neutral`) | ยังไม่ทำ |
| 13 | ลดเวลาติดตั้ง ~10–15 นาที (แพ็ก wheel เป็น Dataset) + lock เวอร์ชันแพ็กเกจ | ยังไม่ทำ |

**ข้อจำกัดเชิงสถาปัตยกรรมที่ต้องตัดสินใจก่อนลงทุนเทรน E2:** tone ของ SBV2 เป็นแบบไบนารี (สูง/ต่ำ) ต่อโฟนีม แต่วรรณยุกต์ไทยเป็นเส้นโค้ง F0
(ขึ้น/ลง/ขึ้นแล้วลง) — การแมปเป็นลำดับ H/L สูญเสียรูปทรงของเส้นโค้ง เป็นเพดานคุณภาพของแนวทางนี้ · `intonation_scale` ปรับได้แค่ความกว้างของ F0
ไม่ได้เพิ่มรูปทรง · ควรประเมินกับผู้ฟังชาวไทยหลายคน ไม่พึ่ง UTMOS (เทรนกับเสียงอังกฤษ) หรือ CER อย่างเดียว
