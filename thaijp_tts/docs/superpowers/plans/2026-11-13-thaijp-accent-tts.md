# Thai→Japanese-accented VTuber TTS Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** แปลงข้อความไทยเป็น phone/tone ญี่ปุ่นที่มีสำเนียง (3 ระดับ) แล้วสังเคราะห์ด้วย Style-Bert-VITS2 JP-Extra ที่ fine-tune เสียงสาวบน Colab พร้อม notebook ที่ทนต่อการตัด session

**Architecture:** แพ็กเกจ Python `thaijp_accent` (CPU, มี unit test) ทำ front-end → rule engine → mora mapper → `Sbv2Input` ส่วน `colab/` เป็น helper สำหรับ Drive/GPU/resume และ notebook 5 ไฟล์ (เขียนเป็น jupytext `.py` แล้ว build เป็น `.ipynb`) เรียกใช้แพ็กเกจกับ Style-Bert-VITS2 ที่ pin เวอร์ชัน

**Tech Stack:** Python 3.10, PyThaiNLP, transformers (`B-K/umt5-thai-g2p`), PyYAML, pytest, jupytext, jiwer, Style-Bert-VITS2 `2.7.0` (git tag), Google Colab + Drive

**Spec:** `docs/superpowers/specs/2026-11-13-thai-jp-accent-vtuber-tts-design.md` (+ ตารางกฎใน `docs/research_report.md`)

## Global Constraints

- Style-Bert-VITS2 pin ที่ tag `2.7.0`; pretrained `litagin/Style-Bert-VITS2-2.0-base-JP-Extra`; script `train_ms_jp_extra.py`
- ระดับสำเนียง: `light ⊂ medium ⊂ strong`; ระดับของกฎ = คอลัมน์ "ระดับ" ในรายงาน ("กลาง–หนัก" นับเป็น medium) ไม่ใช่ป้ายหลักฐาน
- ทุก phone ที่ส่งเข้าโมเดลต้องอยู่ในชุด symbol ญี่ปุ่นของ SBV2 2.7.0 (ห้ามเพิ่ม symbol ไทย)
- front-end ต้อง deterministic: input เดิม → output เดิม (ใช้ RNG seed จาก hash ของข้อความ)
- กฎ 33 ไม่อยู่ในขอบเขต; ไม่ override duration/F0 ราย phoneme
- ข้อมูลเสียงหลัก: あみたろの声素材工房 ITA corpus ผู้พูดคนเดียว ห้ามผสมผู้พูด; เก็บข้อความเครดิตใน `DATA/CREDITS.txt`
- รูทใน Drive: `MyDrive/thaijp_tts/<run_id>/`; checkpoint ทุก 500 step; เก็บล่าสุด 3 ชุด
- ออกแบบสำหรับ Colab ฟรี (T4 16 GB, ไม่มี bf16)
- เกณฑ์ M0: ผู้ฟังจับใจความ ≥ 3/5 ประโยค และ CER ≤ 40%; เกณฑ์จบ M4: light CER ≤ 25% และคะแนนฟังรู้เรื่อง ≥ 3.5/5 หรือครบ 3 รอบ

## Review Focus

1. ข้อความไทยที่มีตัวเลขหรือคำภาษาอังกฤษปน (`ราคา 50 บาท OK`) → ตัวเลขแปลงเป็นคำไทย, คำละตินถูกตัดทิ้งพร้อม warning ไม่ crash (Task 2)
2. ข้อความว่างหรือมีแต่วรรคตอน/ช่องว่าง → `ValueError("no Thai syllables")` ไม่ส่ง phone ว่างเข้าโมเดล (Task 5)
3. checkpoint ล่าสุดเสียเพราะ session ถูกตัดตอนกำลังเขียน → resume จากชุดก่อนหน้าที่โหลดได้ (Task 6)
4. Colab ไม่ได้ GPU → หยุดด้วยข้อความชัดเจน ไม่เทรนบน CPU (Task 6)
5. คำที่ G2P คืน IPA ที่ parse ไม่ได้ → ใช้ override lexicon ถ้ามี มิฉะนั้นข้ามคำนั้นพร้อม warning ระบุคำ (Task 2)

---

## File Structure

```
thaijp_tts/
  pyproject.toml
  thaijp_accent/
    types.py          # Syllable, Mora, Sbv2Input, AccentLevel
    g2p.py            # ThaiG2P protocol, UmT5G2P, normalize_text, parse_ipa, thai_to_syllables
    lexicon.tsv       # override: word \t ipa
    rules.yaml        # 33 กฎ: id, level, evidence, impl
    rules.py          # ฟังก์ชันกฎระดับพยางค์ + apply_rules
    mora.py           # syllables_to_morae, tone maps, pauses
    tone_maps.yaml    # default, learner
    sbv2_symbols.py   # ชุด symbol ญี่ปุ่นที่ vendor จาก SBV2 2.7.0
    kana.py           # mora → katakana
    sbv2_io.py        # to_sbv2_input
    pipeline.py       # ThaiJPFrontend
    data_prep.py      # QC + train/val lists + neutralize_bert
    eval.py           # CER, coverage, listening sheet
  colab/
    runtime.py        # ensure_gpu, pick_batch_size, paths, manifest
    ckpt.py           # find_resumable_checkpoint, prune_checkpoints
    train_runner.py   # run_training with time budget
  notebooks_src/      # 00_m0_smoke_test.py … 04_eval.py (jupytext percent)
  notebooks/          # built .ipynb
  tests/
    data/test_sentences.tsv  # 30 ประโยค
    data/ipa_fixtures.tsv    # คำ \t IPA ที่จับจากโมเดลจริง
```

---

### Task 1: Scaffold + core types

**Files:**
- Create: `pyproject.toml`, `thaijp_accent/__init__.py`, `thaijp_accent/types.py`, `tests/test_types.py`, `.gitignore`

**Interfaces:**
- Produces:
  - `class AccentLevel(IntEnum): LIGHT=1; MEDIUM=2; STRONG=3`
  - `@dataclass Syllable: onset: str; cluster: str | None; vowel: str; long: bool; coda: str | None; tone: int; word_idx: int; hints: dict[str, str] = field(default_factory=dict)` — tone 0=สามัญ 1=เอก 2=โท 3=ตรี 4=จัตวา; onset `""` = เสียงสระนำ (ʔ)
  - `@dataclass Mora: phones: list[str]; pitch: int; word_idx: int; syl_idx: int; is_pause: bool = False`
  - `@dataclass Sbv2Input: phones: list[str]; tones: list[int]; word2ph: list[int]; kana: str`

- [ ] **Step 1:** `git init` ใน `/home/claude/thaijp_tts`; สร้าง `pyproject.toml` (deps: pythainlp, pyyaml, jiwer, jupytext; extras `g2p = [transformers, torch, sentencepiece]`; pytest marker `network`)
- [ ] **Step 2: Write failing test** `test_accent_level_ordering`: `assert AccentLevel.LIGHT < AccentLevel.MEDIUM < AccentLevel.STRONG`; `test_syllable_hints_default_independent`: สอง Syllable ไม่แชร์ dict `hints`
- [ ] **Step 3:** Run `pytest tests/test_types.py -v` → FAIL (ImportError)
- [ ] **Step 4:** Implement types ตาม Interfaces
- [ ] **Step 5:** Run → PASS
- [ ] **Step 6:** Commit `feat: scaffold package and core types`

---

### Task 2: Thai front-end (normalize → G2P → Syllable)

**Files:**
- Create: `thaijp_accent/g2p.py`, `thaijp_accent/lexicon.tsv`, `tests/test_g2p.py`, `tests/data/ipa_fixtures.tsv`

**Interfaces:**
- Consumes: `Syllable`
- Produces:
  - `class ThaiG2P(Protocol): def word_to_ipa(self, word: str) -> str`
  - `class UmT5G2P(ThaiG2P)` — โหลด `B-K/umt5-thai-g2p` แบบ lazy
  - `class FixtureG2P(ThaiG2P)` — อ่าน `ipa_fixtures.tsv` (ใช้ใน test)
  - `normalize_text(text: str) -> tuple[str, list[str]]` → (ข้อความไทย, warnings)
  - `parse_ipa(ipa: str, word_idx: int) -> list[Syllable]`
  - `thai_to_syllables(text: str, g2p: ThaiG2P, lexicon: dict[str, str] | None = None) -> tuple[list[Syllable], list[str]]`

- [ ] **Step 1: จับรูปแบบ output จริง** — รัน `UmT5G2P().word_to_ipa(w)` (marker `network`; ถ้า sandbox เข้า HF ไม่ได้ให้รันใน Colab) กับคำ: ปลา กรุงเทพ ภูเก็ต ผัดไทย เชียงใหม่ ข้าว ส้มตำ งาน ฟ้า เลือก ม้า หมา ไก่ น้ำ แม่ เพื่อน แล้วบันทึกลง `ipa_fixtures.tsv`; จดรูปแบบการแบ่งพยางค์และการเขียนวรรณยุกต์ไว้ใน docstring ของ `parse_ipa`
- [ ] **Step 2: Write failing tests**
  - `test_parse_ipa_kla`: syllable ของ "ปลา" → `onset="p", cluster="l", vowel="a", long=True, coda=None, tone=0`
  - `test_parse_ipa_phuket`: "ภูเก็ต" → 2 พยางค์ `[(pʰ,u,long,None,0), (k,e,short,t,1)]`
  - `test_parse_ipa_diphthong`: "เชียงใหม่" พยางค์แรก `onset="tɕʰ", vowel="ia", coda="ŋ"`
  - `test_normalize_digits`: `normalize_text("ราคา 50 บาท")[0] == "ราคา ห้าสิบ บาท"`
  - `test_normalize_latin_dropped`: `normalize_text("ดี OK")` → ข้อความ `"ดี"` และ warnings มีคำว่า `"OK"`
  - `test_unparseable_word_skipped`: G2P ปลอมที่คืน `"???"` → คำนั้นถูกข้าม, warnings มีชื่อคำ, ไม่ raise
  - `test_lexicon_overrides_g2p`: lexicon `{"ปลา": <ipa ของ ป๋า>}` → tone=4
  - `test_word_idx_increments`: "ปลา กิน" → word_idx `[0, 1]`
- [ ] **Step 3:** Run `pytest tests/test_g2p.py -v -m "not network"` → FAIL
- [ ] **Step 4:** Implement — ตัดคำด้วย `pythainlp.tokenize.word_tokenize(engine="newmm")`; ตัวเลขด้วย `pythainlp.util.num_to_thaiword`; `parse_ipa` แยก onset/cluster/vowel/coda/tone ตามรูปแบบจาก Step 1 (cluster ∈ {r,l,w}; coda ∈ {p,t,k,m,n,ŋ,w,j}; ความยาวจาก `ː`)
- [ ] **Step 5:** Run → PASS
- [ ] **Step 6:** Commit `feat: Thai normalization, G2P adapter and IPA parser`

---

### Task 3: Accent rule engine

**Files:**
- Create: `thaijp_accent/rules.yaml`, `thaijp_accent/rules.py`, `tests/test_rules.py`

**Interfaces:**
- Consumes: `Syllable`, `AccentLevel`
- Produces:
  - `load_rules() -> list[RuleMeta]` (`RuleMeta(id:int, level:AccentLevel, evidence:str, impl:str)`; impl ∈ `syllable|mapper|tone_map|prosody|implicit|out_of_scope`)
  - `apply_rules(sylls: list[Syllable], level: AccentLevel, seed_text: str) -> tuple[list[Syllable], list[tuple[int, int]]]` → (พยางค์ใหม่, log `(rule_id, syllable_index)`); ไม่แก้ list เดิม
  - hints ที่ mapper อ่าน: `coda_mode` ∈ `q|epenthetic|N|mu`, `ia_mode` ∈ `split|e`, `er_vowel` ∈ `o|u`

กฎ `impl: syllable` (แก้ field): 1, 2, 3, 4, 5, 8, 18, 19, 25, 26, 27 · `impl: mapper` (ตั้ง hint): 6, 7, 15, 16, 21, 23 · `implicit` (ชุดเสียงญี่ปุ่นทำเอง): 9, 10, 11, 12, 13, 14, 17, 20, 22, 24, 32 · `tone_map`: 28, 30 · `prosody`: 29, 31 · `out_of_scope`: 33
กฎ "เป็นครั้งคราว" (8, 19) ใช้ `random.Random(sha256(seed_text))`, p = 0.3

- [ ] **Step 1: Write failing tests**
  - `test_yaml_covers_33_rules`: ids == `set(range(1, 34))` และทุก id ที่ `impl == "syllable"|"mapper"` มีฟังก์ชันใน registry
  - `test_levels_match_report`: rule 18 → LIGHT, 15 → MEDIUM, 9 → STRONG, 23 → MEDIUM
  - `test_rule1_l_to_r_in_cluster_light`: ปลา, LIGHT → `cluster == "r"`; log มี `(1, 0)`
  - `test_rule4_n_before_k`: coda n ตามด้วยพยางค์ onset k → coda `"ŋ"` ที่ LIGHT
  - `test_rule5_ng_before_t`: coda ŋ ตามด้วย onset t → `"n"`
  - `test_rule18_shorten_closed_long_light`: long=True, coda="ŋ" → long=False; long=True, coda=None → ไม่เปลี่ยน
  - `test_rule25_high_tone_flattened`: tone 3 → tone ∈ {0, 4} ที่ LIGHT
  - `test_rule26_low_to_mid`: tone 1, long → tone 0
  - `test_rule7_only_at_strong`: coda m: STRONG → `hints["coda_mode"] == "mu"`; MEDIUM → ไม่มี `"mu"`
  - `test_rule16_phrase_final_epenthetic_strong`: พยางค์สุดท้ายมี coda t, STRONG → `coda_mode == "epenthetic"`
  - `test_deterministic`: เรียกสองครั้งด้วย input เดียวกัน → ผลและ log เท่ากัน
  - `test_input_not_mutated`
- [ ] **Step 2:** Run `pytest tests/test_rules.py -v` → FAIL
- [ ] **Step 3:** Implement — registry ด้วย decorator `@rule(id)`; `apply_rules` เรียกกฎที่ `level <= requested` ตามลำดับ id
- [ ] **Step 4:** Run → PASS
- [ ] **Step 5:** Commit `feat: accent rule engine with 33-rule table`

---

### Task 4: Mora mapper, tone maps, pauses

**Files:**
- Create: `thaijp_accent/sbv2_symbols.py`, `thaijp_accent/mora.py`, `thaijp_accent/tone_maps.yaml`, `tests/test_mora.py`

**Interfaces:**
- Consumes: `Syllable` (มี hints จาก Task 3), `AccentLevel`
- Produces:
  - `JA_SYMBOLS: frozenset[str]`, `PUNCT: frozenset[str]` — คัดลอกจาก `style_bert_vits2/nlp/symbols.py` ที่ tag 2.7.0 (ใส่ URL + commit ในคอมเมนต์)
  - `syllables_to_morae(sylls: list[Syllable], level: AccentLevel, tone_map: str = "default") -> list[Mora]`

ตารางแปลง (ตัดสินแล้ว):
- onset: p,pʰ→p · t,tʰ→t · k,kʰ→k · b→b · d→d · m→m · n→n · ŋ→g · f→f · s→s · h→h · tɕ,tɕʰ→ch · j→y · w→w · r,l→r · ""→ไม่มีพยัญชนะ
- vowel: a→a · i→i · u,ɯ→u · e,ɛ→e · o,ɔ→o · ɤ→`hints.er_vowel` (ค่าเริ่ม o) · ia→i a (หรือ e เมื่อ `ia_mode=="e"`) · ɯa,ua→u a
- long → เพิ่ม mora สระเดิม
- cluster → mora แทรก C + (o ถ้า C ∈ {t,d} ไม่งั้น u) ทุกระดับ
- coda: m,n,ŋ→`N` (`mu`→ mora `m u`) · p,t,k→`q` (`epenthetic`→ `q` + p u / t o / k u แบบคาตาคานะ เช่น ケット) · w→mora `u` · j→mora `i`
- pitch: `tone_maps.yaml`
  - `default`: 3→H; 1→L; 0→ใช้ pitch ของ mora ก่อนหน้า (ต้นวลี L); 4→L,H; 2→H,L (1 mora → H)
  - `learner`: 2→H,L; 4,3→L,H; 0,1→L (1 mora: 2→H, 4/3→H)
  - mora ต่อท้ายพยางค์เดียวกันใช้ pitch สุดท้ายของคอนทัวร์
- pause (กฎ 31): ใส่ Mora `phones=[","]`, `is_pause=True`, pitch 0 ที่รอยต่อคำ — STRONG ทุกรอยต่อ; LIGHT/MEDIUM เฉพาะเมื่อคำก่อนหน้ามี ≥ 2 พยางค์

- [ ] **Step 1: Write failing tests**
  - `test_all_phones_in_symbol_set`: ทุก fixture word × ทุก level → ทุก phone ∈ `JA_SYMBOLS | PUNCT`
  - `test_phuket_strong`: ภูเก็ต STRONG → phones รวม `["p","u","u","k","e","q","t","o"]`
  - `test_phuket_light`: ภูเก็ต LIGHT → `["p","u","u","k","e","q"]` (ภู เป็นพยางค์เปิด กฎ 18 จึงไม่ย่อ)
  - `test_cluster_epenthesis`: ปลา → morae `[p u][r a][a]`
  - `test_coda_nasal_N`: งาน → onset `g`, จบด้วย `N`
  - `test_tone_default_rising_two_mora`: tone 4 long → pitch `[0, 1]`
  - `test_tone_learner_high_equals_rising`: tone 3 และ 4 ได้ pitch เดียวกันใน `learner`
  - `test_pause_strong_every_boundary`: 3 คำ → 2 pause
  - `test_pause_light_after_multisyllable_only`
- [ ] **Step 2:** Run `pytest tests/test_mora.py -v` → FAIL
- [ ] **Step 3:** Implement ตามตาราง
- [ ] **Step 4:** Run → PASS
- [ ] **Step 5:** Commit `feat: syllable-to-mora mapper with tone maps and pauses`

---

### Task 5: SBV2 input, katakana, end-to-end frontend

**Files:**
- Create: `thaijp_accent/kana.py`, `thaijp_accent/sbv2_io.py`, `thaijp_accent/pipeline.py`, `tests/test_pipeline.py`, `tests/data/test_sentences.tsv`

**Interfaces:**
- Consumes: `thai_to_syllables`, `apply_rules`, `syllables_to_morae`
- Produces:
  - `mora_to_kana(m: Mora) -> str` (`N`→ン, `q`→ッ, `,`→、, สระซ้ำของพยางค์เดียวกัน→ー)
  - `to_sbv2_input(morae: list[Mora]) -> Sbv2Input` — ใส่ `"_"` หัวท้าย (tone 0); `word2ph` = 1 ต่อ pad + จำนวน phone ต่อ kana 1 ตัว
  - `class ThaiJPFrontend: __init__(g2p: ThaiG2P, lexicon_path: Path | None = None)`; `convert(text: str, level: AccentLevel, tone_map: str = "default") -> tuple[Sbv2Input, dict]` → dict มี `warnings`, `rule_log`
  - CLI: `python -m thaijp_accent.pipeline "<ข้อความ>" --level light`

- [ ] **Step 1:** เขียน `test_sentences.tsv` 30 ประโยค (คอลัมน์ `id \t text \t tags`) ให้ครอบคลุม tag: `tone0..tone4`, `long`, `short`, `cluster`, `coda_p/t/k/m/n/ng/w/j`, `diphthong`; 5 แถวแรก tag `m0`
- [ ] **Step 2: Write failing tests**
  - `test_invariants`: ทุกประโยค × ทุก level → `len(phones)==len(tones)`, `sum(word2ph)==len(phones)`, `len(word2ph)==len(kana)+2`, `set(tones) <= {0,1}`
  - `test_sentence_coverage`: ทุก tag ข้างบนปรากฏ ≥ 1 ครั้ง
  - `test_empty_input_raises`: `convert("  ...  ")` → `ValueError("no Thai syllables")`
  - `test_phuket_kana_strong`: kana == `"プーケット"`
  - `test_deterministic_end_to_end`
- [ ] **Step 3:** Run `pytest tests/test_pipeline.py -v` → FAIL
- [ ] **Step 4:** Implement
- [ ] **Step 5:** Run ทั้งชุด `pytest -m "not network"` → PASS
- [ ] **Step 6:** Commit `feat: end-to-end Thai→SBV2 frontend and test sentences`

---

### Task 6: Colab runtime, checkpoint safety, training runner

**Files:**
- Create: `colab/runtime.py`, `colab/ckpt.py`, `colab/train_runner.py`, `tests/test_colab.py`

**Interfaces:**
- Produces:
  - `ensure_gpu() -> dict` (`name`, `vram_gb`) — ไม่มี GPU → `RuntimeError("No GPU: Runtime > Change runtime type > T4 GPU")`
  - `pick_batch_size(vram_gb: float) -> int`: ≥22→8, ≥15→4, ≥11→2, อื่น ๆ→1
  - `run_paths(drive_root: Path, run_id: str) -> dict[str, Path]` (`data`, `models`, `logs`, `manifest`)
  - `write_manifest(path: Path, **fields) -> None` (JSON: config, sbv2_tag, data_hash, gpu, timestamp)
  - `data_hash(dir: Path) -> str` (sha256 ของชื่อไฟล์ + ขนาด เรียงลำดับ)
  - `find_resumable_checkpoint(models_dir: Path, loader=torch.load) -> int | None` — step สูงสุดที่ `G_{s}.pth`, `D_{s}.pth`, `WD_{s}.pth` โหลดได้ทั้งหมด; ชุดที่โหลดไม่ได้ย้ายไป `models_dir/corrupt/`
  - `prune_checkpoints(models_dir: Path, keep: int = 3) -> list[Path]`
  - `detect_oom(log_text: str) -> bool` — จับ `"CUDA out of memory"` / `"OutOfMemoryError"`
  - `run_training(cmd: list[str], models_dir: Path, budget_s: float, poll_s: float = 30, popen=subprocess.Popen) -> str` → `"finished" | "budget_stop" | "oom"` (notebook `02` เมื่อได้ `"oom"` จะลด batch size ลงครึ่งหนึ่งแล้วเรียกใหม่ จนถึงขั้นต่ำ 1); เมื่อเกิน budget รอให้มี checkpoint ใหม่ (step > ค่าตอนครบ budget) แล้วส่ง SIGINT; เรียก `prune_checkpoints` ทุกครั้งที่มี checkpoint ใหม่

- [ ] **Step 1: Write failing tests** (ใช้ tmp_path, loader และ popen ปลอม)
  - `test_pick_batch_size_t4`: `pick_batch_size(15.0) == 4`
  - `test_resume_skips_corrupt_latest`: มี step 1000 (ดี) และ 1500 (loader raise) → คืน 1000 และ 1500 อยู่ใน `corrupt/`
  - `test_resume_requires_all_three`: step 2000 มีแค่ G → ข้าม
  - `test_resume_none_when_empty`
  - `test_prune_keeps_latest_three`
  - `test_ensure_gpu_raises_without_nvidia_smi` (monkeypatch `shutil.which` → None)
  - `test_run_training_budget_waits_for_next_ckpt`: popen ปลอมที่ยังรันอยู่; สร้าง checkpoint ใหม่หลังครบ budget → คืน `"budget_stop"` และ SIGINT ถูกส่งหลัง checkpoint ปรากฏ
  - `test_data_hash_stable`
  - `test_run_training_reports_oom`: popen ปลอมจบด้วย exit code ≠ 0 และ stderr มี `"CUDA out of memory"` → `"oom"`
- [ ] **Step 2:** Run `pytest tests/test_colab.py -v` → FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run → PASS
- [ ] **Step 5:** Commit `feat: Colab runtime helpers with resumable checkpoints`

---

### Task 7: Data prep + BERT neutralization (E2)

**Files:**
- Create: `thaijp_accent/data_prep.py`, `tests/test_data_prep.py`

**Interfaces:**
- Produces:
  - `qc_clip(wav_path: Path) -> list[str]` — ปัญหา: `"too_short"` (< 1.0 s), `"too_long"` (> 15 s), `"clipping"` (> 0.1% ของ sample ที่ |x| ≥ 0.999), `"silent"` (RMS < 1e-3)
  - `write_lists(entries: list[tuple[str, str]], speaker: str, out_dir: Path, val_n: int = 4, seed: int = 0) -> None` — เขียน `train.list` / `val.list` รูปแบบ SBV2 `wav|speaker|JP|text`
  - `neutralize_bert(dataset_dir: Path, mode: Literal["zeros", "mean"]) -> int` — เขียนทับไฟล์ `*.bert.pt` ทุกไฟล์ด้วยค่าศูนย์หรือค่าเฉลี่ยทั่ว dataset (shape เดิม) แล้วคืนจำนวนไฟล์; เขียน marker `.bert_neutralized_<mode>` เพื่อไม่ทำซ้ำ

- [ ] **Step 1: Write failing tests**: `test_qc_flags_short`, `test_qc_flags_clipping`, `test_qc_clean_clip_ok` (สร้าง wav ด้วย numpy + soundfile); `test_write_lists_format_and_split` (val 4 แถว, ไม่ซ้ำกับ train); `test_neutralize_zeros_keeps_shape`; `test_neutralize_idempotent` (เรียกครั้งที่สองคืน 0)
- [ ] **Step 2:** Run → FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run → PASS
- [ ] **Step 5:** Commit `feat: dataset QC, list writer and BERT neutralization`

---

### Task 8: Inference adapter

**Files:**
- Create: `thaijp_accent/synth.py`, `tests/test_synth.py`

**Interfaces:**
- Consumes: `Sbv2Input`
- Produces: `synthesize(model, inp: Sbv2Input, bert_mode: Literal["kana", "zeros", "mean"], style: str = "Neutral", style_weight: float = 1.0, length: float = 1.0, intonation_scale: float = 1.0, mean_bert: "torch.Tensor | None" = None) -> tuple[int, "np.ndarray"]`

- [ ] **Step 1: ตรวจ API ที่ tag 2.7.0** — อ่าน `style_bert_vits2/tts_model.py` และ `models/infer.py` ว่ารับ `given_phone` / `given_tone` หรือไม่ แล้วบันทึกผลใน docstring ของ `synthesize` ถ้ารับ ให้ส่ง `text=inp.kana, given_phone=inp.phones[1:-1], given_tone=inp.tones[1:-1]` (ตรวจว่า pad ถูกเติมเองหรือไม่); ถ้าไม่รับ ให้เรียก `infer()` ระดับล่างโดยสร้าง tensor `phones`, `tones`, `lang_ids` (JP), `bert` เอง
- [ ] **Step 2: Write failing tests** (model ปลอมที่บันทึก argument):
  - `test_zeros_bert_shape`: bert ที่ส่งเข้า shape `(1024, len(inp.phones))` และเป็นศูนย์ทั้งหมด
  - `test_phones_passed_verbatim`
  - `test_kana_mode_passes_text`
- [ ] **Step 3:** Run → FAIL
- [ ] **Step 4:** Implement
- [ ] **Step 5:** Run → PASS
- [ ] **Step 6:** Commit `feat: SBV2 inference adapter with direct phone/tone input`

---

### Task 9: Evaluation

**Files:**
- Create: `thaijp_accent/eval.py`, `tests/test_eval.py`

**Interfaces:**
- Produces:
  - `thai_cer(ref: str, hyp: str) -> float` — ตัดช่องว่างและวรรคตอนก่อน แล้วใช้ `jiwer.cer`
  - `transcribe(wavs: list[Path], model_id: str = "openai/whisper-large-v3") -> list[str]` (language="th"; network/GPU)
  - `utmos(wavs: list[Path]) -> list[float]` (network/GPU, ใช้ SpeechMOS)
  - `make_listening_sheet(rows: list[dict], out_csv: Path, seed: int = 0) -> None` — สลับลำดับ, ซ่อน condition, คอลัมน์ `clip_id, file, intelligibility_1to5, accentedness_1to5, transcript`
  - `summarize(results_csv: Path) -> "pd.DataFrame"` — ตาราง level × bert_mode × tone_map: CER เฉลี่ย, UTMOS เฉลี่ย, คะแนนเฉลี่ย พร้อมคอลัมน์ `m4_pass` (`level=="light"` และ CER ≤ 0.25 และ intelligibility ≥ 3.5)

- [ ] **Step 1: Write failing tests**: `test_cer_ignores_spaces` (`thai_cer("ปลา กิน", "ปลากิน") == 0`), `test_listening_sheet_hides_condition`, `test_summarize_m4_pass_flag`
- [ ] **Step 2:** Run → FAIL
- [ ] **Step 3:** Implement
- [ ] **Step 4:** Run → PASS
- [ ] **Step 5:** Commit `feat: evaluation metrics and listening sheet`

---

### Task 10: Notebooks (00–04)

**Files:**
- Create: `notebooks_src/00_m0_smoke_test.py`, `01_data_prep.py`, `02_train.py`, `03_infer_thai.py`, `04_eval.py`, `Makefile` (target `notebooks`: `jupytext --to ipynb notebooks_src/*.py -o notebooks/`), `tests/test_notebooks.py`

**Interfaces:**
- Consumes: ทุกอย่างจาก Task 2–9

ทุก notebook เริ่มด้วย cell เดียวกัน: mount Drive → `pip install` ที่ pin เวอร์ชัน (git tag 2.7.0) → clone repo นี้ → `run_paths()`; ทุก cell รันซ้ำได้โดยไม่เกิดผลเสีย; markdown ภาษาไทยก่อนทุก code cell บอกว่า "ทำอะไร / ทำไม / ควรเห็นอะไร"

- `00`: โหลดโมเดล SBV2 つくよみちゃน จาก Hugging Face (repo id ที่ยืนยันแล้ว จดลง cell) สังเคราะห์ 5 ประโยค `m0` ทั้งแบบ kana และแบบ phone ตรง, รัน `transcribe` → CER, สร้างฟอร์มให้คะแนน 5 ประโยค, แสดงผล PASS/FAIL ตามเกณฑ์ M0
- `01`: ดาวน์โหลด corpus あみたろ (เปิดหน้าเงื่อนไข, ให้ผู้ใช้ติ๊ก checkbox ยอมรับก่อนดาวน์โหลด, เขียน `CREDITS.txt`) → `qc_clip` ทั้งชุด (แสดงตารางปัญหา) → resample 44.1 kHz → `write_lists` → preprocess ของ SBV2 (สร้าง bert) → `data_hash` → manifest
- `02`: `ensure_gpu` → `pick_batch_size` → สร้าง config (save ทุก 500 step, `bf16_run=False`) → ตัวเลือก `EXPERIMENT = "E1" | "E2"` (E2 เรียก `neutralize_bert(..., "zeros")` บนสำเนา dataset แยกชื่อ run) → copy dataset ไป `/content/local` → `find_resumable_checkpoint` → `run_training(budget_s=4*3600)` → เปิด TensorBoard → สังเคราะห์ประโยค `m0` จาก checkpoint ล่าสุด
- `03`: ช่องกรอกข้อความไทย + dropdown level / tone_map / bert_mode / style + slider `length`, `intonation_scale` → แสดง phone/kana/rule log → เล่นเสียงและบันทึก wav ลง Drive
- `04`: สังเคราะห์ 30 ประโยค × 3 level × (E1, E2) × 2 tone_map → `transcribe`, `utmos` → `make_listening_sheet` → (หลังผู้ใช้กรอกคะแนน) `summarize` แสดงตารางและ `m4_pass`

- [ ] **Step 1: Write failing test** `test_notebooks_build_and_parse`: build ด้วย jupytext ได้, `nbformat.validate` ผ่าน, ทุก notebook มี cell แรกที่มี `drive.mount`, ทุก code cell มี markdown cell ก่อนหน้า
- [ ] **Step 2:** Run → FAIL
- [ ] **Step 3:** เขียน notebook ทั้ง 5 ตามรายละเอียดข้างบน
- [ ] **Step 4:** Run `make notebooks && pytest tests/test_notebooks.py -v` → PASS
- [ ] **Step 5:** รัน `00` บน CPU ในเครื่อง (ข้าม cell Drive ด้วย env `LOCAL=1`) กับโมเดลつくよみちゃน ถ้าดาวน์โหลดได้ ถ้าไม่ได้ให้จดไว้ว่ารอผู้ใช้รันบน Colab
- [ ] **Step 6:** Commit `feat: Colab notebooks 00-04`

---

### Task 11: README (วิธีใช้สำหรับผู้ใช้)

**Files:**
- Create: `README.md`

- [ ] **Step 1:** เขียน README ภาษาไทย: ลำดับรัน notebook, สิ่งที่ต้องเตรียม (Drive ว่าง ≥ 15 GB), ทำอย่างไรเมื่อ session ถูกตัด (กดรัน `02` ใหม่ทั้ง notebook), เกณฑ์ M0/M4, ข้อความเครดิตあみたろ/つくよみちゃん, license (AGPL-3.0 ของ SBV2)
- [ ] **Step 2:** Run `pytest -m "not network"` ทั้งชุด → PASS
- [ ] **Step 3:** Commit `docs: README for running the pipeline`
