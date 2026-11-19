"""Synthesize 10 long Thai texts with the trained model on the Mac (CPU).

Run:  ../thaijp_lab/.venv/bin/python scripts/synth_long.py [light|medium|strong ...]
Output: <Laboratory>/outputs_mac/long_<id>_<level>.wav  (words are separated by spaces, like tests/data/test_sentences.tsv)
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
PROJ = Path(__file__).resolve().parent.parent
LAB = PROJ.parent
SBV2 = LAB / "thaijp_lab" / "Style-Bert-VITS2"
sys.path[:0] = [str(PROJ), str(SBV2)]
os.chdir(SBV2)  # SBV2 reads configs/ and bert/ from cwd

TEXTS = {
    "L01": "สวัสดี ครับ วัน นี้ อากาศ ดี มาก ผม เลย ชวน เพื่อน ไป เดิน เล่น ที่ สวน สาธารณะ แล้ว ซื้อ ไอศกรีม กิน ด้วย กัน",
    "L02": "เมื่อ วาน ฉัน ไป ตลาด กับ แม่ ซื้อ ปลา ผัก และ ผลไม้ หลาย อย่าง กลับ มา บ้าน แล้ว ช่วย กัน ทำ อาหาร เย็น",
    "L03": "ฉัน ชอบ กิน ผัดไทย ส้มตำ และ ข้าวเหนียว มะม่วง มาก ที่ สุด ใน โลก ถ้า ไป เมือง ไทย ต้อง ลอง ให้ ได้",
    "L04": "ปี หน้า ฉัน อยาก ไป เที่ยว ภูเก็ต กับ เชียงใหม่ ดู ทะเล ภูเขา และ วัด สวย ๆ แล้ว ถ่าย รูป เก็บ ไว้ เยอะ ๆ",
    "L05": "ขอบคุณ มาก ครับ ที่ ช่วย สอน ภาษา ญี่ปุ่น ให้ ผม ผม จะ ตั้งใจ เรียน ทุก วัน และ พยายาม พูด ให้ ชัด ขึ้น",
    "L06": "ร้าน นี้ ข้าว อร่อย ราคา ถูก พนักงาน ใจดี มาก ผม แนะนำ ให้ เพื่อน ทุก คน มา ลอง กิน ที่ นี่ สัก ครั้ง",
    "L07": "พรุ่ง นี้ เช้า ฉัน ต้อง ตื่น เร็ว เพราะ มี นัด กับ หมอ หลัง จาก นั้น จะ ไป ซื้อ ของ ที่ ห้าง ใกล้ บ้าน",
    "L08": "แม่ ชอบ ปลูก ดอกไม้ สวย ๆ หน้า บ้าน ทุก เช้า แม่ จะ รด น้ำ และ คุย กับ ต้นไม้ เหมือน เป็น เพื่อน กัน",
    "L09": "วัน นี้ ฝน ตก หนัก มาก ฉัน เลย ไม่ ได้ ออก ไป ไหน นั่ง ดู หนัง อยู่ บ้าน กับ แมว สอง ตัว ทั้ง วัน",
    "L10": "ยินดี ที่ ได้ รู้จัก ทุก คน ฉัน เป็น เด็ก ผู้หญิง จาก เมือง ไทย ฝาก ตัว ด้วย นะ คะ วัน นี้ มา คุย กัน สนุก ๆ",
}


def main(levels: list[str]) -> None:
    import soundfile as sf
    from style_bert_vits2.tts_model import TTSModel

    from thaijp_accent import mora
    from thaijp_accent.g2p import UmT5G2P
    from thaijp_accent.pipeline import ThaiJPFrontend
    from thaijp_accent.synth import synthesize
    from thaijp_accent.types import AccentLevel

    d = LAB / "sbv2" / "model_assets" / "amitaro_runrun"
    ckpt = sorted(p for p in d.glob("*.safetensors"))[-1]
    model = TTSModel(model_path=ckpt, config_path=d / "config.json",
                     style_vec_path=d / "style_vectors.npy", device="cpu")
    lex = LAB / "lexicon.tsv"
    fe = ThaiJPFrontend(UmT5G2P(device="cpu"), lexicon_path=lex if lex.exists() else PROJ / "thaijp_accent" / "lexicon.tsv")
    out = LAB / "outputs_mac"
    out.mkdir(exist_ok=True)
    for tid, text in TEXTS.items():
        for lv in levels:
            # "<level>-flow" = same accent rules but no pause between words (experiment, not in the core code)
            base, flow = lv.removesuffix("-flow"), lv.endswith("-flow")
            orig = mora._pause_after_word
            if flow:
                mora._pause_after_word = lambda *_: False
            try:
                inp, _ = fe.convert(text, AccentLevel[base.upper()])
            finally:
                mora._pause_after_word = orig
            t = time.time()
            sr, audio = synthesize(model, inp, "kana")
            path = out / f"long_{tid}_{lv}.wav"
            sf.write(path, audio, sr)
            print(f"{path.name}: {len(audio) / sr:.1f}s audio in {time.time() - t:.1f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["light"])
