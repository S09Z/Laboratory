"""Japanese symbol set of Style-Bert-VITS2, vendored from
https://github.com/litagin02/Style-Bert-VITS2/blob/2.7.0/style_bert_vits2/nlp/symbols.py
Only these may be sent to the model (no Thai symbols are added)."""

JA_SYMBOLS = frozenset([
    "N", "a", "a:", "b", "by", "ch", "d", "dy", "e", "e:", "f", "g", "gy", "h", "hy",
    "i", "i:", "j", "k", "ky", "m", "my", "n", "ny", "o", "o:", "p", "py", "q", "r",
    "ry", "s", "sh", "t", "ts", "ty", "u", "u:", "w", "y", "z", "zy",
])
PUNCT = frozenset(["!", "?", "…", ",", ".", "'", "-", "SP", "UNK"])
PAD = "_"
