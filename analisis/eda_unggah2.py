"""EDA lanjutan: kandidat typo/OCR, cakupan sumbu tingkat, statistik tokenizer."""
import re, collections, itertools
import pandas as pd

df = pd.read_csv("data/unggah_ungguh/translation.csv").rename(
    columns={"javanese sentence": "jv", "indonesian sentence": "id", "english sentence": "en"})
LV = {0: "Ngoko", 1: "Ngoko Alus", 2: "Krama", 3: "Krama Alus"}
tok = lambda s: re.findall(r"[\w'\-]+", str(s).lower())
def h(t): print(f"\n===== {t} =====")

h("F1. Kandidat typo/OCR (kata frekuensi 1 yang berjarak edit 1 dari kata frekuensi >= 5)")
cnt = collections.Counter(itertools.chain.from_iterable(df["jv"].map(tok)))
freq = [w for w, c in cnt.items() if c >= 5]
def ed1(a, b):
    if abs(len(a) - len(b)) > 1 or a == b: return False
    if len(a) == len(b): return sum(x != y for x, y in zip(a, b)) == 1
    if len(a) > len(b): a, b = b, a
    return any(b[:i] + b[i+1:] == a for i in range(len(b)))
cands = []
for w, c in cnt.items():
    if c == 1 and len(w) >= 5:
        near = [f for f in freq if ed1(w, f)]
        if near: cands.append((w, near[0], cnt[near[0]]))
print("jumlah kandidat:", len(cands))
for w, n, c in cands[:25]: print(f"  {w} ~ {n} ({c})")

h("F2. Cakupan dua sumbu: dasar (ngoko/krama) x hormat (biasa/alus)")
df["base"] = df["label"].map({0: "ngoko", 1: "ngoko", 2: "krama", 3: "krama"})
df["alus"] = df["label"].map({0: "biasa", 1: "alus", 2: "biasa", 3: "alus"})
print(pd.crosstab(df["base"], df["alus"]))

h("G. Statistik tokenizer")
from transformers import AutoTokenizer
models = {"M2M100-418M": ("facebook/m2m100_418M", "id", "jv"),
          "NLLB-600M": ("facebook/nllb-200-distilled-600M", "ind_Latn", "jav_Latn")}
markers = ["panjenenganipun", "ngandikanipun", "dipunladosaken", "sampeyan", "adalem", "kagungan"]
for name, (repo, src, tgt) in models.items():
    t = AutoTokenizer.from_pretrained(repo)
    print(f"\n--- {name} ({repo}) vocab={len(t)}")
    def n_tok(s): return len(t(s, add_special_tokens=False)["input_ids"])
    unk = t.unk_token_id
    for col in ["id", "jv"]:
        lens = df[col].map(n_tok)
        words = df[col].map(lambda s: len(str(s).split()))
        print(f"  {col}: mean {lens.mean():.1f} max {lens.max()} fertility {lens.sum()/words.sum():.2f} token/kata  >64 tok: {(lens>64).sum()}")
    for l in LV:
        d = df[df.label == l]["jv"]
        ids = list(itertools.chain.from_iterable(t(x, add_special_tokens=False)["input_ids"] for x in d))
        fert = len(ids) / d.map(lambda s: len(s.split())).sum()
        print(f"    jv {LV[l]:<10} fertility {fert:.2f}  unk {sum(i == unk for i in ids)}")
    print("  pecahan kata penanda:", {m: t.tokenize(m) for m in markers})
