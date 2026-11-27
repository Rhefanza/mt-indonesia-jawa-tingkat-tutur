"""Uji CPU: seberapa relevan contoh few-shot hasil retrieval (TF-IDF char 3-5, sisi Indonesia, tingkat sama, train asli)
untuk baris valid Krama dan Krama Alus. Ukuran objektif: cakupan token referensi Jawa oleh target contoh (top-3 vs acak),
dan token referensi yang tidak pernah muncul di target train sama sekali. Jalankan dari root repo."""
import re, json, random, sys
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
sys.path.insert(0, "scripts")
from juri_tetap import load3
tr, va = load3("train"), load3("valid")
tok = lambda s: set(re.findall(r"\w+", str(s).lower()))
semua_target = set().union(*[tok(t) for t in tr.javanese])
R = {}
for lv in ["krama", "krama_alus"]:
    T, V = tr[tr.level == lv].reset_index(drop=True), va[va.level == lv].reset_index(drop=True)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(T.indonesian)
    S = (vec.transform(V.indonesian) @ vec.transform(T.indonesian).T).toarray()
    rng = random.Random(42); r = {"n": len(V), "cos_top1": [], "cak_top3": [], "cak_acak3": [], "oov_train": [], "cak_top3_konten": []}
    for i in range(len(V)):
        ref = tok(V.javanese[i]); top = np.argsort(-S[i])[:3]; acak = rng.sample(range(len(T)), 3)
        cov = lambda ix: len(ref & set().union(*[tok(T.javanese[j]) for j in ix])) / max(1, len(ref))
        r["cos_top1"].append(float(S[i, top[0]])); r["cak_top3"].append(cov(top)); r["cak_acak3"].append(cov(acak))
        r["oov_train"].append(len(ref - semua_target) / max(1, len(ref)))
        if lv == "krama_alus" and i < 6:
            print(f"Q {V.indonesian[i]} | REF {V.javanese[i]}\n  1 ({S[i,top[0]]:.2f}) {T.indonesian[top[0]]} | {T.javanese[top[0]]}")
    R[lv] = {"n": r["n"], **{k: round(float(np.mean(v)), 3) for k, v in r.items() if k != "n" and v},
             "cos_top1_ge_0_5": int(sum(c >= .5 for c in r["cos_top1"]))}
print(json.dumps(R, indent=1))
