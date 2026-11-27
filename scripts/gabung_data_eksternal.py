"""Gabungkan Unggah-Ungguh (train saja) + NgokoKrama + NusaTranslation + NusaX untuk 3 tingkat.

Jalankan dari root repo:  python scripts/gabung_data_eksternal.py
Input : data/external/{ngokokrama/NgokoKrama.xlsx, nusatranslation/jav-*.csv, nusax/train.csv} (di-gitignore)
Output: data/merged/train_eksternal_3lv.csv, data/merged/train_eksternal_3lv_laporan.json (di-gitignore)

- Valid/test Unggah-Ungguh tidak disentuh. Baris eksternal yang mirip valid/test dibuang.
- Label tingkat baris eksternal ditentukan classifier TF-IDF 4 tingkat yang dilatih di train Unggah-Ungguh.
  Sengaja bukan juri DistilBERT / reranker RoBERTa, supaya evaluasi tidak sirkular.
- "Krama" NgokoKrama hanya boleh menjadi krama atau krama_alus; "ngoko" hanya ngoko. Selain itu dibuang.
- NgokoKrama tidak berlisensi: pakai lokal saja, jangan disebarkan. Nonaktifkan dengan --tanpa-ngokokrama.
"""
import json
import re
import sys

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

LEVELS = ["ngoko", "krama", "krama_alus"]
MIN_KATA, MAX_KATA = 3, 25
P_MIN = {"ngoko": 0.60, "krama": 0.60, "krama_alus": 0.85}  # KA < 0,85 sering keliru (audit sampel)
MIRIP = 0.80          # ambang cosine dengan valid/test
CAP_NUSATRANS = 300  # supaya ngoko tidak menenggelamkan tingkat lain
SEED = 42


def norm(s):
    s = str(s).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


def kunci(s):
    return re.sub(r"\W+", " ", str(s).lower()).strip()


def split(n):
    return pd.read_csv(f"data/unggah_ungguh/split/{n}.csv")


tr, va, te = split("train"), split("valid"), split("test")
clf = make_pipeline(
    make_union(TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2),
               TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)),
    LogisticRegression(C=20, max_iter=3000, class_weight="balanced"))
clf.fit(tr["javanese"], tr["level"])
lap = {"akurasi_pelabel_di_valid": {}}
pv, kv = clf.predict(va["javanese"]), clf.predict_proba(va["javanese"]).max(1)
for l in clf.classes_:
    yakin = (pv == l) & (kv >= P_MIN.get(l, 0.60))
    lap["akurasi_pelabel_di_valid"][l] = {
        "recall": round(float((pv[va.level == l] == l).mean()), 3),
        "presisi_saat_yakin": round(float((va.level[yakin] == l).mean()), 3), "n_yakin": int(yakin.sum())}

rows = []
if "--tanpa-ngokokrama" not in sys.argv:
    nk = pd.read_excel("data/external/ngokokrama/NgokoKrama.xlsx", sheet_name="All")
    for c, peran in [("javanese ngoko", "ngoko"), ("javanese krama", "krama")]:
        rows.append(pd.DataFrame({"indonesian": nk["indonesian"], "javanese": nk[c], "source": "ngokokrama", "peran": peran}))
for s in ["train", "valid", "test"]:
    d = pd.read_csv(f"data/external/nusatranslation/jav-{s}.csv")
    rows.append(pd.DataFrame({"indonesian": d["ind_text"], "javanese": d["tgt_text"], "source": "nusatranslation", "peran": "umum"}))
d = pd.read_csv("data/external/nusax/train.csv")
rows.append(pd.DataFrame({"indonesian": d["indonesian"], "javanese": d["javanese"], "source": "nusax", "peran": "umum"}))
x = pd.concat(rows, ignore_index=True).dropna()
buang = {}


def saring(mask, alasan):
    global x
    buang[alasan] = {k: int(v) for k, v in x[~mask].groupby("source").size().items()}
    x = x[mask].reset_index(drop=True)


for c in ["indonesian", "javanese"]:
    x[c] = x[c].map(norm)
lap["awal"] = x.groupby(["source", "peran"]).size().rename(lambda t: t, axis=0).to_dict()
lap["awal"] = {"/".join(k): int(v) for k, v in lap["awal"].items()}
ni, nj = x.indonesian.str.split().str.len(), x.javanese.str.split().str.len()
saring(ni.between(MIN_KATA, MAX_KATA) & nj.between(MIN_KATA, MAX_KATA) & (ni / nj).between(0.6, 1.6), "panjang")
noise = r"https?:|www\.|\[url\]|\[username\]|@\w|#\w|[^\x00-\x7FéèêÉÈ–—…]"
saring(~(x.indonesian.str.contains(noise, regex=True, case=False) | x.javanese.str.contains(noise, regex=True, case=False)), "url_mention_emoji")
saring(x.indonesian.map(kunci) != x.javanese.map(kunci), "sama_dengan_sumber")

ev = pd.concat([va, te])
tolak = np.zeros(len(x), bool)
for c in ["indonesian", "javanese"]:
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4)).fit(pd.concat([ev[c], x[c]]))
    sim = np.asarray((v.transform(x[c]) @ v.transform(ev[c]).T).max(1).todense()).ravel()
    lap[f"kemiripan_maks_{c}"] = round(float(sim.max()), 3)
    tolak |= sim >= MIRIP
saring(pd.Series(~tolak), "mirip_valid_test")

P = clf.predict_proba(x["javanese"])
x["level"], x["p_level"] = clf.classes_[P.argmax(1)], P.max(1).round(3)
boleh = {"ngoko": ["ngoko"], "krama": ["krama", "krama_alus"], "umum": ["ngoko"]}
saring(pd.Series([l in boleh[p] for l, p in zip(x.level, x.peran)]), "tingkat_tidak_cocok")
saring(x.p_level >= x.level.map(P_MIN), "label_tidak_yakin")

uu = tr[tr.level.isin(LEVELS)].assign(source="unggah_ungguh", p_level=1.0)
sudah = set(zip(uu.indonesian.map(kunci), uu.javanese.map(kunci)))
x["k"] = list(zip(x.indonesian.map(kunci), x.javanese.map(kunci)))
saring(~x.k.duplicated() & ~x.k.isin(sudah), "duplikat")
nt = x[x.source == "nusatranslation"]
if len(nt) > CAP_NUSATRANS:
    saring(~x.index.isin(nt.drop(nt.sample(CAP_NUSATRANS, random_state=SEED).index).index), "cap_nusatranslation")

x["group"] = 100000 + pd.factorize(x.indonesian.map(kunci))[0]
kol = ["group", "level", "indonesian", "javanese", "source", "p_level"]
out = pd.concat([uu[kol], x[kol]], ignore_index=True)
out.to_csv("data/merged/train_eksternal_3lv.csv", index=False, encoding="utf-8")
tab = out.groupby(["source", "level"]).size().unstack(fill_value=0)
for n, e in [("valid", va), ("test", te)]:
    lap[f"bocor_persis_{n}"] = {c: len(set(x[c].map(kunci)) & set(e[c].map(kunci))) for c in ["indonesian", "javanese"]}
lap.update(dibuang=buang, hasil={s: r.to_dict() for s, r in tab.iterrows()}, total={k: int(v) for k, v in tab.sum().items()})
json.dump(lap, open("data/merged/train_eksternal_3lv_laporan.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=int)
print(json.dumps(lap, indent=1, ensure_ascii=False, default=int)); print(tab)
