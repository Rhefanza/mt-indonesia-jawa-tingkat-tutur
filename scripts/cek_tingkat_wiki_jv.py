"""Uji murah: perkiraan porsi kalimat Ngoko/Krama/Krama Alus di Wikipedia Jawa (bahan back-translation).
Juri tetap (DistilBERT, dilatih pada kalimat pendek Unggah-Ungguh) -> perkiraan kasar, beda domain. Jalankan dari root repo."""
import re, json, random, sys
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
sys.path.insert(0, "scripts")
import juri_tetap as J
p = hf_hub_download("wikimedia/wikipedia", "20231101.jv/train-00000-of-00001.parquet", repo_type="dataset", local_dir="data/external/wiki_jv")
d = pd.read_parquet(p, columns=["text"])
sents = []
for t in d.text:
    for s in re.split(r"(?<=[.!?])\s+", t):
        s = s.strip()
        if 5 <= len(s.split()) <= 20 and not re.search(r"[\d|=\[\]{}<>]", s):
            sents.append(s)
random.seed(0); smp = random.sample(sents, 3000)
js = J.get_judges(J.load3("train"))
P = np.mean([J.probs(j, smp) for j in js], 0)
lab = P.argmax(-1); conf = P.max(-1)
R = {"artikel": len(d), "kalimat_5_20_kata": len(sents), "sampel": len(smp),
     "porsi": {l: round(float((lab == i).mean()), 3) for i, l in enumerate(J.LEVELS)},
     "porsi_yakin_0_9": {l: round(float(((lab == i) & (conf >= .9)).mean()), 3) for i, l in enumerate(J.LEVELS)}}
print(json.dumps(R))
for i, l in enumerate(J.LEVELS):
    ix = np.where((lab == i) & (conf >= .9))[0][:6]
    for k in ix: print(l, round(float(conf[k]), 2), smp[k])
json.dump(R, open("data/external/wiki_jv/tingkat.json", "w"), indent=1)
