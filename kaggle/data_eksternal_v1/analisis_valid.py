"""Bootstrap grup (valid) dan juri tetap untuk keempat lengan. Jalankan dari root repo."""
import json, sys
import numpy as np, pandas as pd
from sacrebleu.metrics import CHRF
sys.path.insert(0, "scripts")
import juri_tetap as J
O = "kaggle/data_eksternal_v1/output"
L = ["base", "ngokokrama", "tanpa_ngokokrama", "semua"]
D = {k: pd.read_csv(f"{O}/valid_{k}_pilih.csv") for k in L}
b = D["base"]
for k in L: assert (D[k]["index"] == b["index"]).all()
m = CHRF(word_order=2)
chrf = lambda d: m.corpus_score(d["hyp"].astype(str).tolist(), [d["javanese"].tolist()]).score
judges = J.get_judges(J.load3("train"))
R = {k: {} for k in L}
for k in L:
    p = np.mean([J.probs(j, D[k]["hyp"].tolist()) for j in judges], 0).argmax(-1)
    R[k]["juri_tetap"] = J.accuracy(p, D[k]["level"].map(J.LID).values)
for lv in ["all"] + J.LEVELS:
    S = {k: D[k] if lv == "all" else D[k][D[k].level == lv] for k in L}
    S = {k: s.reset_index(drop=True) for k, s in S.items()}
    g = S["base"]["group"].values; u = np.unique(g); ix = {x: np.where(g == x)[0] for x in u}
    rng = np.random.default_rng(0); B = {k: [] for k in L}
    for _ in range(1000):
        i = np.concatenate([ix[x] for x in rng.choice(u, len(u))])
        for k in L: B[k].append(chrf(S[k].iloc[i]))
    for k in L:
        d = np.array(B[k]) - np.array(B["base"])
        R[k][lv] = {"n": len(g), "chrF++": round(chrf(S[k]), 2), "delta": round(chrf(S[k]) - chrf(S["base"]), 2),
                    "interval_90": [round(float(x), 2) for x in np.percentile(d, [5, 95])]}
json.dump(R, open(f"{O}/analisis_valid.json", "w"), indent=1)
for k in L: print(k, json.dumps(R[k]))
print("== KA")
for i in b.index[b.level == "krama_alus"]:
    print(f"[{b['index'][i]}] ID {b.indonesian[i]}\n  REF {b.javanese[i]}\n  B {D['base'].hyp[i]}\n  " + ("= sama" if D['base'].hyp[i]==D['semua'].hyp[i] else f"S {D['semua'].hyp[i]}"))
