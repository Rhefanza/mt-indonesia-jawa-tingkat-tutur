"""Bandingkan lengan semua 15 vs 20 epoch di VALID (bootstrap grup) + juri tetap untuk arah Indonesia -> Jawa. Jalankan dari root repo."""
import sys, json
import numpy as np, pandas as pd
from sacrebleu.metrics import CHRF
sys.path.insert(0, "scripts")
import juri_tetap as J
c = CHRF(word_order=2)
P = {"id_jv": ("kaggle/data_eksternal_v1/output/valid_semua_pilih.csv", "kaggle/data_eksternal_ep20/output/valid_semua_pilih.csv", "javanese"),
     "jv_id": ("kaggle/reverse_jv_id/output/valid_semua.csv", "kaggle/reverse_jv_id_ep20/output/valid_semua.csv", "indonesian")}
R = {}
for arah, (a, b, ref) in P.items():
    A, B = pd.read_csv(a), pd.read_csv(b); assert (A["index"] == B["index"]).all()
    f = lambda d: c.corpus_score(d.hyp.astype(str).tolist(), [d[ref].tolist()]).score
    for lv in ["all"] + J.LEVELS:
        x, y = [(d if lv == "all" else d[d.level == lv]).reset_index(drop=True) for d in (A, B)]
        u = x.group.unique(); ix = {g: np.where(x.group == g)[0] for g in u}; rng = np.random.default_rng(0); z = []
        for _ in range(1000):
            i = np.concatenate([ix[g] for g in rng.choice(u, len(u))]); z.append(f(y.iloc[i]) - f(x.iloc[i]))
        R[f"{arah}/{lv}"] = {"ep15": round(f(x), 2), "ep20": round(f(y), 2), "selisih": round(f(y) - f(x), 2),
                             "interval_90": [round(float(q), 2) for q in np.percentile(z, [5, 95])]}
    if arah == "id_jv":
        js = J.get_judges(J.load3("train"))
        for n, d in (("ep15", A), ("ep20", B)):
            p = np.mean([J.probs(j, d.hyp.astype(str).tolist()) for j in js], 0).argmax(-1)
            R[f"juri_tetap/{n}"] = J.accuracy(p, d.level.map(J.LID).values)
json.dump(R, open("kaggle/data_eksternal_ep20/output/banding_ep15_ep20.json", "w"), indent=1)
for k, v in R.items(): print(k, v)
