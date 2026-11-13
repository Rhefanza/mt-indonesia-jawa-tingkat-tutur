"""EDA dataset Unggah-Ungguh (config translation). Jalankan dari folder project."""
import re, sys, itertools, collections, difflib, math
import pandas as pd

PATH = "data/unggah_ungguh/translation.csv"
LV = {0: "Ngoko", 1: "Ngoko Alus", 2: "Krama", 3: "Krama Alus"}
df = pd.read_csv(PATH)
df = df.rename(columns={"javanese sentence": "jv", "indonesian sentence": "id", "english sentence": "en"})
df["lv"] = df["label"].map(LV)
tok = lambda s: re.findall(r"[\w'\-]+", str(s).lower())

def h(t): print(f"\n===== {t} =====")

# ---------- A. Struktur dan kualitas ----------
h("A1. Ukuran dan nilai kosong")
print("rows", len(df), "| kolom", list(df.columns))
for c in ["jv", "id", "en"]:
    s = df[c].astype(str)
    print(f"{c}: null={df[c].isna().sum()} empty={(s.str.strip()=='').sum()} "
          f"spasi_tepi={(s != s.str.strip()).sum()} spasi_ganda={s.str.contains('  ').sum()}")

h("A2. Duplikat")
print("baris identik (label,jv,id):", df.duplicated(["label", "jv", "id"]).sum())
print("jv identik lintas baris:", df.duplicated(["jv"]).sum())
dup_jv = df[df.duplicated(["jv"], keep=False)].sort_values("jv")
print(dup_jv[["group", "lv", "jv"]].head(12).to_string(index=False))

h("A3. Struktur group")
g = df.groupby("group")
sizes = g.size().value_counts().sort_index()
print("ukuran group:", dict(sizes))
combos = collections.Counter(tuple(sorted(x)) for x in g["label"].apply(list))
for k, v in combos.most_common():
    print("  ", [LV[i] for i in k], v)
nid = g["id"].nunique()
print("group dengan >1 kalimat Indonesia berbeda:", (nid > 1).sum())
id_norm = df["id"].str.strip().str.lower()
id_groups = df.assign(idn=id_norm).groupby("idn")["group"].nunique()
print("kalimat Indonesia yang muncul di >1 group:", (id_groups > 1).sum())
print(id_groups[id_groups > 1].head(8).to_string())

h("A4. Varian identik antar tingkat dalam satu group")
same = collections.Counter()
for _, gg in g:
    rows = gg.set_index("label")["jv"].str.strip()
    for a, b in itertools.combinations(sorted(rows.index), 2):
        if rows[a] == rows[b]:
            same[(LV[a], LV[b])] += 1
print(dict(same) or "tidak ada")

h("A5. Karakter non-ASCII di jv")
chars = collections.Counter(ch for s in df["jv"].astype(str) for ch in s if ord(ch) > 127)
print(chars.most_common(20))

# ---------- B. Leksikal ----------
h("B1. Panjang per tingkat (kata)")
df["jv_len"] = df["jv"].map(lambda s: len(tok(s)))
df["id_len"] = df["id"].map(lambda s: len(tok(s)))
print(df.groupby("lv")[["jv_len", "id_len"]].agg(["mean", "max"]).round(2).to_string())
print("rasio jv/id:", round(df["jv_len"].sum() / df["id_len"].sum(), 3))

h("B2. Kosakata per tingkat dan Jaccard")
voc = {LV[l]: set(itertools.chain.from_iterable(df[df.label == l]["jv"].map(tok))) for l in LV}
for k, v in voc.items(): print(f"  {k}: {len(v)} kata unik")
for a, b in itertools.combinations(voc, 2):
    j = len(voc[a] & voc[b]) / len(voc[a] | voc[b])
    print(f"  Jaccard {a} vs {b}: {j:.3f}")

h("B3. Jarak antar tingkat dalam group (proporsi token berubah)")
dist = collections.defaultdict(list)
for _, gg in g:
    rows = gg.set_index("label")["jv"]
    for a, b in itertools.combinations(sorted(rows.index), 2):
        ta, tb = tok(rows[a]), tok(rows[b])
        sm = difflib.SequenceMatcher(None, ta, tb)
        dist[(LV[a], LV[b])].append(1 - sm.ratio())
for k, v in sorted(dist.items(), key=lambda x: sum(x[1]) / len(x[1])):
    print(f"  {k[0]:>10} vs {k[1]:<10}: {sum(v)/len(v):.3f} (n={len(v)})")

h("B4. Kata penanda khas per tingkat (log-odds, frekuensi >= 8)")
cnt = {l: collections.Counter(itertools.chain.from_iterable(df[df.label == l]["jv"].map(tok))) for l in LV}
tot = {l: sum(c.values()) for l, c in cnt.items()}
allc = sum(cnt.values(), collections.Counter())
for l in LV:
    rest = allc - cnt[l]; rt = sum(rest.values())
    sc = []
    for w, f in cnt[l].items():
        if allc[w] < 8: continue
        lo = math.log((f + .5) / (tot[l] - f + .5)) - math.log((rest[w] + .5) / (rt - rest[w] + .5))
        sc.append((lo, w, f, allc[w]))
    sc.sort(reverse=True)
    print(f"  {LV[l]}:", ", ".join(f"{w}({f}/{a})" for _, w, f, a in sc[:14]))

h("B5. Overlap token jv dengan kalimat Indonesia (indikasi kata yang bisa disalin)")
def ov(r):
    a, b = tok(r["jv"]), set(tok(r["id"]))
    return sum(t in b for t in a) / max(len(a), 1)
df["ov"] = df.apply(ov, axis=1)
print(df.groupby("lv")["ov"].mean().round(3).to_string())

# ---------- C. Baseline metrik ----------
import sacrebleu
h("C1. Copy baseline (hipotesis = kalimat Indonesia)")
for l in LV:
    d = df[df.label == l]
    b = sacrebleu.corpus_bleu(d["id"].tolist(), [d["jv"].tolist()]).score
    c = sacrebleu.corpus_chrf(d["id"].tolist(), [d["jv"].tolist()], word_order=2).score
    print(f"  {LV[l]:<10} BLEU {b:6.2f}  chrF++ {c:6.2f}  (n={len(d)})")

h("C2. Style copy baseline (hipotesis = varian Ngoko, referensi = tingkat lain)")
piv = df.pivot_table(index="group", columns="label", values="jv", aggfunc="first")
for l in [1, 2, 3]:
    d = piv[[0, l]].dropna()
    b = sacrebleu.corpus_bleu(d[0].tolist(), [d[l].tolist()]).score
    c = sacrebleu.corpus_chrf(d[0].tolist(), [d[l].tolist()], word_order=2).score
    print(f"  Ngoko -> {LV[l]:<10} BLEU {b:6.2f}  chrF++ {c:6.2f}  (n={len(d)})")

# ---------- D. Simulasi split ----------
from sklearn.model_selection import GroupShuffleSplit
h("D1. Group-based split 80/10/10 (seed 42)")
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr, rest = next(gss.split(df, groups=df["group"]))
dr = df.iloc[rest]
gss2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
va, te = next(gss2.split(dr, groups=dr["group"]))
sp = {"train": df.iloc[tr], "valid": dr.iloc[va], "test": dr.iloc[te]}
for k, d in sp.items():
    print(f"  {k:<5} rows {len(d):5d} groups {d.group.nunique():4d}", dict(d["lv"].value_counts()))
ids = {k: set(d["id"].str.strip().str.lower()) for k, d in sp.items()}
print("  kalimat Indonesia sama di train & test:", len(ids["train"] & ids["test"]))
from sklearn.model_selection import train_test_split
r_tr, r_te = train_test_split(df, test_size=0.2, random_state=42)
leak = len(set(r_tr["id"].str.strip().str.lower()) & set(r_te["id"].str.strip().str.lower()))
print(f"  pembanding random split 80/20: kalimat Indonesia test yang juga ada di train = {leak} dari {r_te['id'].nunique()}")

# ---------- E. Conversation ----------
h("E. Conversation config")
cv = pd.read_csv("data/unggah_ungguh/conversation.csv")
print("rows", len(cv), "| A:", dict(cv["a utterance category"].map(LV).value_counts()), "| B:", dict(cv["b utterance category"].map(LV).value_counts()))
