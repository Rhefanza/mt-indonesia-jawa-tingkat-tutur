"""Bagi Unggah-Ungguh (config translation) menjadi train/valid/test per group makna.

Jalankan dari folder project:  python scripts/split_data.py
Output: data/unggah_ungguh/split/{train,valid,test}.csv

- Group-based: semua tingkat dari satu kalimat Indonesia berada di split yang sama,
  supaya tidak ada kebocoran (random split membocorkan ~95% kalimat Indonesia test).
- Stratified: proporsi label dijaga mirip di tiap split (StratifiedGroupKFold, 10 fold):
  fold 0 = test, fold 1 = valid, sisanya = train (~80/10/10).
- Normalisasi ringan saja: spasi di tepi/ganda dan tanda kutip miring. Typo tidak diubah.
"""
import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

SRC = Path("data/unggah_ungguh/translation.csv")
OUT = Path("data/unggah_ungguh/split")
SEED = 42
LEVEL = {0: "ngoko", 1: "ngoko_alus", 2: "krama", 3: "krama_alus"}
BASE = {0: "ngoko", 1: "ngoko", 2: "krama", 3: "krama"}
HONOR = {0: "biasa", 1: "alus", 2: "biasa", 3: "alus"}


def norm(s: str) -> str:
    s = str(s).replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    return re.sub(r"\s+", " ", s).strip()


df = pd.read_csv(SRC).rename(columns={
    "javanese sentence": "javanese", "indonesian sentence": "indonesian", "english sentence": "english"})
for c in ["javanese", "indonesian", "english"]:
    df[c] = df[c].map(norm)
df["level"] = df["label"].map(LEVEL)
df["base"] = df["label"].map(BASE)
df["honor"] = df["label"].map(HONOR)

sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=SEED)
fold = pd.Series(-1, index=df.index)
for k, (_, idx) in enumerate(sgkf.split(df, df["label"], groups=df["group"])):
    fold.iloc[idx] = k
df["split"] = fold.map(lambda k: "test" if k == 0 else "valid" if k == 1 else "train")

cols = ["index", "group", "label", "level", "base", "honor", "indonesian", "javanese", "english"]
OUT.mkdir(parents=True, exist_ok=True)
parts = {}
for name in ["train", "valid", "test"]:
    parts[name] = df[df["split"] == name][cols].sort_values(["group", "label"])
    parts[name].to_csv(OUT / f"{name}.csv", index=False, encoding="utf-8")

# ---- verifikasi ----
print(f"{'split':<6}{'rows':>6}{'groups':>8}  " + "  ".join(f"{v:>10}" for v in LEVEL.values()))
for name, d in parts.items():
    counts = d["level"].value_counts()
    print(f"{name:<6}{len(d):>6}{d.group.nunique():>8}  " + "  ".join(f"{counts.get(v, 0):>10}" for v in LEVEL.values()))
g = {k: set(d["group"]) for k, d in parts.items()}
i = {k: set(d["indonesian"].str.lower()) for k, d in parts.items()}
for a, b in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    print(f"overlap {a}-{b}: group {len(g[a] & g[b])}, kalimat Indonesia {len(i[a] & i[b])}")
assert sum(len(d) for d in parts.values()) == len(df)
print("total", len(df), "OK ->", OUT)
