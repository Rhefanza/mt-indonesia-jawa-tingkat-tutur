"""Juri tetap: nilai ulang semua output kontrol 3 tingkat dengan juri yang SAMA.

Masalah yang diperbaiki: setiap kernel melatih ulang juri (w11wo/javanese-distilbert-small), dan output yang identik
bisa dinilai 0,92 di satu run dan 0,85 di run lain. Angka ketepatan tingkat antar eksperimen jadi tidak bisa dibandingkan.

Yang dilakukan skrip ini:
1. Melatih juri dengan resep v2 (4 epoch, lr 5e-5, batch 32, max 64 token, train split 3 tingkat) untuk beberapa seed,
   lalu menyimpan weights ke models/juri_tetap/seed<N>/ (di-gitignore). Jika sudah ada, weights dimuat ulang.
2. Juri tetap = ensemble: rata-rata probabilitas semua seed.
3. Menilai ulang setiap kaggle/*/output*/control_*.csv yang hanya berisi tingkat ngoko/krama/krama_alus.

Jalankan dari root repo:
    python scripts/juri_tetap.py
Hasil: analisis/juri_tetap/hasil.csv, analisis/juri_tetap/juri.json, analisis/juri_tetap/README.md
"""
import glob
import json
import os
import random
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = "w11wo/javanese-distilbert-small"
SEEDS = [42, 13, 2024]
LEVELS = ["ngoko", "krama", "krama_alus"]
LID = {l: i for i, l in enumerate(LEVELS)}
MAXLEN = 64
MODEL_DIR = "models/juri_tetap"
OUT_DIR = "analisis/juri_tetap"
dev = "cuda" if torch.cuda.is_available() else "cpu"
if dev == "cpu":
    torch.set_num_threads(os.cpu_count())


def load3(name):
    d = pd.read_csv(f"data/unggah_ungguh/split/{name}.csv")
    d = d[d["level"].isin(LEVELS)].copy()
    d["label"] = d["level"].map(LID)
    return d.reset_index(drop=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s)


def train_judge(seed, tr):
    """Resep sama dengan train_classifier di kaggle/final_3lv_v2."""
    tok = AutoTokenizer.from_pretrained(BASE)
    mod = AutoModelForSequenceClassification.from_pretrained(BASE, num_labels=len(LEVELS)).to(dev)
    seed_all(seed)
    opt = torch.optim.AdamW(mod.parameters(), lr=5e-5)
    X, y = tr["javanese"].tolist(), tr["label"].tolist()
    for ep in range(4):
        mod.train(); order = list(range(len(X))); random.shuffle(order)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            enc = tok([X[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
            loss = mod(**enc, labels=torch.tensor([y[j] for j in b], device=dev)).loss
            opt.zero_grad(); loss.backward(); opt.step()
        print(f"  seed {seed} epoch {ep + 1}/4", flush=True)
    return tok, mod.eval()


def get_judges(tr):
    judges = []
    for s in SEEDS:
        d = f"{MODEL_DIR}/seed{s}"
        if os.path.exists(f"{d}/config.json"):
            print("muat juri", d, flush=True)
            tok, mod = AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).to(dev).eval()
        else:
            print("latih juri seed", s, flush=True)
            tok, mod = train_judge(s, tr)
            os.makedirs(d, exist_ok=True); mod.save_pretrained(d); tok.save_pretrained(d)
        judges.append((s, tok, mod))
    return judges


def probs(judge, sents):
    _, tok, mod = judge
    out = []
    for i in range(0, len(sents), 64):
        enc = tok([str(x) for x in sents[i:i + 64]], return_tensors="pt", padding=True, truncation=True,
                  max_length=MAXLEN).to(dev)
        with torch.no_grad():
            out.append(F.softmax(mod(**enc).logits.float(), -1).cpu())
    return torch.cat(out).numpy()


def accuracy(pred, y):
    pred, y = np.asarray(pred), np.asarray(y)
    return round(float((pred == y).mean()), 4), {l: round(float((pred[y == i] == i).mean()), 4) for i, l in enumerate(LEVELS)}


def main(pattern="kaggle/*/output*/control_*.csv"):
    tr, va, te = (load3(n) for n in ["train", "valid", "test"])
    judges = get_judges(tr)
    names = [f"seed{s}" for s, _, _ in judges] + ["ensemble"]

    def predict_all(sents):
        P = [probs(j, sents) for j in judges]
        return {**{n: p.argmax(-1) for n, p in zip(names, P)}, "ensemble": np.mean(P, 0).argmax(-1)}

    # akurasi juri pada referensi asli (seberapa bisa dipercaya juri itu sendiri)
    info = {"base": BASE, "seeds": SEEDS, "device": dev, "akurasi_pada_referensi": {}}
    for split, d in [("valid", va), ("test", te)]:
        pr = predict_all(d["javanese"].tolist())
        info["akurasi_pada_referensi"][split] = {n: dict(zip(["all", "per_level"], accuracy(pr[n], d["label"])))
                                                 for n in names}
    print(json.dumps(info["akurasi_pada_referensi"], indent=1), flush=True)

    rows = []
    for f in sorted(glob.glob(pattern)):
        d = pd.read_csv(f)
        if not set(d["requested"]) <= set(LEVELS):
            continue
        f = f.replace("\\", "/")
        exp = f.split("/")[1] + ("/" + f.split("/")[2] if f.split("/")[2] != "output" else "")
        y = d["requested"].map(LID).values
        pr = predict_all(d["output"].tolist())
        orig = accuracy(d["classified"].map(LID).values, y) if "classified" in d else (None, {})
        row = {"eksperimen": exp, "file": os.path.basename(f)[len("control_"):-len(".csv")],
               "juri_kernel": orig[0], "juri_kernel_KA": orig[1].get("krama_alus")}
        for n in names:
            a, lv = accuracy(pr[n], y)
            row[n] = a
            row[f"{n}_KA"] = lv["krama_alus"]
            if n == "ensemble":
                row["ensemble_ngoko"], row["ensemble_krama"] = lv["ngoko"], lv["krama"]
        seeds_acc = [row[f"seed{s}"] for s in SEEDS]
        row["sebaran_seed"] = round(max(seeds_acc) - min(seeds_acc), 4)
        rows.append(row)
        print(f"{exp:40} {row['file']:28} kernel {row['juri_kernel']} | ensemble {row['ensemble']} "
              f"(KA {row['ensemble_KA']}) | sebaran seed {row['sebaran_seed']}", flush=True)

    os.makedirs(OUT_DIR, exist_ok=True)
    res = pd.DataFrame(rows)
    res.to_csv(f"{OUT_DIR}/hasil.csv", index=False)
    json.dump(info, open(f"{OUT_DIR}/juri.json", "w"), indent=1)
    write_readme(res, info)


def write_readme(res, info):
    ref = info["akurasi_pada_referensi"]["test"]
    lines = [
        "# Juri tetap: ketepatan tingkat dinilai ulang dengan juri yang sama",
        "",
        "Dibuat oleh `scripts/juri_tetap.py`. Juri = `w11wo/javanese-distilbert-small`, resep v2, dilatih di train split",
        f"dengan seed {', '.join(map(str, SEEDS))}; **juri tetap = ensemble** (rata-rata probabilitas). Weights di",
        "`models/juri_tetap/` (tidak di-commit; jalankan ulang skrip untuk membuatnya).",
        "",
        "## Akurasi juri pada referensi test asli",
        "",
        "| Juri | Semua | Ngoko | Krama | Krama Alus |",
        "|---|---|---|---|---|",
    ]
    for n, v in ref.items():
        p = v["per_level"]
        lines.append(f"| {n} | {v['all']:.4f} | {p['ngoko']:.4f} | {p['krama']:.4f} | {p['krama_alus']:.4f} |")
    lines += [
        "",
        "## Ketepatan tingkat output (75 kalimat test × 3 tingkat = 225 permintaan)",
        "",
        "`juri kernel` = angka asli yang tercatat di kernel masing-masing (juri berbeda tiap run).",
        "`sebaran seed` = selisih terbesar antar juri seed tunggal untuk output yang sama.",
        "",
        "| Eksperimen | Output | Juri kernel | **Ensemble** | Ngoko | Krama | **Krama Alus** | Sebaran seed |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for _, r in res.iterrows():
        jk = "-" if pd.isna(r["juri_kernel"]) else f"{r['juri_kernel']:.4f}"
        lines.append(f"| {r['eksperimen']} | {r['file']} | {jk} | **{r['ensemble']:.4f}** | {r['ensemble_ngoko']:.4f} | "
                     f"{r['ensemble_krama']:.4f} | **{r['ensemble_KA']:.4f}** | {r['sebaran_seed']:.4f} |")
    lines += ["", "`final_3lv_cendol/output_run1_bug` adalah run yang rusak (chrF++ ~15); disertakan hanya sebagai pembanding.", ""]
    open(f"{OUT_DIR}/README.md", "w", encoding="utf-8").write("\n".join(lines))
    print("ditulis:", f"{OUT_DIR}/README.md", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
