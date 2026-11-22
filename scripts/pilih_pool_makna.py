"""Seleksi varian pool-makna-v1 di valid (CPU), lalu kunci pilihan dan buat kernel run B (test).

Aturan dipatok di diskusi `2026-11-22-kualitas-terjemahan-tingkat-tutur.md` sebelum run A:
- baseline = beam8, λ=2 (resep demo v2); varian = pool gabungan beam8+beam16, λ=2, μ ∈ {0, 0.5, 1}
- promosi: batas bawah CI 90% (2.000 paired bootstrap per `group`) selisih corpus chrF++ Krama Alus > 0,
  kehilangan ≤1 keputusan benar Krama Alus (juri valid), turun ≤1 poin chrF++ di Ngoko dan Krama
- μ > 0 hanya bila uji fakta reverse ≥16/20 DAN gerbang cakupan (≥3 baris pulih fakta inti di audit) lolos,
  dan harus mengungguli μ=0 pada pool yang sama
- pilih chrF++ Krama Alus valid tertinggi; seri → μ terkecil; tidak ada yang lolos → baseline

Pemakaian:
  python scripts/pilih_pool_makna.py seleksi      # pertama: tulis lembar audit cakupan; kedua (audit terisi): kunci pilihan
  python scripts/pilih_pool_makna.py audit-buta   # setelah run B: lembar audit buta 15 grup test
"""
import argparse, hashlib, json, random, shutil, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
from sacrebleu.metrics import CHRF

sys.stdout.reconfigure(encoding="utf-8")  # konsol Windows (cp1252) tidak bisa mencetak μ dan ≥
ROOT = Path(__file__).resolve().parents[1]
K1 = ROOT / "kaggle/pool_makna_v1"
K2 = ROOT / "kaggle/pool_makna_v1_test"
LEVELS = ["ngoko", "krama", "krama_alus"]
LAM, N_BOOT, SEED_BOOT, SEED_AUDIT = 2.0, 2000, 0, 20261122
CH = CHRF(word_order=2)
VARIANTS = {"baseline": (0.0, (8,)), "gabung_mu0": (0.0, (8, 16)), "gabung_mu0.5": (0.5, (8, 16)), "gabung_mu1": (1.0, (8, 16))}


def sha256(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def pick(d, mu, widths):
    """Sama dengan pick() di notebook: dedup per sumber–tingkat, pemecah seri = urutan asal."""
    x = d[d.beam_width.isin(widths)].sort_values(["source_id", "beam_width", "beam_rank"])
    x = x.drop_duplicates(["source_id", "candidate"]).copy()
    rr = np.array([x[f"rr_logp_{l}"].iat[i] for i, l in enumerate(x.requested)])
    x["_s"] = x.forward_logp_mean.values + LAM * rr + (mu * x.reverse_logp_mean.values if mu else 0.0)
    x["_o"] = np.arange(len(x))
    b = x.sort_values(["source_id", "_s", "_o"], ascending=[True, False, True]).drop_duplicates("source_id")
    return b.drop(columns=["_s", "_o"]).set_index("source_id")


def corpus(h, r):
    return CH.corpus_score(list(h), [list(r)]).score


def metrics(sel):
    out = {lv: round(corpus(s.candidate, s.reference), 2) for lv, s in sel.groupby("requested")}
    ka = sel[sel.requested == "krama_alus"]
    out["KA_juri_benar"] = int((ka.valid_judge_label == "krama_alus").sum())
    out["KA_n"] = len(ka)
    return out


def boot_lb(base, var, n=N_BOOT, seed=SEED_BOOT):
    """Batas bawah CI 90% selisih corpus chrF++ Krama Alus (var − base), resampel per group berpasangan."""
    b = base[base.requested == "krama_alus"].sort_index()
    v = var.loc[b.index]
    groups = b.group.values
    ug = np.unique(groups)
    idx = {g: np.where(groups == g)[0] for g in ug}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        s = np.concatenate([idx[g] for g in rng.choice(ug, len(ug))])
        diffs.append(corpus(v.candidate.values[s], v.reference.values[s]) - corpus(b.candidate.values[s], b.reference.values[s]))
    return round(float(np.percentile(diffs, 5)), 3), round(float(np.mean(diffs)), 3)


def oracle(d, base):
    """Celah oracle sentence chrF++ Krama Alus: kandidat terbaik di pool − pilihan baseline."""
    ka = d[d.requested == "krama_alus"].copy()
    ka["sc"] = [CH.sentence_score(c, [r]).score for c, r in zip(ka.candidate, ka.reference)]
    b = base[base.requested == "krama_alus"]
    bs = {sid: CH.sentence_score(c, [r]).score for sid, c, r in zip(b.index, b.candidate, b.reference)}
    res = {}
    for name, w in {"beam8": (8,), "beam16": (16,), "gabung": (8, 16)}.items():
        g = (ka[ka.beam_width.isin(w)].groupby("source_id").sc.max() - pd.Series(bs)).dropna()
        res[name] = {"rerata": round(g.mean(), 2), "median": round(g.median(), 2), "n_ge3": int((g >= 3).sum()), "n": len(g)}
    return res, ka


def audit_sheet(ka, base, path):
    """Maks 3 teks unik per baris Krama Alus valid: chrF++ tertinggi, rank 1 beam8, rank 1 beam16."""
    rows = []
    for sid, g in ka.groupby("source_id"):
        b = base.loc[sid]
        picks = [("oracle_chrF", g.sort_values("sc", ascending=False).candidate.iat[0]),
                 ("rank1_beam8", g[(g.beam_width == 8) & (g.beam_rank == 0)].candidate.iat[0]),
                 ("rank1_beam16", g[(g.beam_width == 16) & (g.beam_rank == 0)].candidate.iat[0])]
        seen = {b.candidate}
        for asal, c in picks:
            if c in seen:
                continue
            seen.add(c)
            rows.append({"source_id": sid, "group": b.group, "indonesian": b.indonesian, "reference": b.reference,
                         "top1_baseline": b.candidate, "asal": asal, "kandidat": c,
                         "top1_kehilangan_fakta_inti": "", "kandidat_pulih_fakta_inti": "", "kandidat_tingkat_turun": ""})
    pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8")
    return len(rows)


def gate_cakupan(path):
    a = pd.read_csv(path, dtype=str).fillna("")
    for c in ["top1_kehilangan_fakta_inti", "kandidat_pulih_fakta_inti", "kandidat_tingkat_turun"]:
        bad = ~a[c].isin(["0", "1"])
        if bad.any():
            sys.exit(f"audit belum lengkap: kolom {c} kosong/tidak valid di {int(bad.sum())} baris ({path})")
    ok = a[(a.top1_kehilangan_fakta_inti == "1") & (a.kandidat_pulih_fakta_inti == "1") & (a.kandidat_tingkat_turun == "0")]
    return ok.source_id.nunique()


def seleksi(args):
    out = K1 / "output"
    lock = K1 / "pilihan.json"
    if lock.exists():
        sys.exit(f"pilihan sudah dikunci di {lock}; tidak boleh diseleksi ulang.")
    man = json.loads((out / "manifest_valid.json").read_text(encoding="utf-8"))
    kand = out / "kandidat_valid.csv"
    assert sha256(kand) == man["outputs"]["kandidat_valid_sha256"], "kandidat_valid.csv tidak cocok dengan manifest"
    d = pd.read_csv(kand, keep_default_na=False, na_values=[""])
    sel = {k: pick(d, mu, w) for k, (mu, w) in VARIANTS.items() if mu == 0}
    orc, ka = oracle(d, sel["baseline"])
    print("baseline valid chrF++ (all):", round(corpus(sel["baseline"].candidate, sel["baseline"].reference), 2),
          "| v2 mencatat 62.66 | manifest", man["gates"]["baseline_valid_chrF++"])
    print("celah oracle Krama Alus:", json.dumps(orc))
    print("uji fakta reverse:", man["gates"]["uji_fakta_benar"], "/ 20")

    audit = K1 / "audit_cakupan.csv"
    if not audit.exists():
        n = audit_sheet(ka, sel["baseline"], audit)
        print(f"\nLembar audit cakupan ditulis: {audit} ({n} kandidat). Isi tiga kolom terakhir dengan 0/1 "
              "berdasarkan sumber Indonesia, lalu jalankan perintah ini lagi.")
        return
    cak = gate_cakupan(audit)
    rev_ok = man["gates"]["uji_fakta_lolos"] and cak >= 3
    print(f"gerbang cakupan: {cak} baris pulih (syarat ≥3) | reverse boleh dipromosikan: {rev_ok}")
    for k, (mu, w) in VARIANTS.items():
        if mu > 0 and rev_ok:
            sel[k] = pick(d, mu, w)

    base, mb = sel["baseline"], metrics(sel["baseline"])
    tabel = {"baseline": {**mb, "lb90": None, "lolos": True}}
    for k, s in sel.items():
        if k == "baseline":
            continue
        m = metrics(s)
        lb, mean = boot_lb(base, s)
        lolos = (lb > 0 and m["KA_juri_benar"] >= mb["KA_juri_benar"] - 1
                 and all(m[lv] >= mb[lv] - 1 for lv in ["ngoko", "krama"]))
        if VARIANTS[k][0] > 0:
            lolos = lolos and m["krama_alus"] > metrics(sel["gabung_mu0"])["krama_alus"]
        tabel[k] = {**m, "lb90": lb, "selisih_rerata": mean, "lolos": bool(lolos)}
    for k, v in tabel.items():
        print(f"{k:14s} {v}")
    kandidat = [k for k, v in tabel.items() if k != "baseline" and v["lolos"]]
    win = max(kandidat, key=lambda k: (tabel[k]["krama_alus"], -VARIANTS[k][0])) if kandidat else "baseline"
    mu, w = VARIANTS[win]
    pil = {"varian": win, "mu": mu, "widths": list(w), "lambda": LAM, "kandidat_valid_sha256": man["outputs"]["kandidat_valid_sha256"],
           "audit_cakupan_sha256": sha256(audit), "gerbang_cakupan": cak, "uji_fakta_benar": man["gates"]["uji_fakta_benar"],
           "celah_oracle": orc, "tabel_valid": tabel, "dikunci": time.strftime("%Y-%m-%d %H:%M:%S")}
    lock.write_text(json.dumps(pil, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nPEMENANG valid: {win} (μ={mu}, beam {list(w)}). Dikunci di {lock}")
    buat_kernel_test(pil)


def buat_kernel_test(pil):
    K2.mkdir(exist_ok=True)
    nb = json.loads((K1 / "pool_makna_v1.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    src = "".join(code[0]["source"])
    assert src.startswith('MODE = "valid"'), "sel pertama notebook berubah"
    p = {k: pil[k] for k in ["varian", "mu", "widths", "lambda", "kandidat_valid_sha256"]}
    code[0]["source"] = [f'MODE = "test"\n', f"PILIHAN = {json.dumps(p)}\n"]
    (K2 / "pool_makna_v1_test.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = json.loads((K1 / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update({"id": "mdevian/unggah-ungguh-pool-makna-v1-test", "title": "Unggah-Ungguh Pool Makna v1 Test",
                 "code_file": "pool_makna_v1_test.ipynb",
                 "kernel_sources": ["mdevian/unggah-ungguh-final-3lv-v2", "mdevian/unggah-ungguh-pool-makna-v1"]})
    (K2 / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Kernel run B dibuat di {K2}. Push: cd kaggle/pool_makna_v1_test && kaggle kernels push -p .")


def audit_buta(args):
    out = K2 / "output"
    b, p = (pd.read_csv(out / f"pred_{n}.csv") for n in ["baseline", "pilihan"])
    assert (b["index"].values == p["index"].values).all()
    rng = random.Random(SEED_AUDIT)
    pilih = []
    for lv, n in [("krama_alus", 10), ("krama", 3), ("ngoko", 2)]:
        g = sorted(set(b.loc[b.level == lv, "group"]) - {x for x, _ in pilih})
        pilih += [(x, lv) for x in rng.sample(g, n)]
    rows, kunci = [], []
    for i, (g, lv) in enumerate(pilih):
        m = (b.group == g) & (b.level == lv)
        hb, hp = b.loc[m, "hyp"].iat[0], p.loc[m, "hyp"].iat[0]
        kiri_pilihan = rng.random() < 0.5
        kiri, kanan = (hp, hb) if kiri_pilihan else (hb, hp)
        rows.append({"no": i + 1, "tingkat_diminta": lv, "indonesian": b.loc[m, "indonesian"].iat[0],
                     "output_A": kiri, "output_B": kanan, "makna_A": "", "tingkat_A": "", "makna_B": "", "tingkat_B": "",
                     "konteks_relasi_kurang": "", "catatan": ""})
        kunci.append({"no": i + 1, "group": g, "A": "pilihan" if kiri_pilihan else "baseline", "identik": hb == hp,
                      "reference": b.loc[m, "javanese"].iat[0]})
    pd.DataFrame(rows).to_csv(K2 / "audit_buta.csv", index=False, encoding="utf-8")
    pd.DataFrame(kunci).to_csv(K2 / "kunci_audit_buta.csv", index=False, encoding="utf-8")
    print(f"audit_buta.csv (untuk penilai) dan kunci_audit_buta.csv (jangan dibuka penilai) ditulis di {K2}.\n"
          "Skala makna: 0 fakta inti rusak, 1 detail hilang, 2 semua terjaga. Tingkat: 0 salah, 1 campur/meragukan, 2 tepat.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("perintah", choices=["seleksi", "audit-buta"])
    a = ap.parse_args()
    {"seleksi": seleksi, "audit-buta": audit_buta}[a.perintah](a)
