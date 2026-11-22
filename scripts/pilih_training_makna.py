"""Seleksi arm training-makna-tingkat-v1 di valid (CPU), audit buta, kunci pilihan, buat kernel test.

Aturan dipatok di diskusi `2026-11-22-perbaikan-sisi-training-data.md` sebelum run valid:
- reproduksi demo di kernel yang sama harus lolos (62,66 / 68,83), kalau tidak seleksi berhenti
- gerbang angka (vs demo yang dijalankan di kernel yang sama): batas bawah CI 90% selisih corpus chrF++ Krama Alus > 0
  (2.000 paired bootstrap per grup), kehilangan ≤1 keputusan benar Krama Alus (juri valid), Ngoko/Krama turun ≤1 poin
- gerbang audit (27 Krama Alus valid, buta, berpasangan): pulih − regresi ≥ 2 (baris yang semua fakta intinya terjaga),
  dan jumlah lolos kedua sumbu (makna dan tingkat) tidak berkurang
- pilih chrF++ Krama Alus tertinggi → jumlah lolos dua sumbu → nllb600_base; tidak ada yang lolos → demo

Pemakaian:
  python scripts/pilih_training_makna.py fakta            # sebelum prediksi ada: lembar fakta inti 27 Krama Alus valid
  python scripts/pilih_training_makna.py seleksi          # setelah run valid (diulang sampai audit terisi, lalu mengunci)
  python scripts/pilih_training_makna.py audit-buta-test  # setelah run test
"""
import argparse, hashlib, json, random, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pilih_pool_makna import boot_lb, metrics  # noqa: E402  (fungsi yang sama dengan seleksi pool-makna)

ROOT = Path(__file__).resolve().parents[1]
K1 = ROOT / "kaggle/training_makna_tingkat_v1"
K2 = ROOT / "kaggle/training_makna_tingkat_v1_test"
LEVELS = ["ngoko", "krama", "krama_alus"]
SEED_AUDIT = 20261122
sha256 = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def frame(path):
    d = pd.read_csv(path)
    d["source_id"] = d["index"].astype(str) + ":" + d["level"]
    return d.rename(columns={"level": "requested", "hyp": "candidate", "javanese": "reference",
                             "juri_valid": "valid_judge_label"}).set_index("source_id")


def cek01(a, cols, allowed=("0", "1")):
    for c in cols:
        bad = ~a[c].astype(str).isin(allowed)
        if bad.any():
            sys.exit(f"lembar belum lengkap: kolom {c} kosong/tidak valid di {int(bad.sum())} baris")


def fakta(args):
    p = K1 / "fakta_ka_valid.csv"
    if p.exists():
        sys.exit(f"{p} sudah ada; jangan ditulis ulang setelah dikunci.")
    va = pd.read_csv(ROOT / "data/unggah_ungguh/split/valid.csv")
    ka = va[va.level == "krama_alus"][["group", "indonesian", "javanese"]].rename(columns={"javanese": "reference"})
    for c in ["fakta_inti", "bergantung_kata_oov", "referensi_bermasalah", "catatan"]:
        ka[c] = ""
    ka.to_csv(p, index=False, encoding="utf-8")
    print(f"Lembar fakta ditulis: {p} ({len(ka)} baris). Isi sebelum melihat prediksi apa pun.")


def seleksi(args):
    out, lock = K1 / "output", K1 / "pilihan.json"
    if lock.exists():
        sys.exit(f"pilihan sudah dikunci di {lock}.")
    man = json.loads((out / "manifest_valid.json").read_text(encoding="utf-8"))
    rep = man["gates"]["reproduksi_demo"]
    print("reproduksi demo:", rep)
    if not rep["ok"]:
        sys.exit("reproduksi demo tidak cocok; selidiki selisihnya sebelum seleksi (aturan diskusi).")
    print("gerbang 1.3B:", {k: v for k, v in man["gates"].items() if k != "reproduksi_demo"})
    if "konverter" in man:
        print("konverter (otomatis):", man["konverter"])
    arms = {}
    for name, info in man["arms"].items():
        f = out / f"valid_{name}.csv"
        assert sha256(f) == info["sha256"], f"{f} tidak cocok dengan manifest"
        arms[name] = frame(f)
    demo, md = arms["demo"], metrics(arms["demo"])
    tabel = {"demo": {**md, "lolos_angka": True}}
    for name, s in arms.items():
        if name == "demo":
            continue
        m = metrics(s); lb, mean = boot_lb(demo, s.loc[demo.index])
        ok = lb > 0 and m["KA_juri_benar"] >= md["KA_juri_benar"] - 1 and all(m[lv] >= md[lv] - 1 for lv in ["ngoko", "krama"])
        tabel[name] = {**m, "lb90": lb, "selisih_rerata": mean, "lolos_angka": bool(ok)}
    for k, v in tabel.items():
        print(f"{k:14s} {v}")
    lolos = [k for k, v in tabel.items() if k != "demo" and v["lolos_angka"]]

    fk = pd.read_csv(K1 / "fakta_ka_valid.csv", dtype=str).fillna("")
    cek01(fk, ["bergantung_kata_oov", "referensi_bermasalah"])
    if (fk.fakta_inti.str.strip() == "").any():
        sys.exit("fakta_ka_valid.csv: kolom fakta_inti belum lengkap")
    audit_k = pd.DataFrame()
    if lolos:
        ab, kunci = K1 / "audit_buta_valid.csv", K1 / "kunci_audit_buta_valid.csv"
        if not ab.exists():
            rng = random.Random(SEED_AUDIT); rows, keys = [], []
            ka_ids = demo.index[demo.requested == "krama_alus"]
            items = [(sid, sys_) for sid in ka_ids for sys_ in ["demo"] + lolos]
            rng.shuffle(items)
            for i, (sid, sys_) in enumerate(items):
                r = arms[sys_].loc[sid]; f = fk[fk.group.astype(int) == int(r.group)].iloc[0]
                rows.append({"kode": i + 1, "group": r.group, "indonesian": r.indonesian, "fakta_inti": f.fakta_inti,
                             "output": r.candidate, "semua_fakta_inti_terjaga": "", "tingkat_arah_tepat": "", "catatan": ""})
                keys.append({"kode": i + 1, "source_id": sid, "sistem": sys_})
            pd.DataFrame(rows).to_csv(ab, index=False, encoding="utf-8")
            pd.DataFrame(keys).to_csv(kunci, index=False, encoding="utf-8")
            print(f"\nAudit buta ditulis: {ab} ({len(rows)} output, sistem disembunyikan; kunci di {kunci.name}).\n"
                  "Isi semua_fakta_inti_terjaga (0/1) dan tingkat_arah_tepat (0/1/?), lalu jalankan seleksi lagi.")
            return
        a = pd.read_csv(ab, dtype=str).fillna("")
        cek01(a, ["semua_fakta_inti_terjaga"]); cek01(a, ["tingkat_arah_tepat"], ("0", "1", "?"))
        audit_k = a.merge(pd.read_csv(kunci, dtype={"kode": str}), on="kode")
        piv = audit_k.pivot(index="source_id", columns="sistem", values=["semua_fakta_inti_terjaga", "tingkat_arah_tepat"])
        mk = piv["semua_fakta_inti_terjaga"].astype(int)
        dua = (mk == 1) & (piv["tingkat_arah_tepat"] == "1")
        for name in lolos:
            pulih = int(((mk.demo == 0) & (mk[name] == 1)).sum()); regresi = int(((mk.demo == 1) & (mk[name] == 0)).sum())
            tabel[name].update({"pulih": pulih, "regresi": regresi, "lolos_dua_sumbu": int(dua[name].sum()),
                                "lolos_dua_sumbu_demo": int(dua.demo.sum())})
            tabel[name]["lolos_audit"] = bool(pulih - regresi >= 2 and dua[name].sum() >= dua.demo.sum())
            print(name, "audit:", {k: tabel[name][k] for k in ["pulih", "regresi", "lolos_dua_sumbu", "lolos_dua_sumbu_demo", "lolos_audit"]})
    final = [k for k in lolos if tabel[k].get("lolos_audit")]
    urut = lambda k: (tabel[k]["krama_alus"], tabel[k]["lolos_dua_sumbu"], k == "nllb600_base")
    win = max(final, key=urut) if final else "demo"
    pil = {"arm": win, "tabel_valid": tabel, "manifest_valid_sha256": sha256(out / "manifest_valid.json"),
           "fakta_sha256": sha256(K1 / "fakta_ka_valid.csv"), "dikunci": time.strftime("%Y-%m-%d %H:%M:%S"),
           "catatan_audit": "audit dikerjakan non-penutur Jawa (draf) kecuali dinyatakan lain di lembar audit"}
    if win != "demo":
        pil["model_sha256"] = man["arms"][win]["model_sha256"]
    lock.write_text(json.dumps(pil, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nPEMENANG valid: {win}. Dikunci di {lock}")
    if win != "demo":
        buat_kernel_test(pil)
    else:
        print("Demo dipertahankan; run test tidak diperlukan.")


def buat_kernel_test(pil):
    K2.mkdir(exist_ok=True)
    nb = json.loads((K1 / "training_makna_tingkat_v1.ipynb").read_text(encoding="utf-8"))
    code = [c for c in nb["cells"] if c["cell_type"] == "code"]
    assert "".join(code[0]["source"]).startswith('MODE = "valid"')
    code[0]["source"] = ['MODE = "test"\n', f"PILIHAN = {json.dumps({'arm': pil['arm'], 'model_sha256': pil['model_sha256']})}\n"]
    (K2 / "training_makna_tingkat_v1_test.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    meta = json.loads((K1 / "kernel-metadata.json").read_text(encoding="utf-8"))
    meta.update({"id": "mdevian/unggah-ungguh-training-makna-tingkat-v1-test", "title": "Unggah-Ungguh Training Makna Tingkat v1 Test",
                 "code_file": "training_makna_tingkat_v1_test.ipynb",
                 "kernel_sources": ["mdevian/unggah-ungguh-final-3lv-v2", "mdevian/unggah-ungguh-training-makna-tingkat-v1"]})
    (K2 / "kernel-metadata.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(f"Kernel test dibuat di {K2}. Push: cd \"{K2}\" && kaggle kernels push -p .")


def audit_buta_test(args):
    out = K2 / "output"
    b, p = (pd.read_csv(out / f"pred_{n}.csv") for n in ["demo", "pilihan"])
    assert (b["index"].values == p["index"].values).all()
    rng = random.Random(SEED_AUDIT); pilih = []
    for lv, n in [("krama_alus", 10), ("krama", 3), ("ngoko", 2)]:
        g = sorted(set(b.loc[b.level == lv, "group"]) - {x for x, _ in pilih})
        pilih += [(x, lv) for x in rng.sample(g, n)]
    rows, kunci = [], []
    for i, (g, lv) in enumerate(pilih):
        m = (b.group == g) & (b.level == lv)
        hb, hp = b.loc[m, "hyp"].iat[0], p.loc[m, "hyp"].iat[0]
        kiri = rng.random() < 0.5
        rows.append({"no": i + 1, "tingkat_diminta": lv, "indonesian": b.loc[m, "indonesian"].iat[0],
                     "output_A": hp if kiri else hb, "output_B": hb if kiri else hp,
                     "makna_A": "", "tingkat_A": "", "makna_B": "", "tingkat_B": "", "catatan": ""})
        kunci.append({"no": i + 1, "group": g, "A": "pilihan" if kiri else "demo", "identik": hb == hp})
    pd.DataFrame(rows).to_csv(K2 / "audit_buta_test.csv", index=False, encoding="utf-8")
    pd.DataFrame(kunci).to_csv(K2 / "kunci_audit_buta_test.csv", index=False, encoding="utf-8")
    print(f"audit_buta_test.csv dan kuncinya ditulis di {K2}. Skala: makna 0/1/2, tingkat 0/1/2.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("perintah", choices=["fakta", "seleksi", "audit-buta-test"])
    a = ap.parse_args()
    {"fakta": fakta, "seleksi": seleksi, "audit-buta-test": audit_buta_test}[a.perintah](a)
