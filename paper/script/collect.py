"""Kumpulkan SEMUA angka paper dari output eksperimen di kaggle/ -> data/results.json.

Aturan:
- chrF++/BLEU TEST dihitung ulang dari CSV prediksi (176 kalimat 3 tingkat) dengan sacrebleu bila CSV ada;
  bila tidak ada, diambil dari metrics JSON / log kernel dan ditandai sumber="kernel".
- Ketepatan tingkat = juri tetap (ensemble DistilBERT, scripts/juri_tetap.py) pada set kontrol
  (75 kalimat x 3 tingkat = 225 permintaan). Bila CSV kontrol tidak ada -> juri kernel, ditandai.
- Angka VALID dari manifest kernel (valid saja, test tidak dibaca).
Jalankan dari mana saja: python collect.py  (butuh models/juri_tetap atau dilatih ulang oleh skrip juri).
"""
import json, os, re, sys, glob
import numpy as np, pandas as pd, sacrebleu

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
K = os.path.join(ROOT, "kaggle")
OUT = os.path.join(os.path.dirname(__file__), "data", "results.json")
LV = ["ngoko", "krama", "krama_alus"]
os.chdir(ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def chrf(h, r): return sacrebleu.corpus_chrf(list(h), [list(r)], word_order=2).score
def bleu(h, r): return sacrebleu.corpus_bleu(list(h), [list(r)]).score


TEST = pd.read_csv("data/unggah_ungguh/split/test.csv")
TEST = TEST[TEST.level.isin(LV)].reset_index(drop=True)
assert len(TEST) == 176


def load_pred(path, col="hyp"):
    d = pd.read_csv(os.path.join(K, path))
    d = d[d.level.isin(LV)].reset_index(drop=True)
    assert len(d) == 176 and (d["index"].values == TEST["index"].values).all(), path
    return d[col].fillna("").astype(str).values


def scores(hyp):
    r = {"all": {"chrF++": chrf(hyp, TEST.javanese), "BLEU": bleu(hyp, TEST.javanese)}}
    for lv in LV:
        m = (TEST.level == lv).values
        r[lv] = {"chrF++": chrf(hyp[m], TEST.javanese[m]), "BLEU": bleu(hyp[m], TEST.javanese[m])}
    return r


def mean_scores(lst):
    out = {}
    for k in lst[0]:
        out[k] = {m: float(np.mean([s[k][m] for s in lst])) for m in lst[0][k]}
        out[k]["chrF++_std"] = float(np.std([s[k]["chrF++"] for s in lst], ddof=1)) if len(lst) > 1 else None
    return out


# ---------------------------------------------------------------- juri tetap
_J = {}
def judges():
    if not _J:
        import juri_tetap as J
        _J["J"] = J; _J["j"] = J.get_judges(J.load3("train"))
    return _J["J"], _J["j"]


def level_acc(control_paths):
    J, js = judges(); accs = []
    for p in control_paths:
        d = pd.read_csv(os.path.join(K, p)); d = d[d.requested.isin(LV)].reset_index(drop=True)
        assert len(d) == 225, (p, len(d))
        P = np.mean([J.probs(j, d.output.fillna("").astype(str).tolist()) for j in js], 0)
        ok = np.array([LV[i] for i in P.argmax(-1)]) == d.requested.values
        accs.append({"all": ok.mean(), **{lv: ok[d.requested.values == lv].mean() for lv in LV}})
    return {k: float(np.mean([a[k] for a in accs])) for k in accs[0]}


def from_metrics(path, decode):
    d = json.load(open(os.path.join(K, path), encoding="utf-8"))["test"][decode]
    t = d["test"]
    return ({k: {"chrF++": t[k]["chrF++"], "BLEU": t[k]["BLEU"]} for k in ["all"] + LV},
            {"all": d["control"]["level_accuracy"], **d["control"]["per_level_accuracy"]})


def from_log(path, variant, decode):
    s = open(os.path.join(K, path), encoding="utf-8").read().replace("\\n", "\n")
    m = re.search(rf"{re.escape(variant)} {re.escape(decode)} TEST (\{{[^}}]*\}}) ctrl ([0-9.]+) (\{{[^}}]*\}})", s)
    t = eval(m.group(1)); c = eval(m.group(3))  # baris log kernel milik sendiri, format dict Python
    return {"all": {"chrF++": t["chrF++"], "BLEU": t["BLEU"]}}, {"all": float(m.group(2)), **c}


# ---------------------------------------------------------------- paired bootstrap vs demo (per grup)
def boot(h_a, h_b, B=1000, seed=0):
    rng = np.random.default_rng(seed); g = TEST.group.values; ug = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in ug}; R = TEST.javanese.values; diff = []
    for _ in range(B):
        s = np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])
        diff.append(chrf(h_a[s], R[s]) - chrf(h_b[s], R[s]))
    diff = np.array(diff)
    return {"ci90": [float(np.percentile(diff, 5)), float(np.percentile(diff, 95))]}


# ---------------------------------------------------------------- baris TEST
F3M, F3N = "finetune_3lv_m2m100/output", "finetune_3lv_nllb/output"
SEEDS = ["s42", "s13", "s2024"]
ROWS = [
    # id, keluarga, nama, pred(s), kolom, kontrol(s), catatan
    ("copy", "Baseline", "Salin input (tanpa terjemahan)", ["baseline_zeroshot/output/predictions.csv"], "hyp_copy", None, ""),
    ("zs_m2m", "Zero-shot", "M2M100-418M zero-shot", ["baseline_zeroshot/output/predictions.csv"], "hyp_m2m100_418m", None, ""),
    ("zs_nllb", "Zero-shot", "NLLB-200-600M zero-shot", ["baseline_zeroshot/output/predictions.csv"], "hyp_nllb_600m", None, ""),
    ("ft_notag", "Fine-tuning", "M2M100 fine-tuning tanpa tag", ["finetune_m2m100/output/pred_m2m100_418m_no_tag.csv"], "hyp",
     ["finetune_m2m100/output/control_m2m100_418m_no_tag.csv"], "4 tingkat"),
    ("ft_tag_m2m", "Fine-tuning", "M2M100 + source tag", ["finetune_m2m100/output/pred_m2m100_418m_atomic.csv"], "hyp",
     ["finetune_m2m100/output/control_m2m100_418m_atomic.csv"], "4 tingkat"),
    ("ft_tag_nllb", "Fine-tuning", "NLLB-600M + source tag", ["finetune_nllb/output/pred_nllb_600m_atomic.csv"], "hyp",
     ["finetune_nllb/output/control_nllb_600m_atomic.csv"], "4 tingkat"),
    ("r3_base", "Fine-tuning", "M2M100 + tag (3 tingkat)", [f"{F3M}/pred3_m2m100_418m_atomic_base_{s}.csv" for s in SEEDS], "hyp",
     [f"{F3M}/control3_m2m100_418m_atomic_base_{s}.csv" for s in SEEDS], "3 seed"),
    ("r3_initos", "Fine-tuning", "M2M100 + tag + init + OS", [f"{F3M}/pred3_m2m100_418m_atomic_init_os_{s}.csv" for s in SEEDS], "hyp",
     [f"{F3M}/control3_m2m100_418m_atomic_init_os_{s}.csv" for s in SEEDS], "3 seed"),
    ("r3_nllb", "Fine-tuning", "NLLB-600M + tag + init + OS + MT", [f"{F3N}/pred3_nllb_600m_atomic_init_os_mt_{s}.csv" for s in SEEDS], "hyp",
     [f"{F3N}/control3_nllb_600m_atomic_init_os_mt_{s}.csv" for s in SEEDS], "3 seed"),
    ("v2_beam", "Resep v2", "M2M100 resep v2, beam 4", ["final_3lv_v2/output/pred_m2m_base_beam4.csv"], "hyp",
     ["final_3lv_v2/output/control_m2m_base_beam4.csv"], ""),
    ("demo", "Resep v2", "M2M100 resep v2 + rerank (DEMO)", ["final_3lv_v2/output/pred_m2m_base_rerank2.0.csv"], "hyp",
     ["final_3lv_v2/output/control_m2m_base_rerank2.0.csv"], ""),
    ("v2_na_ls", "Resep v2", "v2 + data Ngoko Alus + label smoothing + rerank", ["final_3lv_v2/output/pred_m2m_na_ls_rerank1.0.csv"], "hyp",
     ["final_3lv_v2/output/control_m2m_na_ls_rerank1.0.csv"], ""),
    ("v2_nllb", "Resep v2", "NLLB-600M resep v2 + rerank", ["final_3lv_v2/output/pred_nllb_na_rerank2.0.csv"], "hyp",
     ["final_3lv_v2/output/control_nllb_na_rerank2.0.csv"], ""),
    ("pe", "Pasca-proses", "Demo + post-editing leksikon KA", ["final_3lv_postedit/output/pred_rerank2.0_pe_trig_gain.csv"], "hyp",
     ["final_3lv_postedit/output/control_rerank2.0_pe_trig_gain.csv"], ""),
    ("st", "Data tambahan", "Self-training (pseudo KA) + rerank", ["final_3lv_selftrain/output/pred_st_train_rerank0.5.csv"], "hyp",
     ["final_3lv_selftrain/output/control_st_train_rerank0.5.csv"], ""),
]
METRIC_ROWS = [
    ("aug", "Data tambahan", "+ KA sintetis leksikon + rerank", "final_3lv_merge/output/metrics_aug.json", "rerank4.0",
     "final_3lv_merge/output/control_aug_rerank4.0.csv"),
    ("nusax", "Data tambahan", "+ NusaX (Ngoko) + rerank", "final_3lv_merge/output/metrics_nusax.json", "rerank2.0",
     "final_3lv_merge/output/control_nusax_rerank2.0.csv"),
    ("aug_nusax", "Data tambahan", "+ KA sintetis + NusaX + rerank", "final_3lv_merge/output/metrics_aug_nusax.json", "rerank0.5",
     "final_3lv_merge/output/control_aug_nusax_rerank0.5.csv"),
]
LOG_ROWS = [
    ("cendol_A", "Model lain", "Cendol mT5-base, nama tingkat di prompt", "final_3lv_cendol_tune/output/kernel_log_v1.json", "A_lvprompt", "beam4"),
    ("cendol_B", "Model lain", "Cendol mT5-base + OS + LS + rerank", "final_3lv_cendol_tune/output/kernel_log_v1.json", "B_os5_ls", "rerank8.0"),
]


def main():
    res = {"test": {}, "valid": {}, "eda": {}, "sacrebleu": sacrebleu.__version__}
    hyps = {}
    for rid, fam, name, preds, col, ctrl, note in ROWS:
        hs = [load_pred(p, col) for p in preds]
        sc = mean_scores([scores(h) for h in hs])
        hyps[rid] = hs[0]
        res["test"][rid] = {"family": fam, "name": name, "scores": sc, "note": note, "source": preds,
                            "level_acc": level_acc(ctrl) if ctrl else None, "judge": "tetap" if ctrl else None}
        print(rid, round(sc["all"]["chrF++"], 2))
    for rid, fam, name, path, dec, ctrl in METRIC_ROWS:
        sc, _ = from_metrics(path, dec)
        res["test"][rid] = {"family": fam, "name": name, "scores": sc, "note": "", "source": [path],
                            "level_acc": level_acc([ctrl]), "judge": "tetap"}
    for rid, fam, name, path, var, dec in LOG_ROWS:
        sc, acc = from_log(path, var, dec)
        res["test"][rid] = {"family": fam, "name": name, "scores": sc, "note": "", "source": [path],
                            "level_acc": acc, "judge": "kernel"}
    for rid, h in hyps.items():
        if rid != "demo":
            res["test"][rid]["boot_vs_demo"] = boot(h, hyps["demo"])
    # cek: demo harus sama dengan angka tercatat kernel v2
    d = res["test"]["demo"]["scores"]["all"]
    assert abs(d["chrF++"] - 58.62) < 0.01 and abs(d["BLEU"] - 33.17) < 0.01, d

    # ---------------------------------------------------- VALID
    ms = json.load(open(f"{K}/multiseed_demo_v2/output/manifest.json", encoding="utf-8"))
    tm = json.load(open(f"{K}/training_makna_tingkat_v1/output/manifest_valid.json", encoding="utf-8"))
    pl = json.load(open(f"{K}/pool_makna_v1/pilihan.json", encoding="utf-8"))["tabel_valid"]
    lc = json.load(open(f"{K}/learning_curve_v2/analisis_lokal.json", encoding="utf-8"))
    kt = json.load(open(f"{K}/lc_kontrol_langkah_v1/output/manifest.json", encoding="utf-8"))
    kta = json.load(open(f"{K}/lc_kontrol_langkah_v1/analisis_lokal.json", encoding="utf-8"))
    v = res["valid"]
    v["demo"] = {"name": "Demo (v2, epoch 14, rerank)", **tm["arms"]["demo"]}
    for s in ["seed42", "seed43", "seed44"]:
        pl_ = ms["runs"][s]["valid_rerank"]["per_level"]
        v[f"ms_{s}"] = {"name": f"Ulang resep demo, {s}", **{k: pl_[k]["chrF++"] for k in ["all"] + LV},
                        "best_epoch": ms["runs"][s]["best_epoch"]}
    v["nllb600"] = {"name": "NLLB-600M paket resep demo", **{k: tm["arms"]["nllb600_base"][k] for k in ["all"] + LV},
                    "KA_juri_benar": tm["arms"]["nllb600_base"]["KA_juri_benar"]}
    v["pool"] = {"name": "Pool beam8+beam16, rerank", **pl["gabung_mu0"]}
    v["konverter"] = {"name": "Konverter Krama->KA (input referensi Krama)", **tm["konverter"]}
    v["zeroshot_gate"] = {k: tm["gates"][k] for k in ["zeroshot_ngoko_nllb600_base", "zeroshot_ngoko_nllb13_base", "zeroshot_p5_13_minus_600"]}
    v["learning_curve"] = {k: val for k, val in lc.items()}
    v["kontrol_langkah"] = {"gap": kt["gap_akhir"], "boot": {k: kta[k] for k in ["f050_s0", "f050_s1"]},
                            "akhir": {n: r["akhir"]["valid_rerank"]["per_level"] for n, r in kt["runs"].items()},
                            "juri_tetap": {n: kta[n]["juri_tetap_lv"] for n in kta}}

    # ---------------------------------------------------- EDA
    tr = pd.read_csv("data/unggah_ungguh/translation.csv")
    e = res["eda"]
    e["rows_total"] = len(tr)
    e["label_counts"] = tr.label.value_counts().sort_index().to_dict()
    for sp in ["train", "valid", "test"]:
        d = pd.read_csv(f"data/unggah_ungguh/split/{sp}.csv")
        e[sp] = d.level.value_counts().to_dict()
        e[sp + "_groups"] = int(d.group.nunique())
    d3 = pd.concat([pd.read_csv(f"data/unggah_ungguh/split/{sp}.csv") for sp in ["train", "valid", "test"]])
    d3 = d3[d3.level.isin(LV)]
    e["len_words"] = {lv: d3[d3.level == lv].javanese.str.split().str.len().describe()[["mean", "50%", "max"]].round(2).to_dict() for lv in LV}
    e["len_words_id"] = float(d3.indonesian.str.split().str.len().mean())
    e["n3"] = int(len(d3))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(res, open(OUT, "w", encoding="utf-8"), indent=1, ensure_ascii=False, default=float)
    print("tulis", OUT)


if __name__ == "__main__":
    main()
