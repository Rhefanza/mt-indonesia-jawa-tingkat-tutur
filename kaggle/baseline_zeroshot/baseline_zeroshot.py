"""Baseline zero-shot Indonesia -> Jawa pada test split Unggah-Ungguh.

Sistem: copy baseline, M2M100-418M zero-shot, NLLB-200-distilled-600M zero-shot.
Zero-shot tidak punya kontrol tingkat, jadi satu output per kalimat Indonesia
dinilai terhadap referensi tiap tingkat.
Output: /kaggle/working/predictions.csv, /kaggle/working/metrics.json
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "sentencepiece"], check=False)

import glob, json, re, time, random
import pandas as pd, torch, sacrebleu
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          AutoModelForSequenceClassification)

dev = "cuda" if torch.cuda.is_available() else "cpu"
print("device", dev, torch.cuda.get_device_name(0) if dev == "cuda" else "", flush=True)
LEVELS = ["ngoko", "ngoko_alus", "krama", "krama_alus"]

test_path = glob.glob("/kaggle/input/**/test.csv", recursive=True)[0]
test = pd.read_csv(test_path)
print("test", test_path, len(test), dict(test["level"].value_counts()), flush=True)
src = sorted(test["indonesian"].unique())
print("kalimat Indonesia unik:", len(src), flush=True)

MODELS = {
    "m2m100_418m": ("facebook/m2m100_418M", "id", "jv"),
    "nllb_600m": ("facebook/nllb-200-distilled-600M", "ind_Latn", "jav_Latn"),
}


def translate(repo, src_lang, tgt_lang, sents, beams=4, bs=32):
    tok = AutoTokenizer.from_pretrained(repo, src_lang=src_lang)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo).to(dev).eval()
    bos = tok.get_lang_id(tgt_lang) if hasattr(tok, "get_lang_id") else tok.convert_tokens_to_ids(tgt_lang)
    out = []
    for i in range(0, len(sents), bs):
        enc = tok(sents[i:i + bs], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
        with torch.no_grad(), torch.autocast(dev, dtype=torch.float16, enabled=dev == "cuda"):
            g = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams, max_new_tokens=64)
        out += tok.batch_decode(g, skip_special_tokens=True)
    del model; torch.cuda.empty_cache()
    return out


hyp = {"copy": dict(zip(src, src))}
timing = {}
for name, (repo, s, t) in MODELS.items():
    t0 = time.time()
    hyp[name] = dict(zip(src, translate(repo, s, t, src)))
    timing[name] = round(time.time() - t0, 1)
    print(name, "selesai", timing[name], "s", flush=True)

for name in hyp:
    test[f"hyp_{name}"] = test["indonesian"].map(hyp[name])


def bleu(h, r): return sacrebleu.corpus_bleu(h, [r]).score
def chrf(h, r): return sacrebleu.corpus_chrf(h, [r], word_order=2).score


def boot_ci(h, r, n=1000, seed=0):
    rng = random.Random(seed); idx = list(range(len(h))); vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(chrf([h[i] for i in s], [r[i] for i in s]))
    vals.sort()
    return round(vals[int(.025 * n)], 2), round(vals[int(.975 * n)], 2)


metrics = {"n_test": len(test), "timing_s": timing, "systems": {}}
for name in hyp:
    m = {}
    for lv in LEVELS + ["all"]:
        d = test if lv == "all" else test[test["level"] == lv]
        h, r = d[f"hyp_{name}"].tolist(), d["javanese"].tolist()
        m[lv] = {"n": len(d), "BLEU": round(bleu(h, r), 2), "chrF++": round(chrf(h, r), 2),
                 "chrF++_CI95": boot_ci(h, r)}
    metrics["systems"][name] = m

# ---- tingkat output menurut classifier dan penanda pronomina ----
CLF = "JavaneseHonorifics/Unggah-Ungguh-Javanese-Distilbert-Classifier"
try:
    ctok = AutoTokenizer.from_pretrained(CLF)
    cmod = AutoModelForSequenceClassification.from_pretrained(CLF).to(dev).eval()
    id2label = cmod.config.id2label
    print("classifier id2label:", id2label, flush=True)

    def classify(sents):
        preds = []
        for i in range(0, len(sents), 64):
            enc = ctok(sents[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=64).to(dev)
            with torch.no_grad():
                preds += cmod(**enc).logits.argmax(-1).tolist()
        return preds

    ref_pred = classify(test["javanese"].tolist())
    metrics["classifier"] = {"id2label": {int(k): v for k, v in id2label.items()},
                             "acc_on_references": round(sum(p == l for p, l in zip(ref_pred, test["label"])) / len(test), 4)}
    for name in ["m2m100_418m", "nllb_600m"]:
        p = classify(src_out := [hyp[name][s] for s in src])
        metrics["classifier"][f"pred_dist_{name}"] = dict(pd.Series(p).map(lambda x: id2label[x]).value_counts())
except Exception as e:
    metrics["classifier"] = {"error": repr(e)}
    print("classifier gagal:", e, flush=True)

MARK = {"ngoko": ["kowe", "aku", "awakmu"], "krama": ["sampeyan", "kula", "panjenengan", "menika", "punika"],
        "alus": ["panjenengan", "panjenenganipun", "dalem", "adalem", "dhahar", "tindak", "kagungan"]}
toks = lambda s: set(re.findall(r"[\w'\-]+", str(s).lower()))
for name in ["m2m100_418m", "nllb_600m"]:
    outs = [toks(hyp[name][s]) for s in src]
    metrics.setdefault("markers", {})[name] = {k: sum(bool(o & set(v)) for o in outs) for k, v in MARK.items()}
metrics["markers"]["n_sentences"] = len(src)

test.to_csv("/kaggle/working/predictions.csv", index=False)
json.dump(metrics, open("/kaggle/working/metrics.json", "w"), indent=1, ensure_ascii=False, default=str)
print(json.dumps(metrics, indent=1, ensure_ascii=False, default=str))
for s in src[:8]:
    print("\nID  :", s, "\nM2M :", hyp["m2m100_418m"][s], "\nNLLB:", hyp["nllb_600m"][s])
