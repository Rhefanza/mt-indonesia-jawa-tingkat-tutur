"""Fine-tuning Indonesia -> Jawa dengan kontrol tingkat tutur (Unggah-Ungguh, group split).

Satu kernel = satu model (MODEL_KEY). Varian:
  no_tag     : tanpa tag (pembanding)
  atomic     : 4 tag  <ngoko> <ngoko_alus> <krama> <krama_alus>
  factorized : 2 sumbu <base_ngoko>/<base_krama> + <honor_biasa>/<honor_alus>
Evaluasi test: BLEU, chrF++ per tingkat (+CI95 bootstrap), control test
(setiap kalimat Indonesia test diminta ke 4 tingkat, dinilai classifier yang
dilatih ulang di train split), confusion matrix, penanda pronomina.
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "sentencepiece"], check=False)

import glob, json, time, random, re, os
import numpy as np, pandas as pd, torch, sacrebleu
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

MODEL_KEY = "nllb_600m"
CFG = {"nllb_600m": ("facebook/nllb-200-distilled-600M", "ind_Latn", "jav_Latn"),
       "m2m100_418m": ("facebook/m2m100_418M", "id", "jv")}
VARIANTS = ["no_tag", "atomic", "factorized"]
EPOCHS, BS, LR, WARMUP, SEED, MAXLEN = 8, 16, 5e-5, 0.1, 42, 64
OUT = "/kaggle/working"
dev = "cuda"
LEVELS = ["ngoko", "ngoko_alus", "krama", "krama_alus"]
LID = {l: i for i, l in enumerate(LEVELS)}
ATOM = {l: f"<{l}>" for l in LEVELS}
FACT = {"ngoko": "<base_ngoko><honor_biasa>", "ngoko_alus": "<base_ngoko><honor_alus>",
        "krama": "<base_krama><honor_biasa>", "krama_alus": "<base_krama><honor_alus>"}
SPECIAL = {"no_tag": [], "atomic": list(ATOM.values()),
           "factorized": ["<base_ngoko>", "<base_krama>", "<honor_biasa>", "<honor_alus>"]}
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


find = lambda n: glob.glob(f"/kaggle/input/**/{n}.csv", recursive=True)[0]
tr, va, te = (pd.read_csv(find(n)) for n in ["train", "valid", "test"])
log("data", len(tr), len(va), len(te), torch.cuda.get_device_name(0))


def src_text(variant, ind, level):
    if variant == "no_tag":
        return ind
    return f"{ATOM[level] if variant == 'atomic' else FACT[level]} {ind}"


def bleu(h, r): return sacrebleu.corpus_bleu(h, [r]).score
def chrf(h, r): return sacrebleu.corpus_chrf(h, [r], word_order=2).score


def boot_ci(h, r, n=1000, seed=0):
    rng = random.Random(seed); idx = range(len(h)); vals = []
    for _ in range(n):
        s = [rng.choice(idx) for _ in idx]
        vals.append(chrf([h[i] for i in s], [r[i] for i in s]))
    vals.sort()
    return [round(vals[int(.025 * n)], 2), round(vals[int(.975 * n)], 2)]


# ------------------------------------------------------------------ classifier
def train_classifier():
    for base in ["w11wo/javanese-distilbert-small", "xlm-roberta-base"]:
        try:
            ctok = AutoTokenizer.from_pretrained(base)
            cmod = AutoModelForSequenceClassification.from_pretrained(base, num_labels=4).to(dev)
            break
        except Exception as e:
            log("classifier base gagal", base, e)
    seed_all(SEED)
    opt = torch.optim.AdamW(cmod.parameters(), lr=5e-5)
    X, y = tr["javanese"].tolist(), tr["label"].tolist()
    for ep in range(4):
        cmod.train(); order = list(range(len(X))); random.shuffle(order)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            enc = ctok([X[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
            loss = cmod(**enc, labels=torch.tensor([y[j] for j in b], device=dev)).loss
            opt.zero_grad(); loss.backward(); opt.step()

    def classify(sents):
        cmod.eval(); out = []
        for i in range(0, len(sents), 64):
            enc = ctok(sents[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
            with torch.no_grad():
                out += cmod(**enc).logits.argmax(-1).tolist()
        return out

    acc = {n: round(float(np.mean(np.array(classify(d["javanese"].tolist())) == d["label"].values)), 4)
           for n, d in [("valid", va), ("test", te)]}
    log("classifier", base, "acc pada referensi", acc)
    return classify, {"base": base, "acc_on_references": acc}


classify, clf_info = train_classifier()

MARK = {"ngoko": {"kowe", "aku", "awakmu"}, "krama": {"sampeyan", "kula", "menika", "punika", "boten"},
        "alus": {"panjenengan", "panjenenganipun", "dalem", "adalem", "dhahar", "tindak", "kagungan", "sare"}}
toks = lambda s: set(re.findall(r"[\w'\-]+", str(s).lower()))


# ------------------------------------------------------------------ MT
def load(variant):
    repo, sl, tl = CFG[MODEL_KEY]
    tok = AutoTokenizer.from_pretrained(repo, src_lang=sl, tgt_lang=tl)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    if SPECIAL[variant]:
        tok.add_tokens(SPECIAL[variant], special_tokens=True)
        model.resize_token_embeddings(len(tok))
    bos = tok.get_lang_id(tl) if hasattr(tok, "get_lang_id") else tok.convert_tokens_to_ids(tl)
    return tok, model.to(dev), bos


def translate(tok, model, bos, srcs, beams=4, bs=48):
    model.eval(); out = [None] * len(srcs)
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([srcs[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams, max_new_tokens=MAXLEN)
        for j, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
            out[j] = t
    return out


def hyps_for(variant, tok, model, bos, d, beams):
    if variant == "no_tag":
        u = sorted(d["indonesian"].unique())
        m = dict(zip(u, translate(tok, model, bos, u, beams)))
        return [m[s] for s in d["indonesian"]]
    return translate(tok, model, bos, [src_text(variant, s, l) for s, l in zip(d["indonesian"], d["level"])], beams)


def run(variant):
    seed_all(SEED)
    tok, model, bos = load(variant)
    pairs = [(src_text(variant, s, l), t) for s, l, t in zip(tr["indonesian"], tr["level"], tr["javanese"])]
    v0 = chrf(hyps_for(variant, tok, model, bos, va, 1), va["javanese"].tolist())
    log(variant, "valid chrF++ sebelum training", round(v0, 2))
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * ((len(pairs) + BS - 1) // BS)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        model.train(); random.shuffle(pairs); tot = 0
        for i in range(0, len(pairs), BS):
            b = pairs[i:i + BS]
            enc = tok([s for s, _ in b], text_target=[t for _, t in b], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAXLEN).to(dev)
            enc["labels"][enc["labels"] == tok.pad_token_id] = -100
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**enc).loss
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item() * len(b)
        v = chrf(hyps_for(variant, tok, model, bos, va, 1), va["javanese"].tolist())
        hist.append({"epoch": ep + 1, "train_loss": round(tot / len(pairs), 4), "valid_chrF++": round(v, 2)})
        log(variant, hist[-1])
        if v > best:
            best = v; best_state = {k: x.detach().to("cpu", copy=True) for k, x in model.state_dict().items()}
    model.load_state_dict(best_state)
    best_ep = max(hist, key=lambda h: h["valid_chrF++"])["epoch"]

    # ---- test: referensi
    hyp = hyps_for(variant, tok, model, bos, te, 4)
    res = {"variant": variant, "valid_chrF++_before": round(v0, 2), "best_epoch": best_ep, "history": hist, "test": {}}
    for lv in LEVELS + ["all"]:
        m = te["level"] == lv if lv != "all" else te["level"].notna()
        h, r = [x for x, k in zip(hyp, m) if k], te.loc[m, "javanese"].tolist()
        res["test"][lv] = {"n": len(r), "BLEU": round(bleu(h, r), 2), "chrF++": round(chrf(h, r), 2), "chrF++_CI95": boot_ci(h, r)}
    pred = te.copy(); pred["hyp"] = hyp
    pred.to_csv(f"{OUT}/pred_{MODEL_KEY}_{variant}.csv", index=False)

    # ---- control test: tiap kalimat Indonesia diminta ke 4 tingkat
    u = sorted(te["indonesian"].unique())
    req = [(s, l) for s in u for l in LEVELS]
    if variant == "no_tag":
        m = dict(zip(u, translate(tok, model, bos, u)))
        outs = [m[s] for s, _ in req]
    else:
        outs = translate(tok, model, bos, [src_text(variant, s, l) for s, l in req])
    p = classify(outs)
    conf = np.zeros((4, 4), int)
    for (_, l), q in zip(req, p):
        conf[LID[l], q] += 1
    res["control"] = {"n": len(req), "level_accuracy": round(float(np.trace(conf) / conf.sum()), 4),
                      "per_level_accuracy": {l: round(float(conf[i, i] / conf[i].sum()), 4) for i, l in enumerate(LEVELS)},
                      "confusion_rows_requested_cols_predicted": conf.tolist(),
                      "distinct_outputs_per_sentence": round(float(np.mean([len({o for (s2, _), o in zip(req, outs) if s2 == s}) for s in u])), 2),
                      "markers_by_requested_level": {l: {k: sum(bool(toks(o) & v) for (s, l2), o in zip(req, outs) if l2 == l)
                                                         for k, v in MARK.items()} for l in LEVELS}}
    pd.DataFrame({"indonesian": [s for s, _ in req], "requested": [l for _, l in req], "output": outs,
                  "classified": [LEVELS[q] for q in p]}).to_csv(f"{OUT}/control_{MODEL_KEY}_{variant}.csv", index=False)
    log(variant, "test all", res["test"]["all"], "control acc", res["control"]["level_accuracy"])
    json.dump(res, open(f"{OUT}/metrics_{MODEL_KEY}_{variant}.json", "w"), indent=1)
    del model, best_state, opt; torch.cuda.empty_cache()
    return res


summary = {"model": MODEL_KEY, "hparams": {"epochs": EPOCHS, "bs": BS, "lr": LR, "warmup": WARMUP, "seed": SEED, "maxlen": MAXLEN},
           "classifier": clf_info, "runs": {}}
for v in VARIANTS:
    try:
        r = run(v)
        summary["runs"][v] = {"best_epoch": r["best_epoch"], "valid_before": r["valid_chrF++_before"],
                              "test": {k: (x["chrF++"], x["BLEU"]) for k, x in r["test"].items()},
                              "control_acc": r["control"]["level_accuracy"],
                              "control_per_level": r["control"]["per_level_accuracy"]}
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["runs"][v] = {"error": repr(e)}
    json.dump(summary, open(f"{OUT}/summary_{MODEL_KEY}.json", "w"), indent=1)
log("SELESAI"); print(json.dumps(summary, indent=1))
