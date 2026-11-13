"""Ronde 3 (multi-seed) dari ronde 2. Konfigurasi dipilih per model lewat RUN_VARIANTS; seed 42, 13, 2024.

Ronde 2: memperbaiki kontrol sumbu hormat (alus). Semua varian memakai 4 atomic tag.

  atomic_base       : sama seperti ronde 1, 10 epoch (pembanding adil)
  atomic_init       : embedding tag diinisialisasi dari rata-rata embedding kata penanda
  atomic_init_os    : + oversampling contoh Ngoko Alus dan Krama Alus (x2)
  atomic_init_os_mt : + multi-task JV->JV pada sumbu hormat (ngoko<->ngoko_alus, krama<->krama_alus),
                      dari train split saja
Evaluasi sama dengan ronde 1, ditambah akurasi classifier per kelas dan tingkat output identik antar tingkat.
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "sentencepiece"], check=False)

import glob, json, time, random, re
import numpy as np, pandas as pd, torch, sacrebleu
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

MODEL_KEY = "m2m100_418m"
CFG = {"nllb_600m": ("facebook/nllb-200-distilled-600M", "ind_Latn", "jav_Latn"),
       "m2m100_418m": ("facebook/m2m100_418M", "id", "jv")}
VARIANTS = {"atomic_base": dict(init=False, os=False, mt=False),
            "atomic_init": dict(init=True, os=False, mt=False),
            "atomic_init_os": dict(init=True, os=True, mt=False),
            "atomic_init_os_mt": dict(init=True, os=True, mt=True)}
EPOCHS, BS, LR, WARMUP, SEED, MAXLEN = 10, 16, 5e-5, 0.1, 42, 64
SEEDS = [42, 13, 2024]
RUN_VARIANTS = {"nllb_600m": ["atomic_base", "atomic_init_os_mt"], "m2m100_418m": ["atomic_base", "atomic_init_os"]}
OUT = "/kaggle/working"
dev = "cuda"
LEVELS = ["ngoko", "ngoko_alus", "krama", "krama_alus"]
LID = {l: i for i, l in enumerate(LEVELS)}
TAG = {l: f"<{l}>" for l in LEVELS}
INIT_WORDS = {"ngoko": "kowe aku mangan lunga turu", "ngoko_alus": "panjenengane adalem dhahar tindak sare",
              "krama": "sampeyan kula nedha kesah tilem", "krama_alus": "panjenenganipun dalem dhahar tindak sare"}
HONOR_PAIRS = [("ngoko", "ngoko_alus"), ("ngoko_alus", "ngoko"), ("krama", "krama_alus"), ("krama_alus", "krama")]
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


find = lambda n: glob.glob(f"/kaggle/input/**/{n}.csv", recursive=True)[0]
tr, va, te = (pd.read_csv(find(n)) for n in ["train", "valid", "test"])
log("data", len(tr), len(va), len(te), torch.cuda.get_device_name(0))
tag_src = lambda ind, level: f"{TAG[level]} {ind}"
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
    base = "w11wo/javanese-distilbert-small"
    ctok = AutoTokenizer.from_pretrained(base)
    cmod = AutoModelForSequenceClassification.from_pretrained(base, num_labels=4).to(dev)
    seed_all(SEED)
    opt = torch.optim.AdamW(cmod.parameters(), lr=5e-5)
    X, y = tr["javanese"].tolist(), tr["label"].tolist()
    for _ in range(4):
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

    info = {"base": base}
    for n, d in [("valid", va), ("test", te)]:
        p = np.array(classify(d["javanese"].tolist())); y_ = d["label"].values
        info[f"acc_{n}"] = round(float((p == y_).mean()), 4)
        info[f"per_class_acc_{n}"] = {l: round(float((p[y_ == i] == i).mean()), 4) for i, l in enumerate(LEVELS)}
    log("classifier", info)
    return classify, info


classify, clf_info = train_classifier()
MARK = {"ngoko": {"kowe", "aku", "awakmu"}, "krama": {"sampeyan", "kula", "menika", "punika", "boten"},
        "alus": {"panjenengan", "panjenenganipun", "dalem", "adalem", "dhahar", "tindak", "kagungan", "sare"}}
toks = lambda s: set(re.findall(r"[\w'\-]+", str(s).lower()))


# ------------------------------------------------------------------ MT
def load(init):
    repo, sl, tl = CFG[MODEL_KEY]
    tok = AutoTokenizer.from_pretrained(repo, src_lang=sl, tgt_lang=tl)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    tok.add_tokens(list(TAG.values()), special_tokens=True)
    model.resize_token_embeddings(len(tok))
    if init:
        emb = model.get_input_embeddings().weight
        with torch.no_grad():
            for l, words in INIT_WORDS.items():
                ids = tok(words, add_special_tokens=False)["input_ids"]
                emb[tok.convert_tokens_to_ids(TAG[l])] = emb[ids].mean(0)
    bos = tok.get_lang_id(tl) if hasattr(tok, "get_lang_id") else tok.convert_tokens_to_ids(tl)
    return tok, model.to(dev), bos, sl, tl


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


def hyps_for(tok, model, bos, d, beams):
    return translate(tok, model, bos, [tag_src(s, l) for s, l in zip(d["indonesian"], d["level"])], beams)


def honor_pairs():
    pairs = []
    for _, g in tr.groupby("group"):
        jv = dict(zip(g["level"], g["javanese"]))
        for a, b in HONOR_PAIRS:
            if a in jv and b in jv:
                pairs.append((f"{TAG[b]} {jv[a]}", jv[b]))
    return pairs


def run(name, flags, seed):
    seed_all(seed)
    tok, model, bos, sl, tl = load(flags["init"])
    id_pairs = [(tag_src(s, l), t) for s, l, t in zip(tr["indonesian"], tr["level"], tr["javanese"])]
    if flags["os"]:
        id_pairs += [(tag_src(s, l), t) for s, l, t in zip(tr["indonesian"], tr["level"], tr["javanese"]) if l.endswith("alus")]
    jv_pairs = honor_pairs() if flags["mt"] else []
    log(name, "pairs ID->JV", len(id_pairs), "JV->JV", len(jv_pairs))

    def make_batches():
        a, b = id_pairs[:], jv_pairs[:]
        random.shuffle(a); random.shuffle(b)
        bt = [(sl, a[i:i + BS]) for i in range(0, len(a), BS)] + [(tl, b[i:i + BS]) for i in range(0, len(b), BS)]
        random.shuffle(bt); return bt

    v0 = chrf(hyps_for(tok, model, bos, va, 1), va["javanese"].tolist())
    n_batches = len(make_batches())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = EPOCHS * n_batches
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, hist = -1, None, []
    for ep in range(EPOCHS):
        model.train(); tot = n = 0
        for src_lang, b in make_batches():
            tok.src_lang = src_lang
            enc = tok([s for s, _ in b], text_target=[t for _, t in b], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAXLEN).to(dev)
            enc["labels"][enc["labels"] == tok.pad_token_id] = -100
            with torch.autocast("cuda", dtype=torch.float16):
                loss = model(**enc).loss
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item() * len(b); n += len(b)
        tok.src_lang = sl
        v = chrf(hyps_for(tok, model, bos, va, 1), va["javanese"].tolist())
        hist.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "valid_chrF++": round(v, 2)})
        log(name, hist[-1])
        if v > best:
            best = v; best_state = {k: x.detach().to("cpu", copy=True) for k, x in model.state_dict().items()}
    model.load_state_dict(best_state); tok.src_lang = sl
    res = {"variant": name, "seed": seed, "flags": flags, "n_pairs": [len(id_pairs), len(jv_pairs)], "valid_chrF++_before": round(v0, 2),
           "best_epoch": max(hist, key=lambda h: h["valid_chrF++"])["epoch"], "history": hist, "test": {}}

    hyp = hyps_for(tok, model, bos, te, 4)
    for lv in LEVELS + ["all"]:
        m = te["level"] == lv if lv != "all" else te["level"].notna()
        h, r = [x for x, k in zip(hyp, m) if k], te.loc[m, "javanese"].tolist()
        res["test"][lv] = {"n": len(r), "BLEU": round(bleu(h, r), 2), "chrF++": round(chrf(h, r), 2), "chrF++_CI95": boot_ci(h, r)}
    pred = te.copy(); pred["hyp"] = hyp
    pred.to_csv(f"{OUT}/pred_{MODEL_KEY}_{name}_s{seed}.csv", index=False)

    u = sorted(te["indonesian"].unique())
    req = [(s, l) for s in u for l in LEVELS]
    outs = translate(tok, model, bos, [tag_src(s, l) for s, l in req])
    p = classify(outs)
    conf = np.zeros((4, 4), int)
    for (_, l), q in zip(req, p):
        conf[LID[l], q] += 1
    O = {(s, l): o for (s, l), o in zip(req, outs)}
    same = lambda a, b: round(float(np.mean([O[(s, a)] == O[(s, b)] for s in u])), 4)
    res["control"] = {"n": len(req), "level_accuracy": round(float(np.trace(conf) / conf.sum()), 4),
                      "per_level_accuracy": {l: round(float(conf[i, i] / conf[i].sum()), 4) for i, l in enumerate(LEVELS)},
                      "confusion_rows_requested_cols_predicted": conf.tolist(),
                      "identical_rate": {"ngoko=ngoko_alus": same("ngoko", "ngoko_alus"), "krama=krama_alus": same("krama", "krama_alus"),
                                         "ngoko=krama": same("ngoko", "krama")},
                      "markers_by_requested_level": {l: {k: sum(bool(toks(O[(s, l)]) & v) for s in u) for k, v in MARK.items()} for l in LEVELS}}
    pd.DataFrame({"indonesian": [s for s, _ in req], "requested": [l for _, l in req], "output": outs,
                  "classified": [LEVELS[q] for q in p]}).to_csv(f"{OUT}/control_{MODEL_KEY}_{name}_s{seed}.csv", index=False)
    log(name, "test all", res["test"]["all"], "control", res["control"]["level_accuracy"], res["control"]["identical_rate"])
    json.dump(res, open(f"{OUT}/metrics_{MODEL_KEY}_{name}_s{seed}.json", "w"), indent=1)
    del model, best_state, opt; torch.cuda.empty_cache()
    return res


summary = {"model": MODEL_KEY, "hparams": {"epochs": EPOCHS, "bs": BS, "lr": LR, "warmup": WARMUP, "seed": SEED, "maxlen": MAXLEN},
           "classifier": clf_info, "runs": {}}
for name, flags in [(n, VARIANTS[n]) for n in RUN_VARIANTS[MODEL_KEY]]:
  for seed in SEEDS:
    key = f"{name}_s{seed}"
    try:
        r = run(name, flags, seed)
        summary["runs"][key] = {"best_epoch": r["best_epoch"], "test": {k: (x["chrF++"], x["BLEU"]) for k, x in r["test"].items()},
                                 "control_acc": r["control"]["level_accuracy"], "control_per_level": r["control"]["per_level_accuracy"],
                                 "identical_rate": r["control"]["identical_rate"]}
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["runs"][key] = {"error": repr(e)}
    json.dump(summary, open(f"{OUT}/summary_{MODEL_KEY}.json", "w"), indent=1)
agg = {}
for name in RUN_VARIANTS[MODEL_KEY]:
    rs = [summary["runs"].get(f"{name}_s{s}", {}) for s in SEEDS]
    rs = [r for r in rs if "test" in r]
    if not rs: continue
    vals = {"chrF++_all": [r["test"]["all"][0] for r in rs], "BLEU_all": [r["test"]["all"][1] for r in rs],
            "control_acc": [r["control_acc"] for r in rs],
            **{f"control_{l}": [r["control_per_level"][l] for r in rs] for l in LEVELS},
            **{f"chrF++_{l}": [r["test"][l][0] for r in rs] for l in LEVELS},
            "identical_krama=krama_alus": [r["identical_rate"]["krama=krama_alus"] for r in rs],
            "identical_ngoko=ngoko_alus": [r["identical_rate"]["ngoko=ngoko_alus"] for r in rs]}
    agg[name] = {k: {"mean": round(float(np.mean(v)), 4), "std": round(float(np.std(v, ddof=1)) if len(v) > 1 else 0.0, 4), "values": v}
                 for k, v in vals.items()}
summary["aggregate"] = agg
json.dump(summary, open(f"{OUT}/summary_{MODEL_KEY}.json", "w"), indent=1)
log("SELESAI"); print(json.dumps(summary, indent=1))
