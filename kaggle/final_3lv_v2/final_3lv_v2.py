"""FINAL 3 TINGKAT v2 (Ngoko, Krama, Krama Alus). Seed 42. Semua pemilihan di VALID.

Titik awal: os3_mt_e15 (final_3lv_m2m100): test chrF++ 58.49, kontrol 0.884, kontrol Krama Alus 0.733.

Varian training:
  m2m_base     : os3_mt_e15 (dilatih ulang sebagai pembanding)
  m2m_na       : + JV->JV Ngoko Alus -> Krama Alus (209 pasangan; Ngoko Alus memakai krama inggil)
  m2m_na_ls    : + label smoothing 0.1
  nllb_na      : resep m2m_na di NLLB-200-distilled-600M

Decoding per varian:
  beam4        : beam search biasa
  rerank(lam)  : 8 kandidat beam, skor = logprob model + lam * log P(tingkat diminta) dari RERANKER.
                 Reranker = classifier terpisah (w11wo/javanese-roberta-small), BUKAN juri
                 (w11wo/javanese-distilbert-small) supaya akurasi kontrol tidak naik palsu.
                 lam dipilih di valid.
Skor pemilihan valid = chrF++ + 30 * akurasi kontrol (juri). Model terbaik -> /kaggle/working/model.
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "sentencepiece"], check=False)

import glob, json, time, random, re, shutil, os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
import numpy as np, pandas as pd, torch, sacrebleu
import torch.nn.functional as F
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

CFG = {"nllb": ("facebook/nllb-200-distilled-600M", "ind_Latn", "jav_Latn"),
       "m2m": ("facebook/m2m100_418M", "id", "jv")}
VARIANTS = {"m2m_base": dict(model="m2m", na=False, ls=0.0),
            "m2m_na": dict(model="m2m", na=True, ls=0.0),
            "m2m_na_ls": dict(model="m2m", na=True, ls=0.1),
            "nllb_na": dict(model="nllb", na=True, ls=0.0)}
EPOCHS, OS, BS, LR, WARMUP, SEED, MAXLEN, CTRL_W = 15, 3, 16, 5e-5, 0.1, 42, 64, 30.0
NBEST, LAMS = 8, [0.5, 1.0, 2.0, 4.0]
OUT = "/kaggle/working"
dev = "cuda"
LEVELS = ["ngoko", "krama", "krama_alus"]
LID = {l: i for i, l in enumerate(LEVELS)}
TAG = {l: f"<{l}>" for l in LEVELS}
INIT_WORDS = {"ngoko": "kowe aku mangan lunga turu", "krama": "sampeyan kula nedha kesah tilem",
              "krama_alus": "panjenenganipun dalem dhahar tindak sare"}
T0 = time.time()
log = lambda *a: print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)


def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)


find = lambda n: glob.glob(f"/kaggle/input/**/{n}.csv", recursive=True)[0]
tr_raw = pd.read_csv(find("train"))
def _load3(n):
    d = pd.read_csv(find(n))
    d = d[d["level"].isin(LEVELS)].copy()
    d["label"] = d["level"].map(LID)
    return d.reset_index(drop=True)
tr, va, te = (_load3(n) for n in ["train", "valid", "test"])
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


# ------------------------------------------------------------------ classifiers
def train_classifier(base, seed):
    ctok = AutoTokenizer.from_pretrained(base)
    cmod = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(LEVELS)).to(dev)
    seed_all(seed)
    opt = torch.optim.AdamW(cmod.parameters(), lr=5e-5)
    X, y = tr["javanese"].tolist(), tr["label"].tolist()
    for _ in range(4):
        cmod.train(); order = list(range(len(X))); random.shuffle(order)
        for i in range(0, len(order), 32):
            b = order[i:i + 32]
            enc = ctok([X[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
            loss = cmod(**enc, labels=torch.tensor([y[j] for j in b], device=dev)).loss
            opt.zero_grad(); loss.backward(); opt.step()
    cmod.eval()

    def logprobs(sents):
        out = []
        for i in range(0, len(sents), 64):
            enc = ctok(sents[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
            with torch.no_grad():
                out.append(F.log_softmax(cmod(**enc).logits.float(), -1).cpu())
        return torch.cat(out).numpy()

    info = {"base": base}
    for n, d in [("valid", va), ("test", te)]:
        p = logprobs(d["javanese"].tolist()).argmax(-1); y_ = d["label"].values
        info[f"acc_{n}"] = round(float((p == y_).mean()), 4)
        info[f"per_class_acc_{n}"] = {l: round(float((p[y_ == i] == i).mean()), 4) for i, l in enumerate(LEVELS)}
    log("classifier", info)
    return logprobs, ctok, cmod, info


judge_lp, _, _, judge_info = train_classifier("w11wo/javanese-distilbert-small", SEED)
classify = lambda s: judge_lp(s).argmax(-1).tolist()
try:
    rr_lp, rr_tok, rr_mod, rr_info = train_classifier("w11wo/javanese-roberta-small", 7)
except Exception as e:
    log("reranker roberta gagal, pakai xlm-roberta-base:", repr(e))
    rr_lp, rr_tok, rr_mod, rr_info = train_classifier("xlm-roberta-base", 7)
MARK = {"ngoko": {"kowe", "aku", "awakmu"}, "krama": {"sampeyan", "kula", "menika", "punika", "boten"},
        "alus": {"panjenengan", "panjenenganipun", "dalem", "adalem", "dhahar", "tindak", "kagungan", "sare"}}
toks = lambda s: set(re.findall(r"[\w'\-]+", str(s).lower()))


# ------------------------------------------------------------------ MT
def load(mk):
    repo, sl, tl = CFG[mk]
    tok = AutoTokenizer.from_pretrained(repo, src_lang=sl, tgt_lang=tl)
    model = AutoModelForSeq2SeqLM.from_pretrained(repo)
    tok.add_tokens(list(TAG.values()), special_tokens=True)
    model.resize_token_embeddings(len(tok))
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


def seq_scores(tok, model, srcs, cands, chunk=16):
    """Rata-rata log-prob token kandidat (teacher forcing), dicicil supaya logits 128k-vocab muat."""
    out = []
    for i in range(0, len(srcs), chunk):
        enc = tok(srcs[i:i + chunk], text_target=cands[i:i + chunk], return_tensors="pt",
                  padding=True, truncation=True, max_length=MAXLEN).to(dev)
        labels = enc.pop("labels"); labels[labels == tok.pad_token_id] = -100
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            logits = model(**enc, labels=labels).logits
        nll = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        m = (labels != -100).float()
        out += (-(nll * m).sum(1) / m.sum(1).clamp(min=1)).cpu().tolist()
        del logits, nll
    return np.array(out)


def nbest(tok, model, bos, srcs, k=NBEST, bs=8):
    """k kandidat (beam, tanpa output_scores supaya hemat memori) + skor log-prob ternormalisasi panjang."""
    model.eval(); out = [None] * len(srcs)
    order = sorted(range(len(srcs)), key=lambda i: len(srcs[i]))
    for i in range(0, len(order), bs):
        idx = order[i:i + bs]
        enc = tok([srcs[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(dev)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
            g = model.generate(**enc, forced_bos_token_id=bos, num_beams=k, num_return_sequences=k, max_new_tokens=MAXLEN)
        txt = tok.batch_decode(g, skip_special_tokens=True)
        sc = seq_scores(tok, model, [srcs[j] for j in idx for _ in range(k)], txt)
        for n, j in enumerate(idx):
            out[j] = (txt[n * k:(n + 1) * k], sc[n * k:(n + 1) * k])
    torch.cuda.empty_cache()
    return out


def rerank(nb, levels, lam):
    """Pilih kandidat dengan skor model + lam * log P_reranker(tingkat diminta)."""
    flat = [t for c, _ in nb for t in c]
    lp = rr_lp(flat); res = []; p = 0
    for (c, s), l in zip(nb, levels):
        k = len(c); tot = s + lam * lp[p:p + k, LID[l]]; p += k
        res.append(c[int(np.argmax(tot))])
    return res


def ctrl_acc(hyps, levels):
    p = np.array(classify(hyps)); y = np.array([LID[l] for l in levels])
    return float((p == y).mean()), {l: round(float((p[y == i] == i).mean()), 4) for i, l in enumerate(LEVELS)}


def valid_score(h):
    c = chrf(h, va["javanese"].tolist()); a, alv = ctrl_acc(h, va["level"].tolist())
    return c + CTRL_W * a, round(c, 2), round(a, 4), alv


def honor_pairs(na):
    pairs = []
    for _, g in tr_raw.groupby("group"):
        jv = dict(zip(g["level"], g["javanese"]))
        srcs = ["krama"] + (["ngoko_alus"] if na else [])
        for a in srcs:
            if a in jv and "krama_alus" in jv and jv[a] != jv["krama_alus"]:
                pairs.append((f"{TAG['krama_alus']} {jv[a]}", jv["krama_alus"]))
    return pairs


def evaluate(name, hyp_test, ctrl_req, ctrl_out):
    res = {"test": {}}
    for lv in LEVELS + ["all"]:
        m = te["level"] == lv if lv != "all" else te["level"].notna()
        h, r = [x for x, k in zip(hyp_test, m) if k], te.loc[m, "javanese"].tolist()
        res["test"][lv] = {"n": len(r), "BLEU": round(bleu(h, r), 2), "chrF++": round(chrf(h, r), 2), "chrF++_CI95": boot_ci(h, r)}
    p = classify(ctrl_out)
    conf = np.zeros((len(LEVELS), len(LEVELS)), int)
    for (_, l), q in zip(ctrl_req, p):
        conf[LID[l], q] += 1
    u = sorted({s for s, _ in ctrl_req})
    O = {(s, l): o for (s, l), o in zip(ctrl_req, ctrl_out)}
    same = lambda a, b: round(float(np.mean([O[(s, a)] == O[(s, b)] for s in u])), 4)
    res["control"] = {"level_accuracy": round(float(np.trace(conf) / conf.sum()), 4),
                      "per_level_accuracy": {l: round(float(conf[i, i] / conf[i].sum()), 4) for i, l in enumerate(LEVELS)},
                      "confusion_rows_requested_cols_predicted": conf.tolist(),
                      "identical_rate": {"krama=krama_alus": same("krama", "krama_alus"), "ngoko=krama": same("ngoko", "krama")},
                      "markers_by_requested_level": {l: {k: sum(bool(toks(O[(s, l)]) & v) for s in u) for k, v in MARK.items()} for l in LEVELS}}
    pred = te.copy(); pred["hyp"] = hyp_test
    pred.to_csv(f"{OUT}/pred_{name}.csv", index=False)
    pd.DataFrame({"indonesian": [s for s, _ in ctrl_req], "requested": [l for _, l in ctrl_req], "output": ctrl_out,
                  "classified": [LEVELS[q] for q in p]}).to_csv(f"{OUT}/control_{name}.csv", index=False)
    return res


def run(name, flags):
    seed_all(SEED)
    tok, model, bos, sl, tl = load(flags["model"])
    base = [(tag_src(s, l), t, l) for s, l, t in zip(tr["indonesian"], tr["level"], tr["javanese"])]
    id_pairs = [(s, t) for s, t, _ in base] + [(s, t) for s, t, l in base if l == "krama_alus"] * (OS - 1)
    jv_pairs = honor_pairs(flags["na"])
    log(name, "pairs ID->JV", len(id_pairs), "JV->JV", len(jv_pairs))

    def make_batches():
        a, b = id_pairs[:], jv_pairs[:]
        random.shuffle(a); random.shuffle(b)
        bt = [(sl, a[i:i + BS]) for i in range(0, len(a), BS)] + [(tl, b[i:i + BS]) for i in range(0, len(b), BS)]
        random.shuffle(bt); return bt

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
            labels = enc.pop("labels"); labels[labels == tok.pad_token_id] = -100
            with torch.autocast("cuda", dtype=torch.float16):
                logits = model(**enc, labels=labels).logits
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1),
                                   ignore_index=-100, label_smoothing=flags["ls"])
            opt.zero_grad(); scaler.scale(loss).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); tot += loss.item() * len(b); n += len(b)
        tok.src_lang = sl
        sc, c, a, alv = valid_score(translate(tok, model, bos, [tag_src(s, l) for s, l in zip(va["indonesian"], va["level"])], 1))
        hist.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "valid_chrF++": c, "valid_ctrl": a, "valid_ctrl_lv": alv, "score": round(sc, 2)})
        log(name, hist[-1])
        if sc > best:
            best = sc; best_state = {k: x.detach().to("cpu", copy=True) for k, x in model.state_dict().items()}
    model.load_state_dict(best_state); tok.src_lang = sl
    del best_state, opt, scaler, sched; import gc; gc.collect(); torch.cuda.empty_cache()

    # ---- decoding: beam4 vs rerank(lam), dipilih di valid
    va_src = [tag_src(s, l) for s, l in zip(va["indonesian"], va["level"])]
    dec = {"beam4": valid_score(translate(tok, model, bos, va_src, 4))}
    va_nb = nbest(tok, model, bos, va_src)
    for lam in LAMS:
        dec[f"rerank{lam}"] = valid_score(rerank(va_nb, va["level"].tolist(), lam))
    for k, v in dec.items():
        log(name, "valid decode", k, "score", round(v[0], 2), "chrF++", v[1], "ctrl", v[2], v[3])
    best_dec = max(dec, key=lambda k: dec[k][0])

    te_src = [tag_src(s, l) for s, l in zip(te["indonesian"], te["level"])]
    u = sorted(te["indonesian"].unique()); req = [(s, l) for s in u for l in LEVELS]
    c_src = [tag_src(s, l) for s, l in req]
    results = {}
    hyp_b = translate(tok, model, bos, te_src, 4); ctl_b = translate(tok, model, bos, c_src, 4)
    results["beam4"] = evaluate(f"{name}_beam4", hyp_b, req, ctl_b)
    if best_dec != "beam4":
        lam = float(best_dec[len("rerank"):])
        hyp_r = rerank(nbest(tok, model, bos, te_src), te["level"].tolist(), lam)
        ctl_r = rerank(nbest(tok, model, bos, c_src), [l for _, l in req], lam)
        results[best_dec] = evaluate(f"{name}_{best_dec}", hyp_r, req, ctl_r)
    for k, r in results.items():
        log(name, k, "TEST", r["test"]["all"], "ctrl", r["control"]["level_accuracy"], r["control"]["per_level_accuracy"])

    res = {"variant": name, "flags": flags, "n_pairs": [len(id_pairs), len(jv_pairs)], "history": hist,
           "best_epoch": max(hist, key=lambda h: h["score"])["epoch"],
           "valid_decode": {k: {"score": round(v[0], 2), "chrF++": v[1], "ctrl": v[2], "ctrl_lv": v[3]} for k, v in dec.items()},
           "best_decode": best_dec, "test": results}
    json.dump(res, open(f"{OUT}/metrics_{name}.json", "w"), indent=1)
    mdir = f"{OUT}/model_{name}"
    model.half().save_pretrained(mdir); tok.save_pretrained(mdir)
    del model; torch.cuda.empty_cache()
    return res


summary = {"levels": LEVELS, "hparams": {"epochs": EPOCHS, "os": OS, "bs": BS, "lr": LR, "warmup": WARMUP, "seed": SEED,
           "maxlen": MAXLEN, "ctrl_weight": CTRL_W, "nbest": NBEST, "lams": LAMS},
           "judge": judge_info, "reranker": rr_info, "runs": {}}
for name, flags in VARIANTS.items():
    try:
        r = run(name, flags)
        bd = r["best_decode"]
        summary["runs"][name] = {"best_epoch": r["best_epoch"], "best_decode": bd, "valid_score": r["valid_decode"][bd]["score"],
                                 "valid": r["valid_decode"],
                                 **{f"test_{k}": {"chrF++/BLEU": {lv: (x["chrF++"], x["BLEU"]) for lv, x in t["test"].items()},
                                                  "control_acc": t["control"]["level_accuracy"],
                                                  "control_per_level": t["control"]["per_level_accuracy"],
                                                  "identical_rate": t["control"]["identical_rate"]} for k, t in r["test"].items()}}
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["runs"][name] = {"error": repr(e)}
    json.dump(summary, open(f"{OUT}/summary_v2.json", "w"), indent=1)

ok = {k: v for k, v in summary["runs"].items() if "valid_score" in v}
winner = max(ok, key=lambda k: ok[k]["valid_score"])
w = ok[winner]; bd = w["best_decode"]
summary["winner"] = {"variant": winner, "decode": bd}
shutil.move(f"{OUT}/model_{winner}", f"{OUT}/model")
for k in ok:
    shutil.rmtree(f"{OUT}/model_{k}", ignore_errors=True)
repo, sl, tl = CFG[VARIANTS[winner]["model"]]
cfg = {"levels": LEVELS, "tags": TAG, "src_lang": sl, "tgt_lang": tl, "seed": SEED, "variant": winner, "base_model": repo,
       "decode": "beam4", "test": w[f"test_{bd}"]}
if bd != "beam4":
    rr_mod.save_pretrained(f"{OUT}/model/reranker"); rr_tok.save_pretrained(f"{OUT}/model/reranker")
    cfg.update({"decode": "rerank", "rerank_lambda": float(bd[len("rerank"):]), "nbest": NBEST, "reranker_dir": "reranker"})
json.dump(cfg, open(f"{OUT}/model/demo_config.json", "w"), indent=1)
json.dump(summary, open(f"{OUT}/summary_v2.json", "w"), indent=1)
log("PEMENANG (valid):", winner, bd)
log("SELESAI"); print(json.dumps(summary, indent=1))
