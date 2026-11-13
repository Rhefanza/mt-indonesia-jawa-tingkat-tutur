"""FINAL 3 TINGKAT (Ngoko, Krama, Krama Alus) untuk demo. M2M100-418M, seed 42.

Titik awal: atomic_init_os (terbaik di finetune_3lv_m2m100: chrF++ 57.8, kontrol 0.85, tapi
kontrol Krama Alus hanya ~0.60 karena 27/75 permintaan Krama Alus keluar sebagai Krama).

Perbaikan yang diuji (satu seed, dipilih di VALID, bukan test):
  os2_e15     : atomic_init_os lama, 15 epoch (valid chrF++ masih naik di epoch 10)
  os3_e15     : + oversampling Krama Alus x3 (210 -> 630, setara Ngoko/Krama ~592)
  os3_mt_e15  : + multi-task JV->JV Krama -> Krama Alus (pasangan dari train split)
Checkpoint & varian dipilih dengan skor valid = chrF++ + 30 * akurasi kontrol tingkat (classifier).
Model varian terbaik disimpan ke /kaggle/working/model (fp16) + demo_config.json.
"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "sentencepiece"], check=False)

import glob, json, time, random, re, shutil
import numpy as np, pandas as pd, torch, sacrebleu
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                          get_linear_schedule_with_warmup)

MODEL_KEY = "m2m100_418m"
REPO, SL, TL = "facebook/m2m100_418M", "id", "jv"
VARIANTS = {"os2_e15": dict(os=2, mt=False, epochs=15),
            "os3_e15": dict(os=3, mt=False, epochs=15),
            "os3_mt_e15": dict(os=3, mt=True, epochs=15)}
BS, LR, WARMUP, SEED, MAXLEN, CTRL_W = 16, 5e-5, 0.1, 42, 64, 30.0
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
def _load3(n):
    d = pd.read_csv(find(n))
    d = d[d["level"].isin(LEVELS)].copy()
    d["label"] = d["level"].map(LID)
    return d.reset_index(drop=True)
tr, va, te = (_load3(n) for n in ["train", "valid", "test"])
log("data", len(tr), len(va), len(te), dict(tr["level"].value_counts()), torch.cuda.get_device_name(0))
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


# ------------------------------------------------------------------ classifier (juri tingkat tutur)
def train_classifier():
    base = "w11wo/javanese-distilbert-small"
    ctok = AutoTokenizer.from_pretrained(base)
    cmod = AutoModelForSequenceClassification.from_pretrained(base, num_labels=len(LEVELS)).to(dev)
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
def load():
    tok = AutoTokenizer.from_pretrained(REPO, src_lang=SL, tgt_lang=TL)
    model = AutoModelForSeq2SeqLM.from_pretrained(REPO)
    tok.add_tokens(list(TAG.values()), special_tokens=True)
    model.resize_token_embeddings(len(tok))
    emb = model.get_input_embeddings().weight
    with torch.no_grad():
        for l, words in INIT_WORDS.items():
            ids = tok(words, add_special_tokens=False)["input_ids"]
            emb[tok.convert_tokens_to_ids(TAG[l])] = emb[ids].mean(0)
    return tok, model.to(dev), tok.get_lang_id(TL)


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


def valid_score(tok, model, bos):
    """Greedy di valid: chrF++ + akurasi kontrol (classifier menebak tingkat yang diminta)."""
    h = hyps_for(tok, model, bos, va, 1)
    c = chrf(h, va["javanese"].tolist())
    p = np.array(classify(h)); y = va["label"].values
    ctrl = float((p == y).mean())
    ctrl_lv = {l: round(float((p[y == i] == i).mean()), 4) for i, l in enumerate(LEVELS)}
    return c + CTRL_W * ctrl, round(c, 2), round(ctrl, 4), ctrl_lv


def honor_pairs():
    pairs = []
    for _, g in tr.groupby("group"):
        jv = dict(zip(g["level"], g["javanese"]))
        if "krama" in jv and "krama_alus" in jv and jv["krama"] != jv["krama_alus"]:
            pairs.append((f"{TAG['krama_alus']} {jv['krama']}", jv["krama_alus"]))
    return pairs


def run(name, flags):
    seed_all(SEED)
    tok, model, bos = load()
    base = [(tag_src(s, l), t, l) for s, l, t in zip(tr["indonesian"], tr["level"], tr["javanese"])]
    id_pairs = [(s, t) for s, t, _ in base]
    id_pairs += [(s, t) for s, t, l in base if l == "krama_alus"] * (flags["os"] - 1)
    jv_pairs = honor_pairs() if flags["mt"] else []
    log(name, "pairs ID->JV", len(id_pairs), "JV->JV", len(jv_pairs))

    def make_batches():
        a, b = id_pairs[:], jv_pairs[:]
        random.shuffle(a); random.shuffle(b)
        bt = [(SL, a[i:i + BS]) for i in range(0, len(a), BS)] + [(TL, b[i:i + BS]) for i in range(0, len(b), BS)]
        random.shuffle(bt); return bt

    E = flags["epochs"]
    n_batches = len(make_batches())
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    steps = E * n_batches
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * steps), steps)
    scaler = torch.amp.GradScaler("cuda")
    best, best_state, hist = -1, None, []
    for ep in range(E):
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
        tok.src_lang = SL
        sc, c, ctrl, ctrl_lv = valid_score(tok, model, bos)
        hist.append({"epoch": ep + 1, "train_loss": round(tot / n, 4), "valid_chrF++": c, "valid_ctrl": ctrl,
                     "valid_ctrl_lv": ctrl_lv, "score": round(sc, 2)})
        log(name, hist[-1])
        if sc > best:
            best = sc; best_state = {k: x.detach().to("cpu", copy=True) for k, x in model.state_dict().items()}
    model.load_state_dict(best_state); tok.src_lang = SL
    bh = max(hist, key=lambda h: h["score"])
    res = {"variant": name, "seed": SEED, "flags": flags, "n_pairs": [len(id_pairs), len(jv_pairs)],
           "best_epoch": bh["epoch"], "valid_score": bh["score"], "valid_chrF++": bh["valid_chrF++"],
           "valid_ctrl": bh["valid_ctrl"], "history": hist, "test": {}}

    hyp = hyps_for(tok, model, bos, te, 4)
    for lv in LEVELS + ["all"]:
        m = te["level"] == lv if lv != "all" else te["level"].notna()
        h, r = [x for x, k in zip(hyp, m) if k], te.loc[m, "javanese"].tolist()
        res["test"][lv] = {"n": len(r), "BLEU": round(bleu(h, r), 2), "chrF++": round(chrf(h, r), 2), "chrF++_CI95": boot_ci(h, r)}
    pred = te.copy(); pred["hyp"] = hyp
    pred.to_csv(f"{OUT}/pred3f_{name}.csv", index=False)

    u = sorted(te["indonesian"].unique())
    req = [(s, l) for s in u for l in LEVELS]
    outs = translate(tok, model, bos, [tag_src(s, l) for s, l in req])
    p = classify(outs)
    conf = np.zeros((len(LEVELS), len(LEVELS)), int)
    for (_, l), q in zip(req, p):
        conf[LID[l], q] += 1
    O = {(s, l): o for (s, l), o in zip(req, outs)}
    same = lambda a, b: round(float(np.mean([O[(s, a)] == O[(s, b)] for s in u])), 4)
    res["control"] = {"n": len(req), "level_accuracy": round(float(np.trace(conf) / conf.sum()), 4),
                      "per_level_accuracy": {l: round(float(conf[i, i] / conf[i].sum()), 4) for i, l in enumerate(LEVELS)},
                      "confusion_rows_requested_cols_predicted": conf.tolist(),
                      "identical_rate": {"krama=krama_alus": same("krama", "krama_alus"), "ngoko=krama": same("ngoko", "krama")},
                      "markers_by_requested_level": {l: {k: sum(bool(toks(O[(s, l)]) & v) for s in u) for k, v in MARK.items()} for l in LEVELS}}
    pd.DataFrame({"indonesian": [s for s, _ in req], "requested": [l for _, l in req], "output": outs,
                  "classified": [LEVELS[q] for q in p]}).to_csv(f"{OUT}/control3f_{name}.csv", index=False)
    log(name, "TEST", res["test"]["all"], "control", res["control"]["level_accuracy"], res["control"]["per_level_accuracy"],
        res["control"]["identical_rate"])
    json.dump(res, open(f"{OUT}/metrics3f_{name}.json", "w"), indent=1)

    mdir = f"{OUT}/model_{name}"
    model.half().save_pretrained(mdir); tok.save_pretrained(mdir)
    json.dump({"levels": LEVELS, "tags": TAG, "src_lang": SL, "tgt_lang": TL, "seed": SEED, "variant": name,
               "valid_chrF++": res["valid_chrF++"], "valid_ctrl": res["valid_ctrl"],
               "test_chrF++": res["test"]["all"]["chrF++"], "test_control": res["control"]["level_accuracy"]},
              open(f"{mdir}/demo_config.json", "w"), indent=1)
    del model, best_state, opt; torch.cuda.empty_cache()
    return res


summary = {"model": MODEL_KEY, "levels": LEVELS, "hparams": {"bs": BS, "lr": LR, "warmup": WARMUP, "seed": SEED,
           "maxlen": MAXLEN, "ctrl_weight": CTRL_W}, "classifier": clf_info, "runs": {}}
for name, flags in VARIANTS.items():
    try:
        r = run(name, flags)
        summary["runs"][name] = {"best_epoch": r["best_epoch"], "valid_score": r["valid_score"], "valid_chrF++": r["valid_chrF++"],
                                 "valid_ctrl": r["valid_ctrl"],
                                 "test": {k: (x["chrF++"], x["BLEU"]) for k, x in r["test"].items()},
                                 "control_acc": r["control"]["level_accuracy"], "control_per_level": r["control"]["per_level_accuracy"],
                                 "identical_rate": r["control"]["identical_rate"]}
    except Exception as e:
        import traceback; traceback.print_exc()
        summary["runs"][name] = {"error": repr(e)}
    json.dump(summary, open(f"{OUT}/summary3_final.json", "w"), indent=1)

ok = {k: v for k, v in summary["runs"].items() if "valid_score" in v}
winner = max(ok, key=lambda k: ok[k]["valid_score"])
summary["winner"] = winner
shutil.move(f"{OUT}/model_{winner}", f"{OUT}/model")
for k in ok:
    shutil.rmtree(f"{OUT}/model_{k}", ignore_errors=True)
json.dump(summary, open(f"{OUT}/summary3_final.json", "w"), indent=1)
log("PEMENANG (dipilih di valid):", winner, "-> disimpan ke", f"{OUT}/model")
log("SELESAI"); print(json.dumps(summary, indent=1))
