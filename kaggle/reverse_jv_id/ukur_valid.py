"""Ukur model reverse jv->id (pool_makna_v1) di VALID: fine-tuned, zero-shot, dan salin input. CPU, root repo."""
import json, sys
import numpy as np, torch
from sacrebleu.metrics import BLEU, CHRF
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
sys.path.insert(0, "scripts")
from juri_tetap import LEVELS, load3
torch.set_num_threads(8)
tr, va = load3("train"), load3("valid")
src, ref = va.javanese.tolist(), va.indonesian.tolist()
H = {"salin": src}
for name, path in [("fine_tuned", "kaggle/pool_makna_v1/output/reverse"), ("zero_shot", "facebook/m2m100_418M")]:
    tok = AutoTokenizer.from_pretrained("facebook/m2m100_418M", src_lang="jv", tgt_lang="id")
    if name == "fine_tuned":
        t2 = AutoTokenizer.from_pretrained(path, src_lang="jv", tgt_lang="id")
        assert t2(src[0]).input_ids == tok(src[0]).input_ids, "tokenizer beda"
    mod = AutoModelForSeq2SeqLM.from_pretrained(path, dtype=torch.float32).eval()
    out = []
    for i in range(0, len(src), 16):
        enc = tok(src[i:i + 16], return_tensors="pt", padding=True, truncation=True, max_length=64)
        with torch.no_grad():
            g = mod.generate(**enc, forced_bos_token_id=tok.get_lang_id("id"), num_beams=5, max_length=None, max_new_tokens=64)
        out += tok.batch_decode(g, skip_special_tokens=True)
    H[name] = out; print(name, "selesai", flush=True)
R = {"indonesian_valid_ada_di_train": int(va.indonesian.isin(set(tr.indonesian)).sum())}
for n, h in H.items():
    R[n] = {}
    for lv in ["all"] + LEVELS:
        ix = range(len(va)) if lv == "all" else np.where(va.level == lv)[0]
        a, b = [h[i] for i in ix], [[ref[i] for i in ix]]
        R[n][lv] = {"n": len(a), "chrF++": round(CHRF(word_order=2).corpus_score(a, b).score, 2), "BLEU": round(BLEU().corpus_score(a, b).score, 2)}
    R[n]["sama_persis"] = sum(x.strip() == y.strip() for x, y in zip(h, ref))
va["zero_shot"], va["fine_tuned"] = H["zero_shot"], H["fine_tuned"]
va[["index", "group", "level", "javanese", "indonesian", "zero_shot", "fine_tuned"]].to_csv("kaggle/reverse_jv_id/valid.csv", index=False)
json.dump(R, open("kaggle/reverse_jv_id/metrics.json", "w"), indent=1)
print(json.dumps(R))
for lv in LEVELS:
    for r in va[va.level == lv].sample(14, random_state=0).itertuples():
        print(f"[{lv}] JV {r.javanese}\n  REF {r.indonesian}\n  ZS {r.zero_shot}\n  FT {r.fine_tuned}")
