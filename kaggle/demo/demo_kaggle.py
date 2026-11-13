"""Kernel demo Kaggle (CPU): menjalankan demo/app.py dengan link publik Gradio.
Link dikirim ke ntfy.sh topik acak supaya bisa diambil saat kernel masih berjalan.
Demo hidup selama kernel berjalan (maks ~11,5 jam).
File ini dibangun dari demo/app.py (bagian tengah identik); ubah app.py lalu bangun ulang.
"""
import subprocess, sys, glob, os
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gradio", "sentencepiece"], check=False)
cand = [p for p in glob.glob("/kaggle/input/**/demo_config.json", recursive=True) if "/reranker/" not in p]
os.environ["MODEL_DIR"] = os.path.dirname(cand[0])
print("MODEL_DIR", os.environ["MODEL_DIR"], flush=True)
NTFY_TOPIC = "mtjawa-d77fcb5e42ae6258"

"""Demo: Machine Translation Indonesia -> Jawa dengan kontrol tingkat tutur.

Model: M2M100-418M, fine-tuning pada Unggah-Ungguh (group split), atomic tag tingkat tutur,
embedding tag diinisialisasi dari kata penanda + oversampling contoh Krama Alus + multi-task JV->JV.
Jika demo_config.json berisi "decode": "rerank", model membuat N kandidat beam lalu memilih kandidat
dengan skor log-prob model + lambda * log P(tingkat diminta) dari classifier reranker.
"""
import json
import os
import time
from pathlib import Path

import gradio as gr
import torch
import torch.nn.functional as F
from transformers import AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoTokenizer

MODEL_DIR = Path(os.environ.get("MODEL_DIR", Path(__file__).parent / "model"))
CFG = json.loads((MODEL_DIR / "demo_config.json").read_text(encoding="utf-8"))
LEVELS = CFG["levels"]
TAGS = CFG["tags"]
LABEL = {"ngoko": "Ngoko", "ngoko_alus": "Ngoko Alus", "krama": "Krama", "krama_alus": "Krama Alus"}
NOTE = {
    "ngoko": "akrab, sebaya, atau ke yang lebih muda",
    "ngoko_alus": "akrab tetapi menghormati orang yang dibicarakan",
    "krama": "sopan ke orang yang belum akrab atau lebih tua",
    "krama_alus": "paling halus, ke orang yang sangat dihormati",
}
MAXLEN = 64

torch.set_num_threads(max(1, torch.get_num_threads()))
tok = AutoTokenizer.from_pretrained(MODEL_DIR, src_lang=CFG["src_lang"], tgt_lang=CFG["tgt_lang"])
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_DIR).float().eval()
BOS = tok.get_lang_id(CFG["tgt_lang"]) if hasattr(tok, "get_lang_id") else tok.convert_tokens_to_ids(CFG["tgt_lang"])
RERANK = CFG.get("decode") == "rerank"
if RERANK:
    NBEST, LAM = int(CFG["nbest"]), float(CFG["rerank_lambda"])
    rr_dir = MODEL_DIR / CFG.get("reranker_dir", "reranker")
    rr_tok = AutoTokenizer.from_pretrained(rr_dir)
    rr_mod = AutoModelForSequenceClassification.from_pretrained(rr_dir).float().eval()
MAX_CHARS = 300


def _beam(sources, k):
    enc = tok(sources, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
    with torch.no_grad():
        out = model.generate(**enc, forced_bos_token_id=BOS, num_beams=max(k, 4), num_return_sequences=k,
                             max_new_tokens=MAXLEN)
    return tok.batch_decode(out, skip_special_tokens=True)


def _seq_scores(sources, cands):
    """Rata-rata log-prob token kandidat menurut model (teacher forcing)."""
    enc = tok(sources, text_target=cands, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
    labels = enc.pop("labels")
    labels[labels == tok.pad_token_id] = -100
    with torch.no_grad():
        logits = model(**enc, labels=labels).logits
    nll = F.cross_entropy(logits.transpose(1, 2), labels, ignore_index=-100, reduction="none")
    m = (labels != -100).float()
    return -(nll * m).sum(1) / m.sum(1).clamp(min=1)


def _level_logprobs(sents):
    enc = rr_tok(sents, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN)
    with torch.no_grad():
        return F.log_softmax(rr_mod(**enc).logits, -1)


def _generate(text, levels):
    sources = [f"{TAGS[l]} {text}" for l in levels]
    if not RERANK:
        return _beam(sources, 1)
    cands = _beam(sources, NBEST)
    score = _seq_scores([s for s in sources for _ in range(NBEST)], cands)
    lp = _level_logprobs(cands)
    outs = []
    for i, l in enumerate(levels):
        sl = slice(i * NBEST, (i + 1) * NBEST)
        total = score[sl] + LAM * lp[sl, LEVELS.index(l)]
        outs.append(cands[sl][int(total.argmax())])
    return outs


_t = time.time()
_generate("Bapak sudah pulang.", LEVELS)  # pemanasan: panggilan pertama di CPU bisa >1 menit
print(f"pemanasan selesai {time.time() - _t:.1f} detik", flush=True)


def _clean(text):
    text = " ".join((text or "").split())
    if not text:
        raise gr.Error("Tulis kalimat bahasa Indonesia terlebih dahulu.")
    if len(text) > MAX_CHARS:
        raise gr.Error(f"Kalimat terlalu panjang (maksimum {MAX_CHARS} karakter). Model dilatih pada kalimat pendek.")
    return text


def translate_one(text, level_label):
    text = _clean(text)
    level = next(k for k, v in LABEL.items() if v == level_label)
    t0 = time.time()
    out = _generate(text, [level])[0]
    return out, f"{LABEL[level]}: {NOTE[level]} · {time.time() - t0:.1f} detik"


def translate_all(text):
    text = _clean(text)
    t0 = time.time()
    outs = _generate(text, LEVELS)
    rows = [[LABEL[l], o] for l, o in zip(LEVELS, outs)]
    return rows, f"{len(LEVELS)} tingkat · {time.time() - t0:.1f} detik"


def _eval_note():
    t = CFG.get("test")
    if not t:
        return ""
    num = lambda x, d: f"{x:.{d}f}".replace(".", ",")
    per = " · ".join(f"{LABEL[l]} {num(t['control_per_level'][l], 2)}" for l in LEVELS)
    chrf, bleu = t["chrF++/BLEU"]["all"]
    return (f"- Hasil evaluasi (test set, seed {CFG.get('seed', 42)}): chrF++ {num(chrf, 1)} · BLEU {num(bleu, 1)} · "
            f"ketepatan tingkat {num(t['control_acc'], 2)} ({per}).\n")


EXAMPLES = [
    "Itu, kakek disuruh mandi terlebih dahulu!",
    "Mbak Srini makan pecel dengan ajang pincuk.",
    "Bapak sudah pulang dari kantor.",
    "Saya mau pergi ke pasar membeli sayur.",
    "Ibu sedang tidur di kamar.",
]
LEVEL_NAMES = ", ".join(LABEL[l] for l in LEVELS)

with gr.Blocks(title="MT Indonesia → Jawa Tingkat Tutur") as demo:
    gr.Markdown(
        "# Indonesia → Jawa dengan Pilihan Tingkat Tutur\n"
        f"Ketik kalimat bahasa Indonesia, lalu pilih tingkat tutur bahasa Jawa ({LEVEL_NAMES}). "
        "Model: **M2M100-418M** hasil fine-tuning pada dataset **Unggah-Ungguh** dengan tag tingkat tutur"
        + (", ditambah *rerank* kandidat dengan classifier tingkat tutur." if RERANK else ".")
    )
    with gr.Row():
        with gr.Column(scale=3):
            src = gr.Textbox(label="Kalimat bahasa Indonesia", lines=3, max_lines=6,
                             placeholder="Contoh: Bapak sudah pulang dari kantor.")
            level = gr.Radio([LABEL[l] for l in LEVELS], value=LABEL[LEVELS[-1]], label="Tingkat tutur")
            with gr.Row():
                btn_one = gr.Button("Terjemahkan", variant="primary")
                btn_all = gr.Button(f"Bandingkan {len(LEVELS)} tingkat")
            gr.Examples(EXAMPLES, inputs=src, label="Contoh kalimat")
        with gr.Column(scale=4):
            out_one = gr.Textbox(label="Terjemahan bahasa Jawa", lines=3, interactive=False)
            info = gr.Markdown()
            out_all = gr.Dataframe(headers=["Tingkat", "Terjemahan"], label="Semua tingkat",
                                   wrap=True, interactive=False)

    btn_one.click(translate_one, [src, level], [out_one, info])
    src.submit(translate_one, [src, level], [out_one, info])
    btn_all.click(translate_all, [src], [out_all, info])

    gr.Markdown(
        "### Catatan\n"
        + _eval_note()
        + "- Tingkat **Krama Alus** masih paling sering meleset: permintaan Krama Alus kadang menghasilkan Krama biasa.\n"
        "- Model dilatih pada sekitar 1.600 kalimat pendek (rata-rata 8 kata) dari kamus dan buku unggah-ungguh. "
        "Kalimat panjang, istilah teknis, atau dialek daerah bisa diterjemahkan kurang tepat.\n"
        "- Output belum diperiksa penutur Jawa; jangan dipakai untuk komunikasi penting tanpa pengecekan.\n"
        "- Dataset: Unggah-Ungguh (Farhansyah dkk., ACL 2025), lisensi CC BY-NC 4.0. Demo ini untuk keperluan riset/kuliah."
    )


import time, urllib.request
_, local_url, share_url = demo.queue(max_size=16).launch(share=True, prevent_thread_lock=True)
print("LOCAL", local_url, "SHARE", share_url, flush=True)
try:
    urllib.request.urlopen(urllib.request.Request(f"https://ntfy.sh/{NTFY_TOPIC}", data=str(share_url).encode(), method="POST"), timeout=30)
except Exception as e:
    print("ntfy gagal", e, flush=True)
t0 = time.time()
while time.time() - t0 < 11.5 * 3600:
    time.sleep(60)
