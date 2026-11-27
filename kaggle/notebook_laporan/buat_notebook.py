"""Bangkitkan notebook_laporan.ipynb (notebook yang dikumpulkan ke dosen).

Aturan isi: tanpa singkatan buatan sendiri (tulis "Krama Alus", bukan "KA"; "interval kepercayaan 90%", bukan "CI90"),
istilah teknis Inggris tidak diterjemahkan, semua angka dihitung ulang dari bahan mentah di dataset Kaggle.
Jalankan:  python kaggle/notebook_laporan/buat_notebook.py
"""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "paper", "script"))
CELLS = []


def md(s):
    CELLS.append(("markdown", s.strip("\n")))


def code(s):
    CELLS.append(("code", s.strip("\n")))


# ====================================================================== 0. judul
md(r"""
# Penerjemahan Mesin Indonesia → Jawa dengan Kontrol Tingkat Tutur

**Ngoko, Krama, dan Krama Alus dalam satu model, dipilih lewat tag di input**

Tugas Ujian Akhir Semester, mata kuliah Natural Language Processing (semester 5).

| | |
|---|---|
| Nama / NIM | *(isi di sini)* |
| Dataset | Unggah-Ungguh (`JavaneseHonorifics/Unggah-Ungguh`, lisensi CC BY-NC 4.0) |
| Model final | M2M100-418M, full fine-tuning, dengan rerank kandidat |
| Repositori | https://github.com/Rhefanza/mt-indonesia-jawa-tingkat-tutur |

### Ringkasan

Bahasa Jawa punya tingkat tutur: kalimat yang sama ditulis berbeda untuk teman sebaya (Ngoko), orang yang dihormati
(Krama), atau orang yang sangat dihormati (Krama Alus). Sistem penerjemahan mesin umum hanya menghasilkan satu versi.
Proyek ini melatih satu model yang menerima tag tingkat di awal kalimat Indonesia (misalnya `<krama_alus>`) lalu
menghasilkan terjemahan Jawa pada tingkat itu. Pada 176 kalimat test, model final mencapai **chrF++ 58,62** dan
**BLEU 33,17**, dengan **ketepatan tingkat 0,947** (Krama Alus 0,867) menurut classifier penilai yang terpisah dari model.
Masalah terbesar yang tersisa adalah **makna**, bukan tingkat: banyak output Krama Alus sudah benar tingkatnya tetapi
maknanya bergeser.

### Cara menjalankan notebook ini

Notebook ini **melatih tujuh model**: lima untuk membandingkan langkah preprocessing satu per satu, dan dua model dasar
lain (NLLB dan Cendol) dengan resep yang sama (bagian 6). Karena itu notebook membutuhkan **GPU** (Kaggle:
*Settings → Accelerator → GPU T4*) dan waktu sekitar 2,5–3 jam. Bahan yang dibutuhkan
(dataset, classifier penilai, dan hasil eksperimen sebelumnya) dimuat dari dataset Kaggle
`unggah-ungguh-bahan-notebook-laporan`. Cukup tekan *Run All*.

Hasil training di notebook ini bisa sedikit berbeda dari model final di paper walaupun resepnya sama, karena training
di GPU tidak sepenuhnya deterministik. Model final di paper ditampilkan sebagai pembanding di bagian 9.

### Daftar isi

1. Pendahuluan
2. Persiapan
3. Data
4. Exploratory Data Analysis
5. Preprocessing: sebelum dan sesudah
6. Modelling
7. Learning curve
8. Pemilihan checkpoint dan rerank
9. Evaluasi pada data test
10. Analisis kesalahan
11. Eksperimen yang tidak berhasil
12. Kesimpulan, keterbatasan, dan saran
13. Referensi
""")

# ====================================================================== 1. pendahuluan
md(r"""
## 1. Pendahuluan

### 1.1 Masalah

Unggah-ungguh basa (tingkat tutur) adalah aturan memilih kosakata Jawa sesuai hubungan sosial pembicara dan lawan
bicara. Contoh untuk kalimat Indonesia *"Kamu sudah makan?"*:

| Tingkat | Kalimat Jawa | Dipakai untuk |
|---|---|---|
| Ngoko | *Kowe wis mangan?* | teman sebaya, orang yang lebih muda |
| Krama | *Sampeyan sampun nedha?* | orang yang belum akrab, situasi sopan |
| Krama Alus | *Panjenengan sampun dhahar?* | orang tua, guru, orang yang sangat dihormati |

Satu kalimat Indonesia punya beberapa terjemahan benar, dan semuanya bergantung pada siapa yang diajak bicara.
Model penerjemahan umum tidak tahu konteks itu, sehingga biasanya condong ke Ngoko. Penelitian sebelumnya juga
menemukan bahwa model bahasa besar bias ke Ngoko dan lemah pada tingkat hormat (Farhansyah dkk., 2025).

### 1.2 Tujuan dan pertanyaan penelitian

Tujuan: membangun satu model penerjemahan Indonesia → Jawa yang tingkat tuturnya bisa dipilih pengguna.

1. Seberapa besar pengaruh tag tingkat di input (*source tag*, Sennrich dkk., 2016) terhadap kualitas terjemahan
   dan ketepatan tingkat?
2. Perlakuan data apa yang paling membantu tingkat yang paling sedikit datanya (Krama Alus)?
3. Apa yang membatasi kualitas model: kontrol tingkat atau kosakata/makna?

### 1.3 Metrik yang dipakai

| Metrik | Yang diukur | Keterangan |
|---|---|---|
| chrF++ | kemiripan terjemahan dengan referensi pada level karakter dan kata | metrik utama; cocok untuk bahasa beraglutinasi seperti Jawa (Popović, 2017) |
| BLEU | kemiripan n-gram kata dengan referensi | metrik pembanding yang umum (Papineni dkk., 2002) |
| Ketepatan tingkat | proporsi output yang dinilai berada pada tingkat yang diminta | dinilai oleh classifier penilai (DistilBERT Jawa) yang **berbeda** dari classifier yang dipakai saat rerank |
""")

# ====================================================================== 2. persiapan
md(r"""
## 2. Persiapan

Library yang dipakai: `pandas`, `numpy`, `matplotlib`, `sacrebleu` (chrF++ dan BLEU), `transformers` dan `torch`
(classifier penilai tingkat), `wordcloud`. Folder bahan dicari otomatis, jadi notebook berjalan di Kaggle maupun lokal.
""")
code(r"""
import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "sacrebleu", "wordcloud"], check=False)

import glob, json, math, os, random, re, unicodedata, warnings
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")  # kurangi fragmentasi memori GPU
from collections import Counter
import numpy as np, pandas as pd, sacrebleu, torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from IPython.display import display, Markdown
from transformers import AutoTokenizer, AutoModelForSequenceClassification
warnings.filterwarnings("ignore")

def cari_root():
    kandidat = glob.glob("/kaggle/input/**/data/unggah_ungguh/split/test.csv", recursive=True)
    kandidat += glob.glob("dataset/data/unggah_ungguh/split/test.csv") + glob.glob("data/unggah_ungguh/split/test.csv")
    assert kandidat, "bahan tidak ditemukan; tambahkan dataset unggah-ungguh-bahan-notebook-laporan"
    return os.path.abspath(os.path.join(os.path.dirname(kandidat[0]), "..", "..", ".."))

ROOT = cari_root()
K = os.path.join(ROOT, "kaggle")
LV = ["ngoko", "krama", "krama_alus"]
NAMA = {"ngoko": "Ngoko", "krama": "Krama", "krama_alus": "Krama Alus", "ngoko_alus": "Ngoko Alus", "all": "Semua"}
WARNA = {"ngoko": "#4C78A8", "krama": "#F58518", "krama_alus": "#B279A2", "all": "#222222"}
random.seed(42); np.random.seed(42)
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False, "font.size": 10})

def koma(x, d=2):
    # angka gaya Indonesia: 58.62 -> "58,62"
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:,.{d}f}".replace(",", "_").replace(".", ",").replace("_", ".")

def chrf(h, r): return sacrebleu.corpus_chrf(list(h), [list(r)], word_order=2).score
def bleu(h, r): return sacrebleu.corpus_bleu(list(h), [list(r)]).score
def token(s): return re.findall(r"[^\W_]+", unicodedata.normalize("NFC", str(s)).casefold())

print("folder bahan:", ROOT)
print("sacrebleu", sacrebleu.__version__, "| torch", torch.__version__, "| GPU:", torch.cuda.is_available())
""")

md(r"""
### 2.1 Classifier penilai tingkat (juri tetap)

Ketepatan tingkat dinilai oleh classifier `w11wo/javanese-distilbert-small` yang di-fine-tune untuk membedakan
Ngoko, Krama, dan Krama Alus. Supaya angka antar eksperimen bisa dibandingkan, dipakai **satu set penilai yang sama
untuk semua eksperimen**: tiga classifier (seed 42, 13, 2024) yang probabilitasnya dirata-rata. Weights-nya dimuat
dari dataset, bukan dilatih ulang, sehingga hasilnya identik setiap kali dijalankan.
""")
code(r"""
DEV = "cuda" if torch.cuda.is_available() else "cpu"
JURI = []
for s in [42, 13, 2024]:
    d = os.path.join(ROOT, "models", "juri_tetap", f"seed{s}")
    JURI.append((AutoTokenizer.from_pretrained(d), AutoModelForSequenceClassification.from_pretrained(d).to(DEV).eval()))

def prob_juri(kalimat):
    kalimat = [str(x) for x in kalimat]; hasil = []
    for tok, mod in JURI:
        out = []
        for i in range(0, len(kalimat), 64):
            enc = tok(kalimat[i:i + 64], return_tensors="pt", padding=True, truncation=True, max_length=64).to(DEV)
            with torch.no_grad():
                out.append(F.softmax(mod(**enc).logits.float(), -1).cpu())
        hasil.append(torch.cat(out).numpy())
    return np.mean(hasil, 0)

def vonis_juri(kalimat):
    return np.array([LV[i] for i in prob_juri(kalimat).argmax(-1)])

print("classifier penilai dimuat:", len(JURI), "seed, perangkat", DEV)
""")

# ====================================================================== 3. data
md(r"""
## 3. Data

Dataset **Unggah-Ungguh** (Farhansyah dkk., 2025) berisi kalimat Indonesia beserta terjemahan Jawanya pada beberapa
tingkat tutur. Kalimat-kalimat yang merupakan terjemahan dari kalimat yang sama dikelompokkan dalam satu **grup**.
Proyek ini memakai tiga tingkat: Ngoko, Krama, dan Krama Alus. Ngoko Alus tidak dipakai sebagai target karena
datanya paling sedikit dan versi awal proyek menunjukkan tingkat ini paling sulit dibedakan.
""")
code(r"""
df = pd.read_csv(f"{ROOT}/data/unggah_ungguh/translation.csv")
df = df.rename(columns={"javanese sentence": "javanese", "indonesian sentence": "indonesian", "english sentence": "english"})
LABEL = {0: "Ngoko", 1: "Ngoko Alus", 2: "Krama", 3: "Krama Alus"}
print("jumlah baris:", len(df), "| jumlah grup:", df.group.nunique())
display(df.label.map(LABEL).value_counts().rename_axis("Tingkat").reset_index(name="Jumlah kalimat"))
contoh = df[df.group == df.group.value_counts().index[0]][["group", "label", "indonesian", "javanese"]].copy()
contoh["label"] = contoh.label.map(LABEL)
display(contoh.rename(columns={"group": "Grup", "label": "Tingkat", "indonesian": "Indonesia", "javanese": "Jawa"}))
""")
code(r"""
SPLIT = {}
for sp in ["train", "valid", "test"]:
    d = pd.read_csv(f"{ROOT}/data/unggah_ungguh/split/{sp}.csv")
    SPLIT[sp] = d[d.level.isin(LV)].reset_index(drop=True)
TRAIN, VALID, TEST = SPLIT["train"], SPLIT["valid"], SPLIT["test"]
tabel = pd.DataFrame({sp: SPLIT[sp].level.value_counts().reindex(LV) for sp in SPLIT}).rename(index=NAMA)
tabel.loc["Jumlah"] = tabel.sum()
tabel.loc["Jumlah grup"] = [SPLIT[sp].group.nunique() for sp in SPLIT]
display(tabel)
assert (len(TRAIN), len(VALID), len(TEST)) == (1395, 175, 176)
""")

# ====================================================================== 4. EDA
md(r"""
## 4. Exploratory Data Analysis

Tujuan bagian ini adalah menemukan karakteristik data yang memengaruhi keputusan modelling. Setiap temuan diakhiri
dengan **dampaknya ke keputusan**.

### 4.1 Distribusi tingkat per split
""")
code(r"""
fig, ax = plt.subplots(figsize=(7, 3.2))
x = np.arange(3); w = 0.26
for i, lv in enumerate(LV):
    v = [int((SPLIT[sp].level == lv).sum()) for sp in SPLIT]
    b = ax.bar(x + (i - 1) * w, v, w, color=WARNA[lv], label=NAMA[lv])
    ax.bar_label(b, fontsize=8)
ax.set_xticks(x, ["train", "valid", "test"]); ax.set_ylabel("jumlah kalimat")
ax.set_title("Distribusi tingkat tutur per split"); ax.legend(frameon=False)
plt.tight_layout(); plt.show()
p_ka = (TRAIN.level == "krama_alus").mean()
print(f"porsi Krama Alus di train: {koma(100 * p_ka, 1)}%")
""")
md(r"""
**Temuan:** Krama Alus hanya sekitar 15% dari data train, sedangkan Ngoko dan Krama masing-masing sekitar 42%.
**Dampak:** model akan lebih jarang melihat Krama Alus. Karena itu, di bagian 5 kalimat Krama Alus di-oversample tiga
kali, dan ketepatan dilaporkan per tingkat, bukan hanya rata-rata.

### 4.2 Panjang kalimat
""")
code(r"""
semua = pd.concat([TRAIN, VALID, TEST])
semua["kata_jawa"] = semua.javanese.str.split().str.len()
semua["kata_indonesia"] = semua.indonesian.str.split().str.len()
fig, ax = plt.subplots(figsize=(7, 3.2))
bins = np.arange(0, 26)
for lv in LV:
    ax.hist(semua[semua.level == lv].kata_jawa, bins=bins, alpha=0.55, color=WARNA[lv], label=NAMA[lv])
ax.axvline(semua.kata_indonesia.mean(), color="black", ls="--", lw=1, label="rata-rata kalimat Indonesia")
ax.set_xlabel("jumlah kata per kalimat"); ax.set_ylabel("jumlah kalimat"); ax.legend(frameon=False)
ax.set_title("Sebaran panjang kalimat Jawa"); plt.tight_layout(); plt.show()
ringkas = semua.groupby("level").kata_jawa.agg(["mean", "median", "max"]).reindex(LV)
ringkas.index = [NAMA[l] for l in ringkas.index]
ringkas.columns = ["Rata-rata kata", "Median", "Maksimum"]
display(ringkas.round(2))
""")
md(r"""
**Temuan:** kalimat sangat pendek (rata-rata sekitar 8 kata, hampir tidak ada yang lebih dari 20 kata).
**Dampak:** panjang maksimum input dan output cukup 64 token, dan model tidak diuji untuk kalimat panjang atau
teks teknis. Ini dicatat sebagai keterbatasan.

### 4.3 Kualitas teks (noise)
""")
code(r"""
cek = []
for kol in ["indonesian", "javanese"]:
    s = df[kol].astype(str)
    cek += [(kol, "nilai kosong", int(df[kol].isna().sum())),
            (kol, "spasi di awal atau akhir", int((s != s.str.strip()).sum())),
            (kol, "spasi ganda", int(s.str.contains("  ").sum())),
            (kol, "bukan bentuk Unicode NFC", int((s != s.map(lambda t: unicodedata.normalize("NFC", t))).sum()))]
cek.append(("javanese", "baris identik (grup, tingkat, kalimat)", int(df.duplicated(["group", "label", "javanese"]).sum())))
noise = pd.DataFrame(cek, columns=["Kolom", "Jenis masalah", "Jumlah baris"])
noise["Persen dari seluruh baris"] = (100 * noise["Jumlah baris"] / len(df)).map(lambda v: koma(v, 2) + "%")
display(noise)
""")
md(r"""
**Temuan:** noise sangat sedikit (di bawah 1% baris).
**Dampak:** normalisasi teks hampir tidak mengubah data, sehingga tidak diharapkan memengaruhi hasil. Ini dibuktikan
di bagian 5.2.

### 4.4 Kosakata dan kata yang tidak muncul di data train
""")
code(r"""
kosakata_train = Counter(w for s in TRAIN.javanese for w in token(s))
baris = []
for nama, d in [("valid", VALID), ("test", TEST)]:
    for lv in LV + ["all"]:
        dd = d if lv == "all" else d[d.level == lv]
        tok = [w for s in dd.javanese for w in token(s)]
        baris.append((nama, NAMA[lv], len(tok), 100 * np.mean([w not in kosakata_train for w in tok])))
oov = pd.DataFrame(baris, columns=["Split", "Tingkat", "Jumlah token referensi", "Persen token tidak ada di train"])
oov["Persen token tidak ada di train"] = oov["Persen token tidak ada di train"].map(lambda v: koma(v, 1) + "%")
print("ukuran kosakata train:", len(kosakata_train), "kata unik")
display(oov)
""")
md(r"""
**Temuan:** sekitar seperlima token di kalimat referensi valid dan test tidak pernah muncul di data train.
**Dampak:** model harus menebak kata yang belum pernah dilihatnya dalam bahasa Jawa. Ini kandidat utama pembatas
kualitas, dan diuji lagi di bagian 7.2 (learning curve per ukuran data) dan 10.1 (F1-score per frekuensi kata).

### 4.5 Seberapa mirip tingkat-tingkat itu satu sama lain?

Untuk setiap grup yang punya dua tingkat, kalimat Jawa satu tingkat dibandingkan dengan tingkat lain memakai chrF++.
Nilai tinggi berarti kedua tingkat hanya berbeda sedikit kata.
""")
code(r"""
pasang = {("ngoko", "krama"): [], ("krama", "krama_alus"): [], ("ngoko", "krama_alus"): []}
for _, g in TRAIN.groupby("group"):
    jv = dict(zip(g.level, g.javanese))
    for a, b in pasang:
        if a in jv and b in jv:
            pasang[(a, b)].append(sacrebleu.sentence_chrf(jv[a], [jv[b]], word_order=2).score)
mirip = pd.DataFrame([(f"{NAMA[a]} vs {NAMA[b]}", len(v), np.mean(v)) for (a, b), v in pasang.items()],
                     columns=["Pasangan tingkat", "Jumlah grup", "Rata-rata chrF++ antar tingkat"])
mirip["Rata-rata chrF++ antar tingkat"] = mirip["Rata-rata chrF++ antar tingkat"].map(koma)
display(mirip)
""")
md(r"""
**Temuan:** Krama dan Krama Alus jauh lebih mirip satu sama lain daripada Ngoko dan Krama. Perbedaannya sering
hanya beberapa kata hormat (misalnya *sampeyan* → *panjenengan*, *nedha* → *dhahar*).
**Dampak:** kesalahan yang paling mungkin adalah Krama Alus yang "jatuh" menjadi Krama. Karena itu model diberi data
tambahan Krama → Krama Alus (multi-task, bagian 5.5), dan analisis memakai confusion matrix (bagian 9.4).

### 4.6 Kata khas tiap tingkat: wordcloud dibandingkan log-odds

Wordcloud menampilkan kata dengan ukuran sesuai frekuensinya. Kekurangannya: kata yang sering muncul di **kedua**
tingkat juga tampil besar, sehingga wordcloud tidak menunjukkan kata mana yang membedakan tingkat.
""")
code(r"""
from wordcloud import WordCloud
fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for ax, lv in zip(axs, ["krama", "krama_alus"]):
    frek = Counter(w for s in TRAIN[TRAIN.level == lv].javanese for w in token(s))
    wc = WordCloud(width=700, height=380, background_color="white", random_state=42, colormap="viridis")
    ax.imshow(wc.generate_from_frequencies(frek)); ax.axis("off"); ax.set_title(f"Wordcloud {NAMA[lv]} (frekuensi)")
plt.tight_layout(); plt.show()
""")
md(r"""
Kedua wordcloud di atas didominasi kata yang sama (misalnya *sampun*, *ingkang*, *boten*), sehingga perbedaan
Krama dan Krama Alus sulit dilihat. Sebagai gantinya dipakai **weighted log-odds dengan prior Dirichlet**
(Monroe dkk., 2008). Metode ini mengukur seberapa jauh frekuensi relatif sebuah kata berbeda di dua kelompok, lalu
membaginya dengan ketidakpastiannya, sehingga kata langka tidak mendapat skor berlebihan.

Supaya perbandingannya adil, hanya dipakai pasangan kalimat dari grup yang sama (isi kalimatnya sama, hanya
tingkatnya berbeda). Skor positif berarti khas Krama Alus, negatif berarti khas Krama.
""")
code(r"""
kunci = TRAIN[TRAIN.level.isin(["krama", "krama_alus"])].groupby(["group", "indonesian"])["level"].agg(list)
kunci = kunci[kunci.map(lambda x: sorted(x) == ["krama", "krama_alus"])].index
sub = TRAIN.set_index(["group", "indonesian"]).loc[kunci].reset_index()
print("jumlah pasangan Krama–Krama Alus dengan isi sama:", len(kunci))

def log_odds(teks_a, teks_b, alpha=0.1, min_frek=5):
    ca = Counter(w for s in teks_a for w in token(s)); cb = Counter(w for s in teks_b for w in token(s))
    prior = {w: alpha * (ca[w] + cb[w]) for w in set(ca) | set(cb)}
    na, nb, a0 = sum(ca.values()), sum(cb.values()), sum(prior.values())
    hasil = {}
    for w, a in prior.items():
        if ca[w] + cb[w] < min_frek: continue
        la = math.log((ca[w] + a) / (na + a0 - ca[w] - a)); lb = math.log((cb[w] + a) / (nb + a0 - cb[w] - a))
        hasil[w] = ((la - lb) / math.sqrt(1 / (ca[w] + a) + 1 / (cb[w] + a)), ca[w], cb[w])
    return hasil

ka = sub[sub.level == "krama_alus"].javanese; kr = sub[sub.level == "krama"].javanese
skor = log_odds(ka, kr)
urut = sorted(skor.items(), key=lambda kv: kv[1][0])
pilih = urut[:10] + urut[-10:]
fig, ax = plt.subplots(figsize=(7.5, 5.2))
ax.barh([w for w, _ in pilih], [v[0] for _, v in pilih],
        color=[WARNA["krama_alus"] if v[0] > 0 else WARNA["krama"] for _, v in pilih])
for i, (w, v) in enumerate(pilih):
    ax.text(v[0] + (0.2 if v[0] > 0 else -0.2), i, f"{v[1]} / {v[2]}", va="center", ha="left" if v[0] > 0 else "right", fontsize=8)
ax.axvline(0, color="black", lw=0.8)
ax.set_xlabel("z-score log-odds (positif = khas Krama Alus, negatif = khas Krama)")
ax.set_title("Kata paling khas: Krama Alus dibanding Krama\n(angka = frekuensi di Krama Alus / frekuensi di Krama)")
plt.tight_layout(); plt.show()

# cek kestabilan: skor dihitung ulang di dua belahan data, tanda harus sama
skor_belahan = []
for bagian in [kunci[0::2], kunci[1::2]]:
    sb = TRAIN.set_index(["group", "indonesian"]).loc[bagian].reset_index()
    skor_belahan.append(log_odds(sb[sb.level == "krama_alus"].javanese, sb[sb.level == "krama"].javanese, min_frek=1))
stabil = []
for w, v in urut[-5:][::-1] + urut[:5]:
    tanda = [np.sign(s.get(w, (0,))[0]) for s in skor_belahan]
    stabil.append((w, "Krama Alus" if v[0] > 0 else "Krama", "stabil" if tanda[0] == tanda[1] == np.sign(v[0]) else "tidak stabil"))
display(pd.DataFrame(stabil, columns=["Kata", "Khas untuk", "Arah skor di dua belahan data"]))
""")
md(r"""
**Temuan:** kata hormat seperti *panjenengan* dan *adalem* sama sekali tidak muncul di Krama (frekuensi 0), sedangkan
*kula* (aku, Krama) hampir hilang di Krama Alus. Pembeda tingkat ada pada **kata ganti dan kata kerja tertentu**,
bukan pada struktur kalimat.
**Dampak:** penanda tingkat adalah kata-kata pendek yang sering muncul. Preprocessing yang membuang kata umum
(stopword removal) akan membuang justru penanda ini (dibuktikan di bagian 5.3).
""")

# ====================================================================== 5. preprocessing
md(r"""
## 5. Preprocessing: sebelum dan sesudah

Pada penerjemahan mesin dengan model berbasis subword, preprocessing berbeda dari klasifikasi teks. Model memakai
tokenizer SentencePiece yang sudah menangani huruf, tanda baca, dan imbuhan, sehingga teks tidak perlu
"dibersihkan" secara agresif. Yang penting justru **cara membagi data** dan **cara menyusun data training**. Setiap
langkah di bawah ditunjukkan dalam bentuk sebelum dan sesudah.

### 5.1 Pembagian data: split acak dibanding split per grup

Satu kalimat Indonesia muncul di beberapa baris (satu per tingkat). Jika data dibagi secara acak per baris, kalimat
Indonesia yang sama bisa masuk ke train **dan** test sekaligus, sehingga model "sudah melihat" soal ujiannya.
""")
code(r"""
rng = np.random.default_rng(42)
d3 = df[df.label.isin([0, 2, 3])].reset_index(drop=True)
acak = rng.permutation(len(d3)); n_test = int(0.1 * len(d3))
uji_acak, latih_acak = d3.iloc[acak[:n_test]], d3.iloc[acak[n_test:]]
bocor_acak = uji_acak.indonesian.str.strip().str.lower().isin(set(latih_acak.indonesian.str.strip().str.lower())).mean()
bocor_grup = TEST.indonesian.str.strip().str.lower().isin(set(TRAIN.indonesian.str.strip().str.lower())).mean()
grup_sama = len(set(TEST.group) & set(TRAIN.group))
display(pd.DataFrame([
    ("Sebelum: split acak per baris (90/10)", koma(100 * bocor_acak, 1) + "%", "ya"),
    ("Sesudah: split per grup (dipakai)", koma(100 * bocor_grup, 1) + "%", f"tidak ({grup_sama} grup sama)"),
], columns=["Cara pembagian", "Kalimat Indonesia test yang juga ada di train", "Ada kebocoran?"]))
""")
md(r"""
**Hasil:** dengan split acak, sebagian besar kalimat test sudah ada di train, sehingga skor akan terlihat tinggi
padahal model hanya mengingat. Proyek ini memakai **split per grup** (`scripts/split_data.py`): satu grup utuh
masuk ke satu split saja. Akibatnya, skor proyek ini tidak bisa dibandingkan langsung dengan penelitian yang memakai
split acak (misalnya Prayoga dkk., 2026, chrF++ 56,39 pada M2M100), karena skor mereka kemungkinan terangkat oleh
kebocoran ini.

### 5.2 Normalisasi ringan teks
""")
code(r"""
def normalisasi(s):
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", str(s))).strip()
berubah = {kol: int((df[kol].astype(str) != df[kol].map(normalisasi)).sum()) for kol in ["indonesian", "javanese"]}
contoh_ubah = df[df.javanese.astype(str) != df.javanese.map(normalisasi)].javanese.head(3).tolist()
display(pd.DataFrame([(k, v, koma(100 * v / len(df), 2) + "%") for k, v in berubah.items()],
                     columns=["Kolom", "Baris yang berubah", "Persen"]))
for s in contoh_ubah:
    print("sebelum:", repr(s)); print("sesudah:", repr(normalisasi(s)))
""")
md(r"""
**Hasil:** normalisasi (Unicode NFC, merapikan spasi) hanya mengubah segelintir baris, dan perubahannya hanya spasi.
Tokenizer model sudah mengabaikan perbedaan spasi seperti ini, jadi langkah ini aman tetapi tidak berpengaruh pada skor.

### 5.3 Mengapa stopword removal, stemming, dan lowercase tidak dipakai

Di klasifikasi teks, kata umum (stopword) sering dibuang. Di proyek ini, kata umum justru membawa tingkat tutur.
Tabel berikut menampilkan 15 kata paling sering per tingkat, dan menandai mana yang merupakan **penanda tingkat**
(daftar penanda yang dipakai saat training).
""")
code(r"""
PENANDA = {"kowe", "aku", "awakmu", "mangan", "lunga", "turu", "sampeyan", "kula", "menika", "punika", "boten",
           "nedha", "kesah", "tilem", "panjenengan", "panjenenganipun", "dalem", "adalem", "dhahar", "tindak",
           "kagungan", "sare", "wis", "sampun", "ora", "mboten"}
kolom = {}
for lv in LV:
    frek = Counter(w for s in TRAIN[TRAIN.level == lv].javanese for w in token(s)).most_common(15)
    kolom[NAMA[lv]] = [f"{w} ★" if w in PENANDA else w for w, _ in frek]
display(pd.DataFrame(kolom, index=range(1, 16)))
jumlah = {NAMA[lv]: sum("★" in w for w in kolom[NAMA[lv]]) for lv in LV}
print("jumlah penanda tingkat di 15 kata tersering:", jumlah, " (★ = penanda tingkat)")
""")
md(r"""
**Hasil:** sebagian kata tersering adalah penanda tingkat (bertanda ★). Membuang stopword berarti membuang informasi
yang paling dibutuhkan. Alasan lain:

- **Stemming** menghapus imbuhan seperti *-ipun* dan *dipun-*, padahal imbuhan itu adalah ciri Krama
  (misalnya *dipunbekta*). Model juga harus *menghasilkan* kalimat lengkap, bukan hanya membaca.
- **Lowercase** tidak dipakai karena output harus berupa kalimat yang benar (huruf kapital di awal dan pada nama),
  dan chrF++ membedakan huruf besar dan kecil.

### 5.4 Source tag: menandai tingkat yang diminta

Tingkat yang diinginkan ditulis sebagai token khusus di awal kalimat Indonesia (Sennrich dkk., 2016). Embedding
setiap tag diinisialisasi dari rata-rata embedding beberapa kata penandanya, supaya model tidak mulai dari nol.
""")
code(r"""
TAG = {l: f"<{l}>" for l in LV}
INIT_WORDS = {"ngoko": "kowe aku mangan lunga turu", "krama": "sampeyan kula nedha kesah tilem",
              "krama_alus": "panjenenganipun dalem dhahar tindak sare"}
g = TRAIN.groupby("group").filter(lambda x: set(x.level) == set(LV)).groupby("group").head(3).head(3)
display(pd.DataFrame({"Sebelum (input asli)": g.indonesian.values,
                      "Sesudah (input ke model)": [f"{TAG[l]} {s}" for s, l in zip(g.indonesian, g.level)],
                      "Target Jawa": g.javanese.values}))
display(pd.DataFrame([(TAG[l], INIT_WORDS[l]) for l in LV], columns=["Tag", "Kata untuk inisialisasi embedding"]))
""")
md(r"""
### 5.5 Oversampling Krama Alus dan data multi-task

Dua perlakuan untuk mengatasi sedikitnya data Krama Alus:

1. **Oversampling 3 kali**: setiap pasangan Krama Alus dimasukkan tiga kali ke data training.
2. **Multi-task Krama → Krama Alus**: untuk grup yang punya Krama dan Krama Alus, ditambahkan pasangan
   *Jawa Krama → Jawa Krama Alus*. Model belajar langsung kata mana yang harus diganti agar menjadi lebih halus.
""")
code(r"""
OVERSAMPLE = 3
sebelum = TRAIN.level.value_counts().reindex(LV)
sesudah = sebelum.copy(); sesudah["krama_alus"] *= OVERSAMPLE
multitask = []
for _, g in TRAIN.groupby("group"):
    jv = dict(zip(g.level, g.javanese))
    if "krama" in jv and "krama_alus" in jv and jv["krama"] != jv["krama_alus"]:
        multitask.append((f"{TAG['krama_alus']} {jv['krama']}", jv["krama_alus"]))
tabel = pd.DataFrame({"Sebelum": sebelum, f"Sesudah oversampling {OVERSAMPLE} kali": sesudah}).rename(index=NAMA)
tabel.loc["Pasangan multi-task Krama → Krama Alus"] = [0, len(multitask)]
tabel.loc["Jumlah pasangan training"] = [sebelum.sum(), sesudah.sum() + len(multitask)]
display(tabel)
fig, ax = plt.subplots(figsize=(7, 3))
x = np.arange(3)
ax.bar(x - 0.2, sebelum.values, 0.4, color="#BBBBBB", label="sebelum")
ax.bar(x + 0.2, sesudah.values, 0.4, color=[WARNA[l] for l in LV], label="sesudah oversampling")
ax.set_xticks(x, [NAMA[l] for l in LV]); ax.set_ylabel("pasangan Indonesia → Jawa"); ax.legend(frameon=False)
ax.set_title("Data training sebelum dan sesudah oversampling"); plt.tight_layout(); plt.show()
print("contoh pasangan multi-task:"); print("  input :", multitask[0][0]); print("  target:", multitask[0][1])
""")
md(r"""
### 5.6 Ringkasan preprocessing

| Langkah | Sebelum | Sesudah | Dipakai? | Alasan |
|---|---|---|---|---|
| Pembagian data | split acak per baris | split per grup | ya | mencegah kalimat test bocor ke train |
| Normalisasi spasi dan Unicode | teks asli | teks rapi | ya (tidak berpengaruh) | noise sangat sedikit |
| Stopword removal, stemming, lowercase | – | – | tidak | membuang penanda tingkat; output harus kalimat lengkap |
| Source tag | `Kamu sudah makan?` | `<krama_alus> Kamu sudah makan?` | ya | cara memilih tingkat |
| Oversampling Krama Alus | 1 kali | 3 kali | ya | Krama Alus paling sedikit |
| Multi-task Krama → Krama Alus | tidak ada | ada | ya | Krama Alus paling mirip Krama |

Pengaruh setiap langkah terhadap hasil model diuji dengan melatih model sebelum dan sesudah langkah itu (bagian 6),
lalu dibandingkan di bagian 9.
""")

# ====================================================================== 6. modelling
md(r"""
## 6. Modelling

### 6.1 Rencana eksperimen: setiap run menambah satu perlakuan

Untuk melihat pengaruh setiap langkah preprocessing, notebook ini melatih **lima model M2M100-418M** dengan resep yang
sama (15 epoch, seed 42), lalu **dua model lain** (NLLB dan Cendol) dengan resep terbaik. Setiap run hanya berbeda satu hal dari run sebelumnya, sehingga perbedaan hasilnya
bisa dikaitkan ke langkah itu.

| Run | Pembagian data | Source tag | Inisialisasi tag + oversampling Krama Alus | Multi-task Krama → Krama Alus | Yang diuji |
|---|---|---|---|---|---|
| A | **acak per baris** | ya | tidak | tidak | dampak kebocoran data (dibanding B) |
| B | per grup | ya | tidak | tidak | titik awal yang jujur |
| C | per grup | **tidak** | tidak | tidak | dampak source tag (dibanding B) |
| D | per grup | ya | **ya** | tidak | dampak inisialisasi + oversampling (dibanding B) |
| E | per grup | ya | ya | **ya** | dampak multi-task (dibanding D) = resep final |
| E + rerank | sama dengan E | | | | dampak rerank saat decoding (dibanding E) |
| F + rerank | resep E, model **NLLB-200-600M** | | | | perbandingan model (dibanding E + rerank) |
| G + rerank | resep E, model **Cendol mT5-base** | | | | perbandingan model (dibanding E + rerank) |

Semua pemilihan (epoch terbaik dan bobot rerank) dilakukan di **data validasi**. Data test hanya dipakai di akhir.

Eksperimen preprocessing (run A–E) memakai M2M100-418M (Fan dkk., 2021) karena model ini sudah mendukung bahasa
Indonesia dan Jawa dan cukup ringan sehingga lima run bisa dilatih dalam waktu GPU yang wajar. Model dasar mana yang
terbaik tidak ditetapkan di awal, tetapi ditentukan di bagian 8 dengan skor validasi setelah NLLB dan Cendol dilatih
dengan resep yang sama.

### 6.2 Hyperparameter dan fungsi bantu

Notebook ini membutuhkan GPU (Kaggle: *Settings → Accelerator → GPU T4*). Satu run memakan sekitar 15 menit.
""")
code(r"""
import gc, time
from transformers import AutoModelForSeq2SeqLM, get_linear_schedule_with_warmup

MODEL_DASAR = "facebook/m2m100_418M"
EPOCHS, OVERSAMPLE, BATCH, LR, WARMUP, SEED, MAXLEN, BOBOT_TINGKAT = 15, 3, 16, 5e-5, 0.1, 42, 64, 30.0
N_KANDIDAT, DAFTAR_LAMBDA = 8, [0.5, 1.0, 2.0, 4.0]
OUT = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.path.abspath("keluaran")
os.makedirs(OUT, exist_ok=True)
CEPAT = os.environ.get("NOTEBOOK_CEPAT") == "1"   # hanya untuk uji cepat: 1 epoch, data kecil
if CEPAT:
    EPOCHS = 1
else:
    assert torch.cuda.is_available(), "Aktifkan GPU: Settings -> Accelerator -> GPU T4"
PAKAI_FP16 = DEV == "cuda"
T0 = time.time()
def catat(*a): print(f"[{(time.time() - T0) / 60:5.1f} menit]", *a, flush=True)
def seed_all(s):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
TAG = {l: f"<{l}>" for l in LV}
INIT_WORDS = {"ngoko": "kowe aku mangan lunga turu", "krama": "sampeyan kula nedha kesah tilem",
              "krama_alus": "panjenenganipun dalem dhahar tindak sare"}
print("perangkat:", torch.cuda.get_device_name(0) if PAKAI_FP16 else "CPU", "| folder keluaran:", OUT, "| mode cepat:", CEPAT)
""")
md(r"""
### 6.3 Split acak untuk run A

Run A memakai pembagian acak per baris dengan ukuran yang sama dengan split per grup (train 1.395, validasi 175,
test 176). Ini **sengaja dibuat salah** untuk menunjukkan dampak kebocoran.
""")
code(r"""
gabung = pd.concat([TRAIN, VALID, TEST]).reset_index(drop=True)
urut = np.random.default_rng(42).permutation(len(gabung))
TEST_ACAK = gabung.iloc[urut[:176]].reset_index(drop=True)
VALID_ACAK = gabung.iloc[urut[176:351]].reset_index(drop=True)
TRAIN_ACAK = gabung.iloc[urut[351:]].reset_index(drop=True)
bocor = TEST_ACAK.indonesian.isin(set(TRAIN_ACAK.indonesian)).mean()
print(f"split acak: train {len(TRAIN_ACAK)}, validasi {len(VALID_ACAK)}, test {len(TEST_ACAK)}; "
      f"kalimat Indonesia test yang juga ada di train: {koma(100 * bocor, 1)}%")
""")
md(r"""
### 6.4 Classifier untuk pemilihan checkpoint dan untuk rerank

Selama training dibutuhkan dua classifier tingkat yang dilatih di sini dari data train:

- **Classifier pemilih** (`w11wo/javanese-distilbert-small`): menilai ketepatan tingkat di data validasi untuk memilih
  epoch terbaik.
- **Classifier rerank** (`w11wo/javanese-roberta-small`): memilih ulang 8 kandidat terjemahan.

Keduanya **berbeda** dari classifier penilai tetap (bagian 2.1) yang dipakai untuk melaporkan hasil test. Jika
classifier yang sama dipakai untuk memilih dan menilai, angka ketepatan akan naik palsu.
""")
code(r"""
def latih_classifier(nama_model, seed):
    ctok = AutoTokenizer.from_pretrained(nama_model)
    cmod = AutoModelForSequenceClassification.from_pretrained(nama_model, num_labels=3).to(DEV)
    seed_all(seed); opt = torch.optim.AdamW(cmod.parameters(), lr=5e-5)
    X, y = TRAIN.javanese.tolist(), TRAIN.level.map({l: i for i, l in enumerate(LV)}).tolist()
    for _ in range(4):
        cmod.train(); urutan = list(range(len(X))); random.shuffle(urutan)
        for i in range(0, len(urutan), 32):
            b = urutan[i:i + 32]
            enc = ctok([X[j] for j in b], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
            loss = cmod(**enc, labels=torch.tensor([y[j] for j in b], device=DEV)).loss
            opt.zero_grad(); loss.backward(); opt.step()
    cmod.eval()
    def log_prob(kalimat):
        out = []
        for i in range(0, len(kalimat), 64):
            enc = ctok([str(x) for x in kalimat[i:i + 64]], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN).to(DEV)
            with torch.no_grad():
                out.append(F.log_softmax(cmod(**enc).logits.float(), -1).cpu())
        return torch.cat(out).numpy()
    akurasi = (log_prob(VALID.javanese.tolist()).argmax(-1) == VALID.level.map({l: i for i, l in enumerate(LV)}).values).mean()
    catat(f"classifier {nama_model}: akurasi pada referensi validasi {koma(akurasi, 3)}")
    return log_prob

PEMILIH = latih_classifier("w11wo/javanese-distilbert-small", SEED)
RERANKER = latih_classifier("w11wo/javanese-roberta-small", 7)
""")
md(r"""
### 6.5 Fungsi training

Satu fungsi `latih_run` dipakai untuk kelima run. Perlakuan data diatur lewat argumen. Setiap epoch dicatat:
**train loss**, **validation loss**, chrF++ validasi, dan ketepatan tingkat validasi. Checkpoint dengan skor
validasi tertinggi (chrF++ + 30 × ketepatan tingkat) disimpan.
""")
code(r"""
KELUARGA = {  # nama model, kode bahasa sumber, kode bahasa target, learning rate, boleh fp16, panjang maksimum, akumulasi gradien
    "m2m": ("facebook/m2m100_418M", "id", "jv", 5e-5, True, 64, 1),
    "nllb": ("facebook/nllb-200-distilled-600M", "ind_Latn", "jav_Latn", 5e-5, True, 64, 2),
    "cendol": ("indonlp/cendol-mt5-base-inst", None, None, 3e-4, False, 128, 2),  # mT5 tidak stabil di fp16
}
# Akumulasi gradien 2: batch 16 diproses sebagai 2 x 8 supaya muat di GPU T4 (setara batch 16).
PROMPT_CENDOL = "Terjemahkan dari bahasa Indonesia ke bahasa Jawa {}: "
MAXLEN_AKTIF, AMP_AKTIF = MAXLEN, PAKAI_FP16

def muat_model(keluarga, pakai_tag, init_tag):
    nama, sl, tl = KELUARGA[keluarga][:3]
    tok = AutoTokenizer.from_pretrained(nama, src_lang=sl, tgt_lang=tl) if sl else AutoTokenizer.from_pretrained(nama)
    model = AutoModelForSeq2SeqLM.from_pretrained(nama)
    if pakai_tag:
        tok.add_tokens(list(TAG.values()), special_tokens=True)
        if len(tok) > model.get_input_embeddings().weight.shape[0]:
            model.resize_token_embeddings(len(tok))
        if init_tag:
            emb = model.get_input_embeddings().weight
            with torch.no_grad():
                for l, kata in INIT_WORDS.items():
                    ids = tok(kata, add_special_tokens=False)["input_ids"]
                    emb[tok.convert_tokens_to_ids(TAG[l])] = emb[ids].mean(0)
    bos = None if tl is None else (tok.get_lang_id(tl) if hasattr(tok, "get_lang_id") else tok.convert_tokens_to_ids(tl))
    return tok, model.to(DEV), bos

def terjemahkan(tok, model, bos, src, beams=4, bs=48):
    model.eval(); out = [None] * len(src); urutan = sorted(range(len(src)), key=lambda i: len(src[i]))
    for i in range(0, len(urutan), bs):
        idx = urutan[i:i + bs]
        enc = tok([src[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN_AKTIF).to(DEV)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=AMP_AKTIF):
            g = model.generate(**enc, forced_bos_token_id=bos, num_beams=beams, max_new_tokens=MAXLEN_AKTIF)
        for j, t in zip(idx, tok.batch_decode(g, skip_special_tokens=True)):
            out[j] = t
    return out

def loss_batch(tok, model, src, tgt, label_smoothing=0.0):
    enc = tok(src, text_target=tgt, return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN_AKTIF).to(DEV)
    labels = enc.pop("labels"); labels[labels == tok.pad_token_id] = -100
    with torch.autocast(DEV, dtype=torch.float16, enabled=AMP_AKTIF):
        logits = model(**enc, labels=labels).logits
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100, label_smoothing=label_smoothing)

def skor_kandidat(tok, model, src, kandidat, chunk=16):
    # rata-rata log-prob token setiap kandidat (teacher forcing)
    out = []
    for i in range(0, len(src), chunk):
        enc = tok(src[i:i + chunk], text_target=kandidat[i:i + chunk], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN_AKTIF).to(DEV)
        labels = enc.pop("labels"); labels[labels == tok.pad_token_id] = -100
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=AMP_AKTIF):
            logits = model(**enc, labels=labels).logits
        nll = F.cross_entropy(logits.float().transpose(1, 2), labels, ignore_index=-100, reduction="none")
        m = (labels != -100).float(); out += (-(nll * m).sum(1) / m.sum(1).clamp(min=1)).cpu().tolist()
    return np.array(out)

def n_terbaik(tok, model, bos, src, k=N_KANDIDAT, bs=8):
    model.eval(); out = [None] * len(src); urutan = sorted(range(len(src)), key=lambda i: len(src[i]))
    for i in range(0, len(urutan), bs):
        idx = urutan[i:i + bs]
        enc = tok([src[j] for j in idx], return_tensors="pt", padding=True, truncation=True, max_length=MAXLEN_AKTIF).to(DEV)
        with torch.no_grad(), torch.autocast(DEV, dtype=torch.float16, enabled=AMP_AKTIF):
            g = model.generate(**enc, forced_bos_token_id=bos, num_beams=k, num_return_sequences=k, max_new_tokens=MAXLEN_AKTIF)
        teks = tok.batch_decode(g, skip_special_tokens=True)
        sk = skor_kandidat(tok, model, [src[j] for j in idx for _ in range(k)], teks)
        for n, j in enumerate(idx):
            out[j] = (teks[n * k:(n + 1) * k], sk[n * k:(n + 1) * k])
    return out

def rerank(kandidat, tingkat, lam):
    datar = [t for c, _ in kandidat for t in c]; lp = RERANKER(datar); hasil = []; p = 0
    for (c, s), l in zip(kandidat, tingkat):
        k = len(c); total = s + lam * lp[p:p + k, LV.index(l)]; p += k
        hasil.append(c[int(np.argmax(total))])
    return hasil

def skor_validasi(hyp, valid_df):
    c = chrf(hyp, valid_df.javanese.tolist())
    pred = PEMILIH(hyp).argmax(-1); y = valid_df.level.map({l: i for i, l in enumerate(LV)}).values
    ket = (pred == y).mean(); ket_ka = (pred[y == 2] == 2).mean()
    return c + BOBOT_TINGKAT * ket, c, ket, ket_ka

def latih_run(nama, train_df, valid_df, test_df, pakai_tag=True, init_tag=False, oversample=1, multitask=False, pakai_rerank=False,
              keluarga="m2m"):
    global MAXLEN_AKTIF, AMP_AKTIF
    _, sl, tl, lr, boleh_fp16, MAXLEN_AKTIF, akumulasi = KELUARGA[keluarga]; AMP_AKTIF = PAKAI_FP16 and boleh_fp16
    mikro = BATCH // akumulasi
    gc.collect(); torch.cuda.empty_cache()
    seed_all(SEED); catat(f"=== Run {nama} dimulai ({KELUARGA[keluarga][0]})")
    if CEPAT:
        train_df, valid_df, test_df = train_df.head(96), valid_df.head(48), test_df.head(48)
    tok, model, bos = muat_model(keluarga, pakai_tag, init_tag)
    if keluarga == "cendol":
        sumber = lambda s, l: f"{TAG[l]} {PROMPT_CENDOL.format(NAMA[l])}{s}"
    else:
        sumber = lambda s, l: f"{TAG[l]} {s}" if pakai_tag else s
    pasangan = [(sumber(s, l), t, l) for s, l, t in zip(train_df.indonesian, train_df.level, train_df.javanese)]
    id_jv = [(s, t) for s, t, _ in pasangan] + [(s, t) for s, t, l in pasangan if l == "krama_alus"] * (oversample - 1)
    jv_jv = []
    if multitask:
        for _, g in train_df.groupby("group"):
            jv = dict(zip(g.level, g.javanese))
            if "krama" in jv and "krama_alus" in jv and jv["krama"] != jv["krama_alus"]:
                jv_jv.append((f"{TAG['krama_alus']} {jv['krama']}", jv["krama_alus"]))
    catat(f"pasangan Indonesia→Jawa: {len(id_jv)}, pasangan Krama→Krama Alus: {len(jv_jv)}")
    def buat_batch():
        a, b = id_jv[:], jv_jv[:]; random.shuffle(a); random.shuffle(b)
        bt = [(sl, a[i:i + BATCH]) for i in range(0, len(a), BATCH)] + [(tl, b[i:i + BATCH]) for i in range(0, len(b), BATCH)]
        random.shuffle(bt); return bt
    langkah = EPOCHS * len(buat_batch())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(WARMUP * langkah), langkah)
    scaler = torch.amp.GradScaler(DEV, enabled=AMP_AKTIF)
    v_src = [sumber(s, l) for s, l in zip(valid_df.indonesian, valid_df.level)]
    terbaik, state_terbaik, riwayat = -1, None, []
    for ep in range(1, EPOCHS + 1):
        model.train(); total = n = 0
        for bahasa, b in buat_batch():
            if sl: tok.src_lang = bahasa
            opt.zero_grad()
            for j in range(0, len(b), mikro):
                mb = b[j:j + mikro]
                loss = loss_batch(tok, model, [s for s, _ in mb], [t for _, t in mb])
                scaler.scale(loss * len(mb) / len(b)).backward()
                total += loss.item() * len(mb); n += len(mb)
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sched.step()
        if sl: tok.src_lang = sl
        model.eval(); vl = vn = 0
        with torch.no_grad():
            for i in range(0, len(v_src), mikro):
                bt = list(zip(v_src, valid_df.javanese))[i:i + mikro]
                vl += loss_batch(tok, model, [s for s, _ in bt], [t for _, t in bt]).item() * len(bt); vn += len(bt)
        skor, c, ket, ket_ka = skor_validasi(terjemahkan(tok, model, bos, v_src, beams=1), valid_df)
        riwayat.append({"epoch": ep, "train_loss": total / n, "valid_loss": vl / vn, "valid_chrf": c,
                        "valid_ketepatan": ket, "valid_ketepatan_krama_alus": ket_ka, "skor": skor})
        catat(f"{nama} epoch {ep:2d} | train loss {total / n:.4f} | validation loss {vl / vn:.4f} | "
              f"chrF++ validasi {c:.2f} | ketepatan {ket:.3f} | skor {skor:.2f}")
        if skor > terbaik:
            terbaik = skor; state_terbaik = {k: x.detach().to("cpu", copy=True) for k, x in model.state_dict().items()}
    model.load_state_dict(state_terbaik); del state_terbaik, opt, scaler, sched; gc.collect(); torch.cuda.empty_cache()
    hasil = {"nama": nama, "keluarga": keluarga, "riwayat": pd.DataFrame(riwayat), "epoch_terpilih": max(riwayat, key=lambda r: r["skor"])["epoch"]}
    t_src = [sumber(s, l) for s, l in zip(test_df.indonesian, test_df.level)]
    unik = sorted(test_df.indonesian.unique()); permintaan = [(s, l) for s in unik for l in LV]
    k_src = [sumber(s, l) for s, l in permintaan]
    hasil["test"] = terjemahkan(tok, model, bos, t_src, beams=4)
    hasil["kontrol"] = pd.DataFrame({"indonesian": [s for s, _ in permintaan], "requested": [l for _, l in permintaan],
                                     "output": terjemahkan(tok, model, bos, k_src, beams=4)})
    if pakai_rerank:
        kand_v = n_terbaik(tok, model, bos, v_src)
        tabel_lam = {"beam search 4 (tanpa rerank)": skor_validasi(terjemahkan(tok, model, bos, v_src, beams=4), valid_df)}
        for lam in DAFTAR_LAMBDA:
            tabel_lam[f"rerank, lambda {lam}"] = skor_validasi(rerank(kand_v, valid_df.level.tolist(), lam), valid_df)
        hasil["tabel_lambda"] = tabel_lam
        lam = max(DAFTAR_LAMBDA, key=lambda x: tabel_lam[f"rerank, lambda {x}"][0]); hasil["lambda"] = lam
        catat(f"{nama}: lambda terpilih di validasi = {lam}")
        hasil["test_rerank"] = rerank(n_terbaik(tok, model, bos, t_src), test_df.level.tolist(), lam)
        hasil["kontrol_rerank"] = hasil["kontrol"].assign(output=rerank(n_terbaik(tok, model, bos, k_src), [l for _, l in permintaan], lam))
    hasil["test_df"] = test_df.reset_index(drop=True)
    pd.DataFrame(riwayat).to_csv(f"{OUT}/riwayat_run_{nama}.csv", index=False)
    test_df.assign(hyp=hasil["test"]).to_csv(f"{OUT}/prediksi_run_{nama}.csv", index=False)
    hasil["kontrol"].to_csv(f"{OUT}/kontrol_run_{nama}.csv", index=False)
    if pakai_rerank:
        test_df.assign(hyp=hasil["test_rerank"]).to_csv(f"{OUT}/prediksi_run_{nama}_rerank.csv", index=False)
        hasil["kontrol_rerank"].to_csv(f"{OUT}/kontrol_run_{nama}_rerank.csv", index=False)
    del model; gc.collect(); torch.cuda.empty_cache()
    catat(f"=== Run {nama} selesai, epoch terpilih {hasil['epoch_terpilih']}")
    return hasil

RUN = {}
""")
md(r"""
### 6.6 Menjalankan kelima run

Setiap sel di bawah melatih satu model (sekitar 15 menit di GPU T4). Log per epoch dicetak selama training.
""")
code(r"""
RUN["A"] = latih_run("A", TRAIN_ACAK, VALID_ACAK, TEST_ACAK, pakai_tag=True)
""")
code(r"""
RUN["B"] = latih_run("B", TRAIN, VALID, TEST, pakai_tag=True)
""")
code(r"""
RUN["C"] = latih_run("C", TRAIN, VALID, TEST, pakai_tag=False)
""")
code(r"""
RUN["D"] = latih_run("D", TRAIN, VALID, TEST, pakai_tag=True, init_tag=True, oversample=OVERSAMPLE)
""")
code(r"""
RUN["E"] = latih_run("E", TRAIN, VALID, TEST, pakai_tag=True, init_tag=True, oversample=OVERSAMPLE, multitask=True, pakai_rerank=True)
""")
md(r"""
### 6.7 Model lain dengan resep yang sama

Sebelum memilih M2M100, proyek ini juga mencoba model lain. Supaya perbandingannya adil, keduanya dilatih di sini
dengan resep run E (tag, inisialisasi, oversampling, multi-task, rerank):

- **F: NLLB-200-distilled-600M** (NLLB Team, 2024), model penerjemah multibahasa yang lebih besar.
- **G: Cendol mT5-base** (Cahyawijaya dkk., 2024), model instruksi untuk bahasa Indonesia dan daerah. Input diberi
  instruksi yang menyebut nama tingkat, learning rate 3 × 10⁻⁴, dan tanpa fp16 karena mT5 tidak stabil di fp16.
""")
code(r"""
RUN["F"] = latih_run("F", TRAIN, VALID, TEST, pakai_tag=True, init_tag=True, oversample=OVERSAMPLE, multitask=True, pakai_rerank=True,
                     keluarga="nllb")
""")
code(r"""
RUN["G"] = latih_run("G", TRAIN, VALID, TEST, pakai_tag=True, init_tag=True, oversample=OVERSAMPLE, multitask=True, pakai_rerank=True,
                     keluarga="cendol")
""")

# ====================================================================== 7. learning curve
md(r"""
## 7. Learning curve

### 7.1 Learning curve per epoch untuk kelima run

Grafik ini dibuat dari log training di atas. Garis putus-putus vertikal menandai epoch yang terpilih di setiap run.
""")
code(r"""
NAMA_RUN = {"A": "A: split acak", "B": "B: split per grup + tag", "C": "C: tanpa tag",
            "D": "D: + inisialisasi + oversampling", "E": "E: + multi-task (resep final)", "F": "F: NLLB-600M, resep E",
            "G": "G: Cendol mT5-base, resep E"}
WARNA_RUN = {"A": "#999999", "B": "#4C78A8", "C": "#E45756", "D": "#F58518", "E": "#2E8B57", "F": "#8E44AD", "G": "#8C564B"}
fig, axs = plt.subplots(2, 2, figsize=(12, 7.5), sharex=True)
for kode, r in RUN.items():
    h = r["riwayat"]
    for ax, kol, judul in [(axs[0, 0], "train_loss", "Train loss"), (axs[0, 1], "valid_loss", "Validation loss"),
                           (axs[1, 0], "valid_chrf", "chrF++ validasi (greedy)"), (axs[1, 1], "valid_ketepatan", "Ketepatan tingkat validasi")]:
        ax.plot(h.epoch, h[kol], "-o", ms=3, color=WARNA_RUN[kode], label=NAMA_RUN[kode])
        ax.axvline(r["epoch_terpilih"], color=WARNA_RUN[kode], ls=":", lw=0.8); ax.set_title(judul)
axs[0, 0].set_yscale("log"); axs[0, 1].set_yscale("log")
axs[1, 0].set_xlabel("epoch"); axs[1, 1].set_xlabel("epoch"); axs[0, 0].legend(frameon=False, fontsize=8)
plt.suptitle("Learning curve per epoch (run A dinilai pada validasi acaknya sendiri)"); plt.tight_layout(); plt.show()
display(pd.DataFrame([(NAMA_RUN[k], r["epoch_terpilih"], koma(r["riwayat"].train_loss.iloc[-1], 4),
                       koma(r["riwayat"].valid_loss.min(), 4), int(r["riwayat"].valid_loss.idxmin()) + 1,
                       koma(r["riwayat"].valid_chrf.max()))
                      for k, r in RUN.items()],
                     columns=["Run", "Epoch terpilih", "Train loss akhir", "Validation loss terendah", "Epoch validation loss terendah",
                              "chrF++ validasi tertinggi"]))
""")
md(r"""
**Cara membaca:**

- **Train loss** terus turun sampai mendekati nol, artinya model makin hafal data train.
- **Validation loss** biasanya mencapai titik terendah lebih awal lalu naik lagi, tanda model mulai **overfitting**
  dalam hal probabilitas kata. Namun chrF++ validasi sering tetap naik atau datar sesudah titik itu, karena chrF++
  hanya menilai kalimat terbaik yang dihasilkan, bukan seluruh distribusi probabilitas. Karena itu checkpoint dipilih
  dengan chrF++ dan ketepatan tingkat, bukan dengan validation loss.
- **Ketepatan tingkat** run C (tanpa tag) tertahan di sekitar sepertiga, karena model tidak tahu tingkat mana yang
  diminta. Run dengan tag naik tajam di beberapa epoch pertama: kontrol tingkat dipelajari jauh lebih cepat daripada
  kualitas terjemahan.
- Run A punya kurva validasi paling tinggi, tetapi itu karena validasi acaknya bocor (bagian 5.1).

### 7.2 Learning curve per ukuran data (eksperimen sebelumnya)

Bagian ini memakai hasil eksperimen terpisah (kernel `unggah-ungguh-learning-curve-v2`, 7 run) karena melatih ulang
semuanya membutuhkan sekitar 2 jam GPU lagi. Resep final dilatih dengan 25%, 50%, 75%, dan 100% data train (subset
per grup, seimbang per tingkat). chrF++ dihitung ulang dari prediksi tersimpan, dan ketepatan tingkat dinilai oleh
classifier penilai tetap.
""")

code(r"""
lc_lokal = json.load(open(f"{K}/learning_curve_v2/analisis_lokal.json", encoding="utf-8"))
baris = []
for f in sorted(glob.glob(f"{K}/learning_curve_v2/output/valid_f*.csv")):
    nama = os.path.basename(f)[6:-4]; d = pd.read_csv(f); d = d[d.level.isin(LV)]
    h = d.hyp.fillna("").astype(str).values; r = d.javanese.values
    ok = vonis_juri(h) == d.level.values
    baris.append({"run": nama, "persen": int(round(100 * lc_lokal[nama]["frac"])), "kalimat_train": lc_lokal[nama]["n_kalimat"],
                  "all": chrf(h, r), **{lv: chrf(h[d.level == lv], r[d.level == lv]) for lv in LV},
                  "ketepatan": ok.mean(), "ketepatan_krama_alus": ok[d.level.values == "krama_alus"].mean(),
                  "token_tidak_di_train": 100 * lc_lokal[nama]["oov_ref_token"]["all"]})
    assert abs(baris[-1]["all"] - lc_lokal[nama]["chrF++"]["all"]) < 0.02, nama
LC = pd.DataFrame(baris)
rata = LC.groupby("kalimat_train").mean(numeric_only=True).reset_index()

fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
for lv in ["all"] + LV:
    axs[0].plot(rata.kalimat_train, rata[lv], "-o", color=WARNA[lv], lw=2 if lv == "all" else 1.2, ms=4, label=NAMA[lv])
    axs[0].scatter(LC.kalimat_train, LC[lv], color=WARNA[lv], s=10, alpha=0.4)
axs[0].set_xscale("log", base=2); axs[0].set_xticks(rata.kalimat_train, [f"{n}" for n in rata.kalimat_train])
axs[0].set_xlabel("jumlah kalimat train"); axs[0].set_ylabel("chrF++ validasi"); axs[0].legend(frameon=False, fontsize=8)
axs[0].set_title("Kualitas terjemahan terus naik")
axs[1].plot(rata.kalimat_train, 100 * rata.ketepatan_krama_alus, "-s", color=WARNA["krama_alus"], label="ketepatan Krama Alus")
axs[1].plot(rata.kalimat_train, rata.token_tidak_di_train, "-^", color="#777777", label="token referensi yang tidak ada di train")
axs[1].set_xscale("log", base=2); axs[1].set_xticks(rata.kalimat_train, [f"{n}" for n in rata.kalimat_train])
axs[1].set_ylim(0, 105); axs[1].set_xlabel("jumlah kalimat train"); axs[1].set_ylabel("persen"); axs[1].legend(frameon=False, fontsize=8)
axs[1].set_title("Ketepatan tingkat sudah datar sejak 25% data")
plt.tight_layout(); plt.show()
tampil = rata[["persen", "kalimat_train", "all", "krama_alus", "ketepatan_krama_alus", "token_tidak_di_train"]].copy()
tampil.columns = ["Persen data", "Kalimat train", "chrF++ validasi", "chrF++ Krama Alus", "Ketepatan Krama Alus", "Persen token tidak ada di train"]
display(tampil.round(2))
""")
md(r"""
**Cara membaca:** setiap kali data train digandakan, chrF++ validasi naik sekitar 7 poin dan belum menunjukkan tanda
jenuh. Sebaliknya, ketepatan Krama Alus sudah tinggi sejak 25% data. Bersamaan dengan itu, porsi token referensi yang
tidak pernah dilihat model turun dari sekitar 38% menjadi 23%. Artinya, model sudah bisa **mengontrol tingkat** dengan
sedikit data, tetapi masih kekurangan **kosakata** untuk menerjemahkan isi kalimat dengan benar.

### 7.3 Kontrol: apakah kenaikan itu karena data, atau karena jumlah langkah training?

Pada learning curve 7.2, subset kecil juga mendapat langkah training lebih sedikit (epoch sama, batch lebih sedikit).
Jadi kenaikan chrF++ bisa saja karena model dengan data penuh **dilatih lebih lama**, bukan karena datanya lebih banyak
(Sennrich & Zhang, 2019). Untuk mengujinya, subset 50% dilatih dengan **jumlah langkah yang sama** dengan data penuh
(1.905 langkah).
""")
code(r"""
kl = json.load(open(f"{K}/lc_kontrol_langkah_v1/output/manifest.json", encoding="utf-8"))
kl_lokal = json.load(open(f"{K}/lc_kontrol_langkah_v1/analisis_lokal.json", encoding="utf-8"))
baris = []
for nama, label in [("f050_s0", "50% data, subset 1"), ("f050_s1", "50% data, subset 2"), ("f100", "100% data")]:
    d = pd.read_csv(f"{K}/lc_kontrol_langkah_v1/output/valid_{nama}_akhir.csv"); d = d[d.level.isin(LV)]
    c = chrf(d.hyp.fillna("").astype(str).values, d.javanese.values)
    assert abs(c - kl["runs"][nama]["akhir"]["valid_rerank"]["per_level"]["all"]["chrF++"]) < 0.02
    baris.append((label, kl["runs"][nama]["steps_done"], c))
tabel = pd.DataFrame(baris, columns=["Run", "Langkah training", "chrF++ validasi"])
fig, ax = plt.subplots(figsize=(6, 3))
b = ax.bar(tabel.Run, tabel["chrF++ validasi"], color=["#9DB4CC", "#9DB4CC", "#4C78A8"])
ax.bar_label(b, labels=[koma(x) for x in tabel["chrF++ validasi"]]); ax.set_ylim(50, 66)
ax.set_ylabel("chrF++ validasi"); ax.set_title("Jumlah langkah training disamakan (1.905)"); plt.tight_layout(); plt.show()
tabel["chrF++ validasi"] = tabel["chrF++ validasi"].map(koma)
display(tabel)
for nama, label in [("f050_s0", "subset 1"), ("f050_s1", "subset 2")]:
    lo, hi = kl_lokal[nama]["gap_boot_CI90"]
    print(f"selisih 100% data terhadap 50% data ({label}): {koma(kl['gap_akhir'][nama])} poin chrF++, "
          f"interval kepercayaan 90% (bootstrap per grup): {koma(lo)} sampai {koma(hi)}")
""")
md(r"""
**Hasil:** walaupun jumlah langkah disamakan, model dengan data penuh tetap unggul sekitar 5–8 poin chrF++. Jadi
keunggulan itu memang berasal dari **jumlah data**, bukan dari training yang lebih lama. Batasan: satu seed, hanya
data validasi, dan jumlah token per langkah tidak disamakan.
""")


# ====================================================================== 8. pemilihan
md(r"""
## 8. Pemilihan checkpoint dan rerank

Semua keputusan diambil di **data validasi**: epoch terbaik setiap run, bobot rerank, dan pilihan model (M2M100,
NLLB, atau Cendol). Data test belum dilihat pada tahap ini.
""")
code(r"""
display(pd.DataFrame([(NAMA_RUN[k], r["epoch_terpilih"], koma(r["riwayat"].skor.max()),
                       koma(r["riwayat"].loc[r["epoch_terpilih"] - 1, "valid_ketepatan_krama_alus"], 3))
                      for k, r in RUN.items()],
                     columns=["Run", "Epoch terpilih", "Skor validasi (chrF++ + 30 × ketepatan)", "Ketepatan Krama Alus validasi"]))
for k in ["E", "F", "G"]:
    print(f"Pemilihan bobot rerank untuk {NAMA_RUN[k]} (data validasi), lambda terpilih {koma(RUN[k]['lambda'], 1)}:")
    display(pd.DataFrame([(d.replace("lambda", "λ ="), koma(v[1]), koma(v[2], 3), koma(v[3], 3), koma(v[0])) for d, v in RUN[k]["tabel_lambda"].items()],
                         columns=["Decoding", "chrF++", "Ketepatan tingkat", "Ketepatan Krama Alus", "Skor pemilihan"]))
skor_model = {k: max(v[0] for v in RUN[k]["tabel_lambda"].values()) for k in ["E", "F", "G"]}
PEMENANG = max(skor_model, key=skor_model.get)
display(pd.DataFrame([(NAMA_RUN[k], koma(v)) for k, v in skor_model.items()], columns=["Model", "Skor validasi terbaik"]))
print("Model dengan skor validasi tertinggi:", NAMA_RUN[PEMENANG])
""")

# ====================================================================== 9. evaluasi
md(r"""
## 9. Evaluasi pada data test

### 9.1 Sebelum dan sesudah setiap langkah

chrF++ dan BLEU dihitung pada 176 kalimat test. Ketepatan tingkat dinilai oleh **classifier penilai tetap** (bagian
2.1) pada set kontrol: setiap kalimat Indonesia unik di test diterjemahkan ke ketiga tingkat (225 permintaan).
Kolom terakhir adalah **selisih chrF++ terhadap run E + rerank** dengan **interval kepercayaan 90%** dari paired
bootstrap per grup (1.000 kali; Koehn, 2004). Jika interval itu tidak memuat 0, selisihnya dapat dianggap nyata.
Run A dinilai pada test acaknya sendiri, jadi tidak dibandingkan dengan bootstrap. Baris F dan G membandingkan model
dasar yang berbeda dengan resep yang sama.

Sebagai pembanding ditampilkan juga **model final di paper** (dibekukan, dilatih di kernel `unggah-ungguh-final-3lv-v2`
dengan resep yang sama dengan run E + rerank).
""")
code(r"""
REF = TEST.javanese.values

def nilai(hyp, test_df, kontrol=None):
    hyp = np.array(hyp, dtype=object); ref = test_df.javanese.values
    r = {"chrF++": chrf(hyp, ref), "BLEU": bleu(hyp, ref)}
    for lv in LV:
        m = (test_df.level == lv).values; r[f"chrF++ {lv}"] = chrf(hyp[m], ref[m])
    if kontrol is not None:
        v = vonis_juri(kontrol.output.fillna("").astype(str)); ok = v == kontrol.requested.values
        r["ketepatan"] = ok.mean(); r["ketepatan krama_alus"] = ok[kontrol.requested.values == "krama_alus"].mean()
        r["vonis"] = v
    return r

def bootstrap_selisih(h_a, h_b, B=1000, seed=0):
    rng = np.random.default_rng(seed); g = TEST.group.values; ug = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in ug}; beda = []
    for _ in range(B):
        s = np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])
        beda.append(chrf(h_a[s], REF[s]) - chrf(h_b[s], REF[s]))
    return np.percentile(beda, 5), np.percentile(beda, 95)

EVAL = {}
for k in ["A", "B", "C", "D", "E"]:
    EVAL[k] = nilai(RUN[k]["test"], RUN[k]["test_df"], None if k == "A" else RUN[k]["kontrol"])
for k in ["E", "F", "G"]:
    EVAL[f"{k}_rerank"] = nilai(RUN[k]["test_rerank"], TEST, RUN[k]["kontrol_rerank"])
kp = pd.read_csv(f"{K}/final_3lv_v2/output/control_m2m_base_rerank2.0.csv"); kp = kp[kp.requested.isin(LV)].reset_index(drop=True)
hp = pd.read_csv(f"{K}/final_3lv_v2/output/pred_m2m_base_rerank2.0.csv"); hp = hp[hp.level.isin(LV)].reset_index(drop=True)
assert (hp["index"].values == TEST["index"].values).all()
EVAL["paper"] = nilai(hp.hyp.fillna("").astype(str).values, TEST, kp)
assert CEPAT or abs(EVAL["paper"]["chrF++"] - 58.62) < 0.01

HYP_RUN = {k: np.array(RUN[k]["test"], dtype=object) for k in ["B", "C", "D", "E"]}
for k in ["E", "F", "G"]:
    HYP_RUN[f"{k}_rerank"] = np.array(RUN[k]["test_rerank"], dtype=object)
HYP_RUN["paper"] = hp.hyp.fillna("").astype(str).values
if not CEPAT:
    for k in ["B", "C", "D", "E", "F_rerank", "G_rerank", "paper"]:
        EVAL[k]["interval"] = bootstrap_selisih(HYP_RUN[k], HYP_RUN["E_rerank"])

LABEL = {"A": "A: split acak + tag (test acak, bocor)", "B": "B: split per grup + tag", "C": "C: split per grup, tanpa tag",
         "D": "D: B + inisialisasi tag + oversampling Krama Alus", "E": "E: D + multi-task Krama → Krama Alus",
         "E_rerank": "E + rerank (8 kandidat)", "F_rerank": "F: NLLB-600M, resep E + rerank",
         "G_rerank": "G: Cendol mT5-base, resep E + rerank", "paper": "Model final di paper (dibekukan)"}
baris = []
for k in ["A", "C", "B", "D", "E", "E_rerank", "F_rerank", "G_rerank", "paper"]:
    r = EVAL[k]; sel = r["chrF++"] - EVAL["E_rerank"]["chrF++"]; iv = r.get("interval")
    baris.append((LABEL[k], koma(r["chrF++"]), koma(r["BLEU"]), koma(r.get("ketepatan"), 3), koma(r.get("ketepatan krama_alus"), 3),
                  "acuan" if k == "E_rerank" else ("tidak sebanding (test berbeda)" if k == "A" else
                                                   (f"{koma(sel)} [{koma(iv[0])} sampai {koma(iv[1])}]" if iv else koma(sel)))))
TABEL_RUN = pd.DataFrame(baris, columns=["Run", "chrF++", "BLEU", "Ketepatan tingkat", "Ketepatan Krama Alus",
                                         "Selisih chrF++ terhadap E + rerank [interval kepercayaan 90%]"])
display(TABEL_RUN)
""")
md(r"""
### 9.2 Pengaruh setiap langkah preprocessing

Tabel berikut menghitung perubahan dari satu run ke run berikutnya, jadi setiap baris adalah satu langkah
"sebelum → sesudah".
""")
code(r"""
langkah = [("Split acak → split per grup", "A", "B"), ("Tanpa tag → dengan source tag", "C", "B"),
           ("+ inisialisasi tag + oversampling Krama Alus", "B", "D"), ("+ multi-task Krama → Krama Alus", "D", "E"),
           ("+ rerank saat decoding", "E", "E_rerank"),
           ("Ganti model: M2M100 → NLLB-600M (keduanya + rerank)", "E_rerank", "F_rerank"),
           ("Ganti model: M2M100 → Cendol mT5-base (keduanya + rerank)", "E_rerank", "G_rerank")]
baris = []
for nama, a, b in langkah:
    ra, rb = EVAL[a], EVAL[b]
    d_ket = None if ra.get("ketepatan") is None else rb["ketepatan"] - ra["ketepatan"]
    d_ka = None if ra.get("ketepatan krama_alus") is None else rb["ketepatan krama_alus"] - ra["ketepatan krama_alus"]
    baris.append((nama, f"{koma(ra['chrF++'])} → {koma(rb['chrF++'])}", koma(rb["chrF++"] - ra["chrF++"]),
                  "–" if d_ket is None else koma(d_ket, 3), "–" if d_ka is None else koma(d_ka, 3)))
display(pd.DataFrame(baris, columns=["Langkah", "chrF++ sebelum → sesudah", "Perubahan chrF++", "Perubahan ketepatan tingkat",
                                     "Perubahan ketepatan Krama Alus"]))
fig, axs = plt.subplots(1, 2, figsize=(12, 3.8))
urut_run = ["C", "B", "D", "E", "E_rerank"]; label = ["C\ntanpa tag", "B\n+ tag", "D\n+ inisialisasi\n+ oversampling", "E\n+ multi-task", "E\n+ rerank"]
b = axs[0].bar(label, [EVAL[k]["chrF++"] for k in urut_run], color=["#E45756", "#4C78A8", "#F58518", "#2E8B57", "#1B5E20"])
axs[0].bar_label(b, labels=[koma(EVAL[k]["chrF++"]) for k in urut_run], fontsize=8); axs[0].set_ylabel("chrF++ test")
axs[0].set_title("Kualitas terjemahan per langkah"); axs[0].set_ylim(0, max(EVAL[k]["chrF++"] for k in urut_run) + 8)
x = np.arange(len(urut_run))
axs[1].plot(x, [EVAL[k]["ketepatan"] for k in urut_run], "-o", color="#222222", label="semua tingkat")
axs[1].plot(x, [EVAL[k]["ketepatan krama_alus"] for k in urut_run], "-s", color=WARNA["krama_alus"], label="Krama Alus")
axs[1].set_xticks(x, label); axs[1].set_ylim(0, 1.05); axs[1].set_ylabel("ketepatan tingkat (classifier penilai tetap)")
axs[1].set_title("Ketepatan tingkat per langkah"); axs[1].legend(frameon=False)
plt.tight_layout(); plt.show()
""")
md(r"""
### 9.3 chrF++ per tingkat dan perbandingan model dasar

Lima kelompok pertama adalah langkah preprocessing pada M2M100; dua kelompok terakhir adalah model dasar lain dengan
resep yang sama.
""")
code(r"""
pilih = ["C", "B", "D", "E", "E_rerank", "F_rerank", "G_rerank"]
fig, ax = plt.subplots(figsize=(13, 3.8)); x = np.arange(len(pilih)); w = 0.26
for i, lv in enumerate(LV):
    v = [EVAL[k][f"chrF++ {lv}"] for k in pilih]
    b = ax.bar(x + (i - 1) * w, v, w, color=WARNA[lv], label=NAMA[lv]); ax.bar_label(b, labels=[koma(t, 1) for t in v], fontsize=7)
ax.set_xticks(x, ["C: tanpa tag", "B: + tag", "D: + inisialisasi\n+ oversampling", "E: + multi-task", "E + rerank",
                 "F: NLLB-600M\n+ rerank", "G: Cendol mT5\n+ rerank"])
ax.set_ylabel("chrF++ test"); ax.legend(frameon=False); ax.set_title("chrF++ per tingkat tutur"); plt.tight_layout(); plt.show()
display(pd.DataFrame([(LABEL[k], koma(EVAL[k]["chrF++"]), koma(EVAL[k]["ketepatan"], 3), koma(EVAL[k]["ketepatan krama_alus"], 3))
                      for k in ["E_rerank", "F_rerank", "G_rerank"]],
                     columns=["Model (resep sama)", "chrF++", "Ketepatan tingkat", "Ketepatan Krama Alus"]))
print("Model yang dipilih di validasi (bagian 8):", NAMA_RUN[PEMENANG])
""")
md(r"""
**Cara membaca:** chrF++ dan ketepatan Krama Alus tidak selalu bergerak bersama. Model dengan chrF++ tertinggi belum
tentu paling tepat memilih tingkat, dan sebaliknya. Skor pemilihan di bagian 8 (chrF++ + 30 × ketepatan tingkat)
menggabungkan keduanya, tetapi ketepatan yang dipakai adalah rata-rata semua tingkat, sehingga keunggulan atau
kelemahan khusus pada Krama Alus hanya sedikit memengaruhi pilihan. Jika tujuan utama sistem adalah ketepatan Krama Alus,
pilihan model bisa berbeda; keputusan ini perlu dinyatakan sebelum melihat data test.
""")
md(r"""
### 9.4 Confusion matrix tingkat

Baris adalah tingkat yang **diminta**, kolom adalah tingkat menurut classifier penilai tetap. Angka di luar diagonal
adalah output yang tingkatnya meleset.
""")
code(r"""
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (k, judul) in zip(axs, [("C", "C: tanpa tag"), ("E", "E: beam search 4"), ("E_rerank", "E + rerank")]):
    kontrol = RUN["E"]["kontrol_rerank"] if k == "E_rerank" else RUN[k]["kontrol"]
    cm = pd.crosstab(kontrol.requested, pd.Series(EVAL[k]["vonis"], index=kontrol.index)).reindex(index=LV, columns=LV, fill_value=0).values
    ax.imshow(cm / np.maximum(cm.sum(1, keepdims=True), 1), cmap="Blues", vmin=0, vmax=1)
    for i in range(3):
        for j in range(3):
            p = cm[i, j] / max(cm[i].sum(), 1)
            ax.text(j, i, f"{cm[i, j]}\n({koma(100 * p, 0)}%)", ha="center", va="center", color="white" if p > 0.6 else "black", fontsize=9)
    ax.set_xticks(range(3), [NAMA[l] for l in LV]); ax.set_yticks(range(3), [NAMA[l] for l in LV])
    ax.set_xlabel("tingkat menurut classifier penilai"); ax.set_ylabel("tingkat yang diminta"); ax.set_title(judul)
    print(f"{judul}: Krama Alus keluar sebagai Krama = {cm[2, 1]} dari {cm[2].sum()}")
plt.tight_layout(); plt.show()
""")
md(r"""
**Cara membaca:** tanpa tag (C), model selalu menghasilkan satu versi yang sama untuk ketiga permintaan, sehingga
hampir semua output jatuh di satu kolom. Dengan tag, diagonal terisi. Kesalahan yang tersisa hampir semuanya ada di
satu sel, yaitu **Krama Alus yang keluar sebagai Krama**, sesuai temuan EDA bahwa kedua tingkat ini paling mirip
(bagian 4.5). Rerank menargetkan sel itu.

### 9.5 Seberapa bisa dipercaya classifier penilai?

Classifier penilai diuji pada kalimat **referensi** test (tulisan manusia) yang tingkatnya sudah pasti benar.
""")
code(r"""
v = vonis_juri(TEST.javanese)
tabel = pd.DataFrame([(NAMA[lv], int((TEST.level == lv).sum()), int((v[TEST.level == lv] == lv).sum())) for lv in LV],
                     columns=["Tingkat referensi", "Jumlah kalimat", "Dinilai benar"])
tabel["Akurasi"] = (tabel["Dinilai benar"] / tabel["Jumlah kalimat"]).map(lambda x: koma(x, 3))
display(tabel)
""")
md(r"""
Classifier penilai hampir sempurna pada Ngoko dan Krama, tetapi kadang menilai kalimat Krama Alus asli sebagai Krama.
Artinya, angka ketepatan Krama Alus cenderung **sedikit di bawah** nilai sebenarnya. Classifier juga hanya menilai
tingkat, **bukan makna** (dibahas di bagian 10).

### 9.6 Perbandingan dengan model lain (eksperimen sebelumnya)

Variasi lain dicoba di kernel terpisah: model 4 tingkat, beberapa seed, data tambahan, self-training, dan post-editing. Melatih ulang
semuanya di notebook ini tidak praktis, jadi hasilnya dihitung ulang dari prediksi tersimpan. Ketepatan tingkat
dinilai oleh classifier penilai tetap yang sama.
""")

code(r"""
F3M, F3N, SEEDS = "finetune_3lv_m2m100/output", "finetune_3lv_nllb/output", ["s42", "s13", "s2024"]
MODEL = [  # (kode, kelompok, nama lengkap, file prediksi, kolom hipotesis, file kontrol)
    ("ft_notag", "Fine-tuning 4 tingkat", "M2M100 fine-tuning tanpa tag", ["finetune_m2m100/output/pred_m2m100_418m_no_tag.csv"], "hyp",
     ["finetune_m2m100/output/control_m2m100_418m_no_tag.csv"]),
    ("ft_tag_m2m", "Fine-tuning 4 tingkat", "M2M100 + source tag", ["finetune_m2m100/output/pred_m2m100_418m_atomic.csv"], "hyp",
     ["finetune_m2m100/output/control_m2m100_418m_atomic.csv"]),
    ("ft_tag_nllb", "Fine-tuning 4 tingkat", "NLLB-600M + source tag", ["finetune_nllb/output/pred_nllb_600m_atomic.csv"], "hyp",
     ["finetune_nllb/output/control_nllb_600m_atomic.csv"]),
    ("r3_base", "Fine-tuning 3 tingkat", "M2M100 + source tag (rata-rata 3 seed)", [f"{F3M}/pred3_m2m100_418m_atomic_base_{s}.csv" for s in SEEDS], "hyp",
     [f"{F3M}/control3_m2m100_418m_atomic_base_{s}.csv" for s in SEEDS]),
    ("r3_initos", "Fine-tuning 3 tingkat", "+ inisialisasi tag + oversampling (rata-rata 3 seed)", [f"{F3M}/pred3_m2m100_418m_atomic_init_os_{s}.csv" for s in SEEDS], "hyp",
     [f"{F3M}/control3_m2m100_418m_atomic_init_os_{s}.csv" for s in SEEDS]),
    ("r3_nllb", "Fine-tuning 3 tingkat", "NLLB-600M + inisialisasi + oversampling + multi-task (rata-rata 3 seed)",
     [f"{F3N}/pred3_nllb_600m_atomic_init_os_mt_{s}.csv" for s in SEEDS], "hyp", [f"{F3N}/control3_nllb_600m_atomic_init_os_mt_{s}.csv" for s in SEEDS]),
    ("v2_beam", "Resep final", "M2M100 resep final, beam search 4", ["final_3lv_v2/output/pred_m2m_base_beam4.csv"], "hyp",
     ["final_3lv_v2/output/control_m2m_base_beam4.csv"]),
    ("final", "Resep final", "M2M100 resep final + rerank (model final di paper)", ["final_3lv_v2/output/pred_m2m_base_rerank2.0.csv"], "hyp",
     ["final_3lv_v2/output/control_m2m_base_rerank2.0.csv"]),
    ("v2_na_ls", "Sesudah model final", "+ data Ngoko Alus + label smoothing + rerank", ["final_3lv_v2/output/pred_m2m_na_ls_rerank1.0.csv"], "hyp",
     ["final_3lv_v2/output/control_m2m_na_ls_rerank1.0.csv"]),
    ("v2_nllb", "Sesudah model final", "NLLB-600M resep final + rerank", ["final_3lv_v2/output/pred_nllb_na_rerank2.0.csv"], "hyp",
     ["final_3lv_v2/output/control_nllb_na_rerank2.0.csv"]),
    ("st", "Sesudah model final", "+ self-training data Krama Alus semu + rerank", ["final_3lv_selftrain/output/pred_st_train_rerank0.5.csv"], "hyp",
     ["final_3lv_selftrain/output/control_st_train_rerank0.5.csv"]),
    ("pe", "Sesudah model final", "Model final + post-editing leksikon Krama Alus", ["final_3lv_postedit/output/pred_rerank2.0_pe_trig_gain.csv"], "hyp",
     ["final_3lv_postedit/output/control_rerank2.0_pe_trig_gain.csv"]),
]

def muat_prediksi(path, kolom):
    d = pd.read_csv(f"{K}/{path}"); d = d[d.level.isin(LV)].reset_index(drop=True)
    assert len(d) == 176 and (d["index"].values == TEST["index"].values).all(), path
    return d[kolom].fillna("").astype(str).values

def muat_kontrol(path):
    d = pd.read_csv(f"{K}/{path}"); d = d[d.requested.isin(LV)].reset_index(drop=True)
    assert len(d) == 225 and not d.duplicated(["indonesian", "requested"]).any(), path
    d["vonis"] = vonis_juri(d.output.fillna("").astype(str)); return d

REF = TEST.javanese.values
HASIL, HYP, KONTROL = {}, {}, {}
for kode, kel, nama, preds, kolom, ctrl in MODEL:
    hs = [muat_prediksi(p, kolom) for p in preds]; HYP[kode] = hs[0]
    r = {"kelompok": kel, "nama": nama, "chrF++": np.mean([chrf(h, REF) for h in hs]), "BLEU": np.mean([bleu(h, REF) for h in hs])}
    for lv in LV:
        m = (TEST.level == lv).values; r[f"chrF++ {lv}"] = np.mean([chrf(h[m], REF[m]) for h in hs])
    if ctrl:
        ks = [muat_kontrol(c) for c in ctrl]; KONTROL[kode] = ks[0]
        r["ketepatan"] = np.mean([(k.vonis == k.requested).mean() for k in ks])
        r["ketepatan krama_alus"] = np.mean([(k.vonis == k.requested)[k.requested == "krama_alus"].mean() for k in ks])
    HASIL[kode] = r
assert abs(HASIL["final"]["chrF++"] - 58.62) < 0.01 and abs(HASIL["final"]["BLEU"] - 33.17) < 0.01
print("model final:", koma(HASIL["final"]["chrF++"]), "chrF++,", koma(HASIL["final"]["BLEU"]), "BLEU,",
      "ketepatan tingkat", koma(HASIL["final"]["ketepatan"], 3))
""")
code(r"""
# model yang hanya punya angka agregat (prediksi per kalimat tidak tersimpan)
for kode, nama, f, dec, ctrl in [("aug", "+ Krama Alus sintetis dari leksikon + rerank", "metrics_aug.json", "rerank4.0", "control_aug_rerank4.0.csv"),
                                ("nusax", "+ data NusaX (Ngoko) + rerank", "metrics_nusax.json", "rerank2.0", "control_nusax_rerank2.0.csv"),
                                ("aug_nusax", "+ Krama Alus sintetis + NusaX + rerank", "metrics_aug_nusax.json", "rerank0.5", "control_aug_nusax_rerank0.5.csv")]:
    t = json.load(open(f"{K}/final_3lv_merge/output/{f}", encoding="utf-8"))["test"][dec]["test"]
    k = muat_kontrol(f"final_3lv_merge/output/{ctrl}")
    HASIL[kode] = {"kelompok": "Sesudah model final", "nama": nama, "chrF++": t["all"]["chrF++"], "BLEU": t["all"]["BLEU"],
                   **{f"chrF++ {lv}": t[lv]["chrF++"] for lv in LV}, "ketepatan": (k.vonis == k.requested).mean(),
                   "ketepatan krama_alus": (k.vonis == k.requested)[k.requested == "krama_alus"].mean()}
log_cendol = open(f"{K}/final_3lv_cendol_tune/output/kernel_log_v1.json", encoding="utf-8").read().replace("\\n", "\n")
for kode, nama, varian, dec in [("cendol_A", "Cendol mT5-base, nama tingkat di prompt", "A_lvprompt", "beam4"),
                                ("cendol_B", "Cendol mT5-base + oversampling + label smoothing + rerank", "B_os5_ls", "rerank8.0")]:
    m = re.search(rf"{varian} {re.escape(dec)} TEST (\{{[^}}]*\}}) ctrl ([0-9.]+) (\{{[^}}]*\}})", log_cendol)
    t, c = json.loads(m.group(1).replace("'", '"')), json.loads(m.group(3).replace("'", '"'))
    HASIL[kode] = {"kelompok": "Model lain (classifier bawaan kernel)", "nama": nama, "chrF++": t["chrF++"], "BLEU": t["BLEU"],
                   "ketepatan": float(m.group(2)), "ketepatan krama_alus": c["krama_alus"]}

def bootstrap_selisih(h_a, h_b, B=1000, seed=0):
    rng = np.random.default_rng(seed); g = TEST.group.values; ug = np.unique(g)
    idx = {x: np.where(g == x)[0] for x in ug}; beda = []
    for _ in range(B):
        s = np.concatenate([idx[x] for x in rng.choice(ug, len(ug))])
        beda.append(chrf(h_a[s], REF[s]) - chrf(h_b[s], REF[s]))
    return np.percentile(beda, 5), np.percentile(beda, 95)

for kode in ([] if CEPAT else HYP):
    if kode != "final":
        HASIL[kode]["interval"] = bootstrap_selisih(HYP[kode], HYP["final"])

URUT = ["ft_notag", "ft_tag_m2m", "ft_tag_nllb", "r3_base", "r3_initos", "r3_nllb", "v2_beam", "final",
        "v2_na_ls", "v2_nllb", "nusax", "aug", "aug_nusax", "st", "pe", "cendol_A", "cendol_B"]
baris = []
for i, kode in enumerate(URUT, 1):
    r = HASIL[kode]; sel = r["chrF++"] - HASIL["final"]["chrF++"]
    iv = r.get("interval")
    baris.append((i, r["kelompok"], r["nama"], koma(r["chrF++"]), koma(r["BLEU"]),
                  koma(r.get("ketepatan"), 3), koma(r.get("ketepatan krama_alus"), 3),
                  "acuan" if kode == "final" else (f"{koma(sel)} [{koma(iv[0])} sampai {koma(iv[1])}]" if iv else f"{koma(sel)} (tanpa prediksi per kalimat)")))
TABEL_UTAMA = pd.DataFrame(baris, columns=["No.", "Kelompok", "Model", "chrF++", "BLEU", "Ketepatan tingkat", "Ketepatan Krama Alus",
                                           "Selisih chrF++ terhadap model final [interval kepercayaan 90%]"]).set_index("No.")
display(TABEL_UTAMA)
""")
md(r"""
Baris 1–3 memakai 4 tingkat (termasuk Ngoko Alus), sehingga tidak setara penuh dengan baris 4 ke bawah. Baris Cendol
dinilai oleh classifier bawaan kernel karena prediksi per kalimatnya tidak tersimpan. Grafik berikut menempatkan setiap
model menurut kualitas terjemahan dan ketepatan Krama Alus sekaligus; kanan atas adalah posisi ideal.
""")
code(r"""
fig, ax = plt.subplots(figsize=(8, 4.6))
kel_warna = {"Fine-tuning 4 tingkat": "#9DB4CC", "Fine-tuning 3 tingkat": "#4C78A8", "Resep final": "#C0392B",
             "Sesudah model final": "#2E8B57", "Model lain (classifier bawaan kernel)": "#F58518"}
for i, kode in enumerate(URUT, 1):
    r = HASIL[kode]
    if r.get("ketepatan krama_alus") is None or kode == "pe": continue
    ax.scatter(r["chrF++"], r["ketepatan krama_alus"], s=90 if kode == "final" else 40, color=kel_warna[r["kelompok"]],
               marker="D" if "Model lain" in r["kelompok"] else "o", edgecolor="white", zorder=3)
    ax.annotate(f"{i}" + (" (final)" if kode == "final" else ""), (r["chrF++"], r["ketepatan krama_alus"]), xytext=(5, 3),
                textcoords="offset points", fontsize=8)
for k, c in kel_warna.items(): ax.scatter([], [], color=c, label=k)
ax.set_xlabel("chrF++ test (176 kalimat)"); ax.set_ylabel("ketepatan Krama Alus (75 permintaan)")
ax.legend(frameon=False, fontsize=8, loc="lower right"); ax.set_title("Nomor = nomor baris di tabel 9.1")
plt.tight_layout(); plt.show()
""")

# ====================================================================== 10. analisis kesalahan
md(r"""
## 10. Analisis kesalahan

### 10.1 F1-score kata menurut seberapa sering kata itu muncul di data train

Setiap kata di kalimat referensi dan output dikelompokkan menurut frekuensinya di data train: tidak pernah muncul,
muncul 1–4 kali, atau muncul 5 kali atau lebih. Untuk setiap kelompok dihitung F1-score kecocokan kata antara output
dan referensi (cara analisis seperti alat compare-mt, Neubig dkk., 2019).
""")
code(r"""
def kelompok_frek(w):
    f = kosakata_train[w]; return "tidak pernah muncul" if f == 0 else ("1–4 kali" if f < 5 else "5 kali atau lebih")
KELOMPOK = ["tidak pernah muncul", "1–4 kali", "5 kali atau lebih"]
baris, dukungan = [], Counter()
for kode, nama in [("B", "B: + tag"), ("E", "E: beam search 4"), ("E_rerank", "E + rerank"),
                   ("F_rerank", "F: NLLB-600M + rerank"), ("G_rerank", "G: Cendol mT5 + rerank")]:
    cocok, nr, nh = Counter(), Counter(), Counter()
    for ref, hyp in zip(REF, HYP_RUN[kode]):
        cr, ch = Counter(token(ref)), Counter(token(hyp))
        for w, c in (cr & ch).items(): cocok[kelompok_frek(w)] += c
        for w, c in cr.items(): nr[kelompok_frek(w)] += c
        for w, c in ch.items(): nh[kelompok_frek(w)] += c
    dukungan = nr
    baris.append((nama, *[100 * 2 * cocok[g] / max(nr[g] + nh[g], 1) for g in KELOMPOK]))
F1 = pd.DataFrame(baris, columns=["Model"] + KELOMPOK).set_index("Model")
fig, ax = plt.subplots(figsize=(7.5, 3.4))
for (nama, r), mk in zip(F1.iterrows(), ["o", "s", "^", "D", "v"]):
    ax.plot(range(3), r.values, marker=mk, label=nama)
ax.set_xticks(range(3), [f"{g}\n({dukungan[g]} token referensi)" for g in KELOMPOK])
ax.set_ylabel("F1-score kata (%)"); ax.set_xlabel("frekuensi kata di data train"); ax.legend(frameon=False)
ax.set_title("Kata yang jarang atau tidak pernah dilihat paling sering salah"); plt.tight_layout(); plt.show()
display(F1.round(1))
""")
md(r"""
**Cara membaca:** kata yang sering muncul di train jauh lebih sering diterjemahkan dengan benar daripada kata yang
tidak pernah dilihat model. Pola ini sejalan dengan learning curve per ukuran data (bagian 7.2): yang membatasi model
adalah **kosakata**, bukan kontrol tingkat. Analisis ini bersifat deskriptif dan tidak membuktikan sebab-akibat.

### 10.2 Contoh: satu kalimat, tiga tingkat
""")
code(r"""
k = RUN["E"]["kontrol_rerank"].assign(vonis=EVAL["E_rerank"]["vonis"])
kalimat = "Jika sudah waktu malam, jangan pergi sendirian!"
pilih = k[k.indonesian == kalimat] if (k.indonesian == kalimat).any() else k[k.indonesian == k.indonesian.iloc[0]]
print("Input Indonesia:", pilih.indonesian.iloc[0])
lama = kp[kp.indonesian == pilih.indonesian.iloc[0]].set_index("requested").output
display(pd.DataFrame({"Tingkat diminta": pilih.requested.map(NAMA).values, "Output run E + rerank": pilih.output.values,
                      "Menurut classifier penilai": pd.Series(pilih.vonis.values).map(NAMA).values,
                      "Output model final di paper": [lama.get(l, "–") for l in pilih.requested]}))
""")
md(r"""
Pada model final di paper, output Krama Alus untuk kalimat ini adalah *"Yen sampun wancinipun dalu, sampun tindak sare!"*,
yang berarti *"…jangan pergi tidur!"*, padahal input meminta *"jangan pergi sendirian!"*. Classifier menilainya benar
karena kata *tindak* (pergi) dan *sare* (tidur) sama-sama khas Krama Alus. Kesalahan makna seperti ini **tidak
tertangkap** oleh classifier tingkat. Bandingkan dengan output run E di tabel di atas.

### 10.3 Contoh output Krama Alus di data test beserta referensinya
""")
code(r"""
ka = TEST.level.values == "krama_alus"
skor = [sacrebleu.sentence_chrf(h, [r], word_order=2).score for h, r in zip(HYP_RUN["E_rerank"][ka], REF[ka])]
d = pd.DataFrame({"Indonesia": TEST.indonesian.values[ka], "Referensi Krama Alus": REF[ka], "Output E + rerank": HYP_RUN["E_rerank"][ka],
                  "chrF++ kalimat": skor}).sort_values("chrF++ kalimat")
pd.set_option("display.max_colwidth", 80)
display(pd.concat([d.head(3), d.tail(2)]).assign(**{"chrF++ kalimat": lambda x: x["chrF++ kalimat"].map(lambda v: koma(v, 1))}))
""")
md(r"""
### 10.4 Kategori kesalahan otomatis

Setiap output test dicek dengan aturan sederhana yang bisa dihitung otomatis. Satu kalimat bisa masuk lebih dari satu
kategori.

| Kategori | Aturan |
|---|---|
| Tingkat meleset | classifier penilai tetap menilai output berada di tingkat lain dari yang diminta |
| Kata Indonesia tidak diterjemahkan | ada kata di output yang juga ada di kalimat Indonesia, tidak ada di referensi, dan tidak pernah muncul di teks Jawa data train |
| Terlalu pendek | jumlah kata output kurang dari 60% referensi (kemungkinan ada bagian kalimat yang hilang) |
| Terlalu panjang | jumlah kata output lebih dari 160% referensi (kemungkinan ada tambahan yang tidak diminta) |
| Kata berulang | kata yang sama muncul tiga kali berturut-turut |
| Pilihan kata atau makna berbeda | tidak masuk kategori lain, tetapi chrF++ kalimatnya di bawah 40 |

Aturan ini bersifat **indikatif**: kategori terakhir menangkap kalimat yang jauh dari referensi, yang bisa berarti
makna salah atau hanya terjemahan lain yang sama-sama benar. Kategori "kata Indonesia tidak diterjemahkan" sebagian
besar menangkap kata yang memang disalin dari input (misalnya *ikan* yang seharusnya *iwak*, *jembatan* yang
seharusnya *kreteg*), tetapi juga beberapa kata serapan yang sah dalam bahasa Jawa (misalnya *batik*, *cocok*),
sehingga jumlahnya sedikit berlebih. Pemeriksaan yang pasti membutuhkan penutur asli.
""")
code(r"""
KATEGORI = ["tingkat meleset", "kata Indonesia tidak diterjemahkan", "terlalu pendek", "terlalu panjang", "kata berulang",
            "pilihan kata atau makna berbeda"]
kosakata_jawa_train = set(kosakata_train)

def kategori_kesalahan(hyp_arr):
    vonis = vonis_juri(hyp_arr); hasil = []
    for i, (src, ref, hyp, lv) in enumerate(zip(TEST.indonesian, REF, hyp_arr, TEST.level)):
        t_ref, t_hyp, t_src = token(ref), token(hyp), set(token(src))
        kat = []
        if vonis[i] != lv: kat.append("tingkat meleset")
        if [w for w in t_hyp if w in t_src and w not in set(t_ref) and w not in kosakata_jawa_train]:
            kat.append("kata Indonesia tidak diterjemahkan")
        rasio = len(t_hyp) / max(len(t_ref), 1)
        if rasio < 0.6: kat.append("terlalu pendek")
        if rasio > 1.6: kat.append("terlalu panjang")
        if any(t_hyp[j] == t_hyp[j + 1] == t_hyp[j + 2] for j in range(len(t_hyp) - 2)): kat.append("kata berulang")
        skor = sacrebleu.sentence_chrf(hyp, [ref], word_order=2).score
        if not kat and skor < 40: kat.append("pilihan kata atau makna berbeda")
        hasil.append({"indonesia": src, "tingkat": lv, "referensi": ref, "output": hyp, "chrF++ kalimat": skor,
                      "vonis": vonis[i], "kategori": kat})
    return pd.DataFrame(hasil)

KESALAHAN = {k: kategori_kesalahan(HYP_RUN[k]) for k in ["B", "E", "E_rerank", "F_rerank", "G_rerank", "paper"]}
LABEL_K = {"B": "B: + tag", "E": "E: + multi-task", "E_rerank": "E + rerank", "F_rerank": "F: NLLB + rerank",
           "G_rerank": "G: Cendol + rerank", "paper": "model final di paper"}
hitung = pd.DataFrame({LABEL_K[k]: [int(d.kategori.map(lambda x: c in x).sum()) for c in KATEGORI] for k, d in KESALAHAN.items()},
                      index=KATEGORI)
hitung.loc["kalimat tanpa masalah terdeteksi"] = [int((d.kategori.map(len) == 0).sum()) for d in KESALAHAN.values()]
display(hitung.rename_axis("Kategori (jumlah dari 176 kalimat test)"))
fig, ax = plt.subplots(figsize=(12, 3.8)); x = np.arange(len(KATEGORI)); w = 0.13
for i, (k, c) in enumerate(zip(KESALAHAN, ["#4C78A8", "#2E8B57", "#1B5E20", "#8E44AD", "#8C564B", "#999999"])):
    ax.bar(x + (i - 2.5) * w, hitung.loc[KATEGORI, LABEL_K[k]], w, color=c, label=LABEL_K[k])
ax.set_xticks(x, [c.replace(" ", "\n", 1) for c in KATEGORI], fontsize=8); ax.set_ylabel("jumlah kalimat")
ax.set_title("Kategori kesalahan pada 176 kalimat test (sebelum dan sesudah)"); ax.legend(frameon=False, fontsize=8)
plt.tight_layout(); plt.show()
""")
md(r"""
Contoh untuk setiap kategori dari run E + rerank:
""")
code(r"""
d = KESALAHAN["E_rerank"]; contoh = []
for c in KATEGORI:
    sub = d[d.kategori.map(lambda x: c in x)].sort_values("chrF++ kalimat")
    if len(sub):
        r = sub.iloc[0]
        contoh.append((c, NAMA[r.tingkat], r.indonesia, r.referensi, r.output,
                       NAMA[r.vonis] if c == "tingkat meleset" else "", koma(r["chrF++ kalimat"], 1)))
display(pd.DataFrame(contoh, columns=["Kategori", "Tingkat diminta", "Indonesia", "Referensi", "Output E + rerank",
                                      "Menurut classifier penilai", "chrF++ kalimat"]))
""")
md(r"""
**Cara membaca:** bandingkan kolom run B, E, dan E + rerank. Kategori yang turun dari kiri ke kanan adalah kesalahan
yang berhasil dikurangi oleh langkah preprocessing dan rerank (terutama tingkat meleset). Kategori yang tetap tinggi,
terutama pilihan kata atau makna, adalah masalah yang belum terselesaikan oleh perlakuan data mana pun dan sejalan
dengan temuan kosakata di bagian 10.1.

### 10.5 Kesalahan menurut panjang kalimat dan tingkat tutur
""")
code(r"""
d = KESALAHAN["E_rerank"].copy()
d["panjang"] = pd.cut(TEST.indonesian.str.split().str.len().values, [0, 5, 9, 100], labels=["1–5 kata", "6–9 kata", "10 kata atau lebih"])
rata = d.pivot_table(index="tingkat", columns="panjang", values="chrF++ kalimat", aggfunc="mean", observed=False).reindex(LV)
jumlah = d.pivot_table(index="tingkat", columns="panjang", values="chrF++ kalimat", aggfunc="count", observed=False).reindex(LV)
fig, ax = plt.subplots(figsize=(7, 3.4))
im = ax.imshow(rata.values, cmap="RdYlGn", vmin=20, vmax=80)
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{koma(rata.values[i, j], 1)}\n({int(jumlah.values[i, j])} kalimat)", ha="center", va="center", fontsize=9)
ax.set_xticks(range(3), rata.columns); ax.set_yticks(range(3), [NAMA[l] for l in LV])
ax.set_xlabel("panjang kalimat Indonesia"); plt.colorbar(im, label="rata-rata chrF++ kalimat")
ax.set_title("Run E + rerank: rata-rata chrF++ per kalimat"); plt.tight_layout(); plt.show()
""")
md(r"""
**Cara membaca:** sel dengan sedikit kalimat (misalnya Krama Alus panjang) tidak stabil, jadi hanya pola besar yang
bisa disimpulkan. Kalimat yang lebih panjang cenderung mendapat skor lebih rendah, karena makin banyak kata yang
harus benar dan makin besar peluang ada kata yang belum pernah dilihat model.

### 10.6 Audit makna 27 output Krama Alus (model final di paper)

Ke-27 output Krama Alus model final di paper pada data test diperiksa satu per satu: apakah maknanya sama dengan
kalimat Indonesia, dan apakah tingkatnya benar. Audit ini dilakukan oleh tim proyek dengan bantuan asisten AI
(Claude dan Codex) dan **belum diperiksa penutur asli bahasa Jawa**, sehingga hasilnya bersifat indikatif. Rincian ada
di `kaggle/pool_makna_v1/README.md` di repositori.
""")
code(r"""
audit = pd.DataFrame([("Makna salah", 18), ("Makna salah dan arah hormat salah", 5), ("Hanya tingkat yang salah", 1), ("Benar", 3)],
                     columns=["Hasil audit", "Jumlah output"])
fig, ax = plt.subplots(figsize=(8, 1.6)); kiri = 0
for (nama, n), c in zip(audit.values, ["#C0392B", "#E67E22", WARNA["krama_alus"], "#2E8B57"]):
    ax.barh([0], [n], left=kiri, color=c, label=f"{nama} ({n})"); ax.text(kiri + n / 2, 0, str(n), ha="center", va="center", color="white")
    kiri += n
ax.axis("off"); ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.05))
ax.set_title("Audit makna 27 output Krama Alus (data test)"); plt.tight_layout(); plt.show()
""")
md(r"""
**Hasil:** dari 27 output Krama Alus, hanya 1 yang salah karena tingkatnya saja. Sebanyak 23 output salah **makna**.
Jadi masalah utama model bukan lagi memilih tingkat, melainkan menerjemahkan isi kalimat dengan benar.
""")

# ====================================================================== 11. eksperimen gagal
md(r"""
## 11. Eksperimen yang tidak berhasil

Hasil negatif tetap dilaporkan supaya tidak diulang dan supaya kesimpulan tidak bias ke hasil yang bagus saja.

| Eksperimen | Ide | Hasil | Pelajaran |
|---|---|---|---|
| Krama Alus sintetis dari leksikon | ganti kata Krama dengan padanan Krama Alus dari kamus untuk membuat data baru | ketepatan Krama Alus turun | data sintetis berbasis kamus menghasilkan kalimat tidak alami |
| Post-editing leksikon | ganti kata Krama di output dengan kata Krama Alus | hampir tidak aktif (0–3 output berubah) | output yang meleset tidak memuat kata yang ada di kamus |
| Self-training | latih ulang dengan output Krama Alus model sendiri yang lolos filter classifier | berbeda hanya 1–4 dari 75 kalimat | filter tingkat meloloskan kalimat yang maknanya rusak |
| Rerank dengan skor makna (terjemahan balik Jawa → Indonesia) | pilih kandidat yang paling bisa diterjemahkan balik ke input | chrF++ Krama Alus validasi turun | kandidat benar jarang ada di 8–16 kandidat; perbaikan harus dari training |
| NLLB-1,3B dan konverter Krama → Krama Alus | model lebih besar; model khusus pengubah tingkat | gagal di uji awal; konverter merusak makna di 13 dari 26 kalimat | metrik tingkat bisa naik sambil makna turun |
| Mengulang resep final dengan 3 seed | mengukur kestabilan | chrF++ validasi 62,17–62,65 | variasi terbesar berasal dari pemilihan checkpoint |
""")

# ====================================================================== 12. kesimpulan
md(r"""
## 12. Kesimpulan, keterbatasan, dan saran

### 12.1 Jawaban pertanyaan penelitian

Angka di bawah diambil langsung dari run yang dilatih di notebook ini (bagian 9), jadi selalu sesuai dengan hasil
run terakhir.
""")
code(r"""
E, Er, B, C, D = EVAL["E"], EVAL["E_rerank"], EVAL["B"], EVAL["C"], EVAL["D"]
display(Markdown(f'''
1. **Pengaruh source tag.** Dari run C (tanpa tag) ke run B (dengan tag), chrF++ berubah {koma(C["chrF++"])} → {koma(B["chrF++"])}
   dan ketepatan tingkat {koma(C["ketepatan"], 3)} → {koma(B["ketepatan"], 3)}. Tanpa tag, ketepatan berada di sekitar
   sepertiga karena model tidak tahu tingkat yang diminta.
2. **Perlakuan untuk Krama Alus.** Ketepatan Krama Alus: run B {koma(B["ketepatan krama_alus"], 3)} → run D (inisialisasi +
   oversampling) {koma(D["ketepatan krama_alus"], 3)} → run E (multi-task) {koma(E["ketepatan krama_alus"], 3)} → E + rerank
   {koma(Er["ketepatan krama_alus"], 3)}. chrF++ akhir E + rerank {koma(Er["chrF++"])}, dibanding model final di paper
   {koma(EVAL["paper"]["chrF++"])}.
3. **Model dasar.** Dengan resep yang sama, chrF++ / ketepatan Krama Alus: M2M100 {koma(Er["chrF++"])} /
   {koma(Er["ketepatan krama_alus"], 3)}, NLLB-600M {koma(EVAL["F_rerank"]["chrF++"])} / {koma(EVAL["F_rerank"]["ketepatan krama_alus"], 3)},
   Cendol mT5 {koma(EVAL["G_rerank"]["chrF++"])} / {koma(EVAL["G_rerank"]["ketepatan krama_alus"], 3)}. Model dengan skor validasi
   tertinggi: {NAMA_RUN[PEMENANG]}.
4. **Kebocoran data.** Run A (split acak) mendapat chrF++ {koma(EVAL["A"]["chrF++"])} pada test acaknya, dibanding
   {koma(B["chrF++"])} untuk resep yang sama pada split per grup. Selisih ini bukan kemampuan model, melainkan kalimat
   test yang sudah terlihat saat training.
5. **Pembatas kualitas.** Kontrol tingkat dipelajari dalam beberapa epoch pertama (bagian 7.1) dan sudah tinggi dengan
   25% data (bagian 7.2). Yang membatasi adalah kosakata dan makna: chrF++ masih naik ketika data ditambah, bahkan
   ketika jumlah langkah training disamakan (bagian 7.3), kata yang tidak pernah dilihat jauh lebih sering salah
   (bagian 10.1), dan 23 dari 27 output Krama Alus model final di paper salah makna (bagian 10.6).
'''))
""")
md(r"""
### 12.2 Keterbatasan

- Data kecil (sekitar 1.700 kalimat pendek); kalimat panjang, teknis, dan dialek belum diuji.
- Model final dan semua run di notebook ini hanya dari 1 seed (seed 42).
- Output **belum diperiksa penutur asli bahasa Jawa**; audit makna masih indikatif.
- Ketepatan tingkat dinilai classifier yang kadang keliru pada Krama Alus, dan classifier tidak menilai makna.
- Lisensi data non-komersial: model hanya untuk riset dan kuliah.

### 12.3 Saran

- Menambah data paralel Indonesia–Krama Alus yang ditulis atau diperiksa penutur asli (dengan izin), karena learning
  curve belum jenuh.
- Evaluasi manusia oleh penutur Jawa, terutama untuk makna.
- Metrik atau filter yang menilai makna, bukan hanya tingkat, sebelum mengklaim perbaikan Krama Alus.
""")

# ====================================================================== 13. referensi
try:
    import references as R
    kunci = ["bleu", "chrf", "fan2021", "farhansyah", "nllb2024", "prayoga2026", "sennrich", "cendol", "nusax", "wongso2021"]
    def fmt(k):
        return "".join(f"*{t}*" if it else t for t, it in R.entry(k))
    ref_list = [fmt(k) for k in kunci]
except Exception as e:  # pragma: no cover
    raise SystemExit(f"gagal memuat referensi paper: {e}")
ref_list += [
    "Koehn, P. (2004). Statistical significance tests for machine translation evaluation. In *Proceedings of the 2004 "
    "Conference on Empirical Methods in Natural Language Processing* (pp. 388–395). https://aclanthology.org/W04-3250",
    "Monroe, B. L., Colaresi, M. P., & Quinn, K. M. (2008). Fightin' words: Lexical feature selection and evaluation for "
    "identifying the content of political conflict. *Political Analysis, 16*(4), 372–403. https://doi.org/10.1093/pan/mpn018",
    "Neubig, G., Dou, Z.-Y., Hu, J., Michel, P., Pruthi, D., & Wang, X. (2019). compare-mt: A tool for holistic comparison "
    "of language generation systems. In *Proceedings of the 2019 Conference of the North American Chapter of the Association "
    "for Computational Linguistics (Demonstrations)* (pp. 35–41). https://doi.org/10.18653/v1/N19-4007",
    "Sennrich, R., & Zhang, B. (2019). Revisiting low-resource neural machine translation: A case study. In *Proceedings of "
    "the 57th Annual Meeting of the Association for Computational Linguistics* (pp. 211–221). https://doi.org/10.18653/v1/P19-1021",
]
ref_list = sorted(r.replace("�", "–") for r in ref_list)
md("## 13. Referensi\n\n" + "\n".join(f"- {r}" for r in ref_list))


# ====================================================================== tulis
def main():
    nb = {"cells": [], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                    "language_info": {"name": "python"}}, "nbformat": 4, "nbformat_minor": 5}
    for i, (t, s) in enumerate(CELLS):
        c = {"cell_type": t, "id": f"sel{i:02d}", "metadata": {}, "source": s.splitlines(keepends=True)}
        if t == "code":
            c.update({"execution_count": None, "outputs": []})
        nb["cells"].append(c)
    out = os.path.join(HERE, "notebook_laporan.ipynb")
    json.dump(nb, open(out, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("tulis", out, "|", len(CELLS), "sel")


if __name__ == "__main__":
    main()
