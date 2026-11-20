---
title: MT Indonesia Jawa Tingkat Tutur
emoji: 🗣️
colorFrom: green
colorTo: yellow
sdk: gradio
sdk_version: 6.30.0
app_file: app.py
pinned: false
license: cc-by-nc-4.0
short_description: Indonesia ke Jawa dengan pilihan Ngoko/Krama/Alus
---

# Demo: MT Indonesia → Jawa dengan Kontrol Tingkat Tutur

Demo project UAS NLP. Pengguna mengetik kalimat bahasa Indonesia, memilih tingkat tutur
(Ngoko, Krama, Krama Alus), lalu model menerjemahkan ke bahasa Jawa sesuai tingkat itu.
Tombol **Bandingkan 3 tingkat** menampilkan ketiga terjemahan berdampingan.
Daftar tingkat dibaca dari `model/demo_config.json`, jadi `app.py` juga tetap bisa memuat model 4 tingkat lama.

## Model di demo (3 tingkat, v2)
- **M2M100-418M**, full fine-tuning pada dataset
  [Unggah-Ungguh](https://huggingface.co/datasets/JavaneseHonorifics/Unggah-Ungguh) (CC BY-NC 4.0),
  hanya baris Ngoko, Krama, Krama Alus (train 1.395 / valid 175 / test 176).
- Kontrol tingkat lewat source prefix tag (`<ngoko>`, `<krama>`, `<krama_alus>`).
- Teknik: embedding tag diinisialisasi dari kata penanda, oversampling Krama Alus ×3, multi-task JV→JV
  Krama → Krama Alus, 15 epoch; checkpoint dipilih dengan skor valid = chrF++ + 30 × ketepatan tingkat.
- **Rerank saat decoding**: 8 kandidat beam, dipilih dengan log-prob model + 2 × log P(tingkat diminta)
  dari classifier `w11wo/javanese-roberta-small` (folder `model/reranker`). Classifier ini sengaja berbeda
  dari classifier penilai (`w11wo/javanese-distilbert-small`) supaya ketepatan tingkat tidak naik palsu.
- Weights dilatih di kernel Kaggle `mdevian/unggah-ungguh-final-3lv-v2` (seed 42, fp16); varian dan λ dipilih di valid.
- Hasil test (1 seed): chrF++ 58,62 · BLEU 33,17 · ketepatan tingkat 0,92 (Ngoko 1,00 · Krama 0,97 · Krama Alus 0,77).
  Tanpa rerank, model yang sama: ketepatan 0,86 (Krama Alus 0,61) dengan chrF++ 58,13.

Model demo sebelumnya (4 tingkat, `mdevian/unggah-ungguh-final-m2m100`) dan hasil multi-seed-nya ada di bawah.

## Hasil evaluasi (test, mean ± std 3 seed)

**4 tingkat**

| Model | Konfigurasi | chrF++ | BLEU | Ketepatan tingkat | Ngoko Alus | Krama Alus |
|---|---|---|---|---|---|---|
| NLLB-600M | zero-shot | 27,84 | 7,73 | - | - | - |
| NLLB-600M | +init+os+mt | **58,68 ± 0,22** | **32,28 ± 0,25** | 0,77 | 0,60 | **0,64** |
| M2M100-418M (demo) | +init+os | 58,14 ± 0,11 | 31,67 ± 0,34 | **0,79** | **0,63** | 0,58 |

**3 tingkat** (Ngoko, Krama, Krama Alus; setup seperti JAIC 2026; 176 kalimat test yang sama)

| Model | Dilatih pada | chrF++ | BLEU |
|---|---|---|---|
| NLLB-600M +init+os+mt | 4 tingkat | **58,67 ± 0,25** | **32,28 ± 0,30** |
| NLLB-600M +init+os+mt | 3 tingkat | 56,92 ± 0,78 | 30,36 ± 1,11 |
| M2M100-418M +init+os | 4 tingkat | 57,98 ± 0,23 | 31,63 ± 0,43 |
| M2M100-418M +init+os | 3 tingkat | 57,82 ± 0,44 | 31,22 ± 0,69 |

Melatih dengan 4 tingkat sama atau lebih baik daripada melatih khusus 3 tingkat, karena Ngoko Alus
berbagi kosakata hormat dengan Krama Alus.

Ketepatan tingkat diukur dengan classifier DistilBERT yang dilatih ulang di train split
(akurasi pada referensi alus 0,70-0,81, sehingga angka alus cenderung under-estimate).

## Eksperimen lanjutan setelah model demo (3 tingkat, 1 seed)
Dua arah yang dicoba sesudah model demo v2. Model demo **belum diganti**: hasil di bawah hanya 1 seed dan
selisih chrF++-nya masih di dalam CI95 (sekitar ±2,5).

**Data latih tambahan untuk M2M100** (`kaggle/final_3lv_merge/`). Resep sama dengan model demo; antar varian
hanya data latih yang berbeda. Angka test dengan decoding pemenang valid tiap varian.

| Varian | Data train | chrF++ | BLEU | Ketepatan tingkat | Krama Alus |
|---|---|---|---|---|---|
| `base` (= model demo) | 1.815 | 58,62 | 33,17 | **0,94** | **0,85** |
| `aug` (+213 Krama Alus sintetis dari leksikon) | 2.028 | 58,82 | 32,90 | 0,91 | 0,79 |
| `nusax` (+NusaX-MT, tag `<ngoko>`) | 2.235 | 59,74 | 33,42 | 0,93 | 0,84 |
| `aug_nusax` | 2.448 | **60,44** | **34,42** | 0,92 | 0,81 |

- NusaX menaikkan chrF++ terutama di Ngoko (54 → 56), sesuai isi NusaX yang berbentuk Ngoko.
- Krama Alus sintetis menurunkan ketepatan Krama Alus: substitusi kata hormat tanpa mengubah struktur kalimat
  membuat model lebih sering menghasilkan Krama biasa.
- Leksikon dan data gabungan dibuat dengan `scripts/build_lexicon_augment.ipynb`. NusaX (CC BY-SA) dan file
  gabungan `data/merged/` tidak disimpan di repo.

**Cendol mT5-base** (`indonlp/cendol-mt5-base-inst`, 580M, Apache-2.0) sebagai pengganti M2M100.
Tag tingkat ditambah instruksi teks: `<tag> Terjemahkan dari bahasa Indonesia ke bahasa Jawa: ...`; fp32.

| Kernel | Varian | chrF++ | BLEU | Ketepatan tingkat | Krama Alus |
|---|---|---|---|---|---|
| `final_3lv_cendol` | lr 5e-5 (resep v2) | 56,75 | 30,19 | 0,80 | 0,49 |
| | lr 3e-4 | 61,64 | 35,91 | 0,85 | 0,60 |
| `final_3lv_cendol_tune` | `ref` (lr 3e-4, rerank 16 kandidat) | 61,38 | 35,63 | 0,90 | 0,77 |
| | `A_lvprompt` (nama tingkat disebut di instruksi) | **62,78** | **37,67** | 0,86 | 0,65 |
| | `B_os5_ls` (A + oversampling ×5 + label smoothing 0,1; pemenang valid) | 61,96 | 35,94 | 0,90 | 0,76 |
| | `C_merge` (B + data tambahan) | gagal: CUDA out of memory | | | |

- Cendol menerjemahkan lebih baik (chrF++ sekitar +3 dibanding M2M100), tetapi lebih jarang menuruti tag Krama Alus.
- `output_run1_bug/` adalah run pertama yang rusak (train loss > 200, chrF++ sekitar 15); hasilnya diganti oleh `output/`.
- Weights Cendol tidak tersimpan; angka diambil dari summary di log kernel.

**Post-editing leksikon Krama → Krama Alus** (`kaggle/final_3lv_postedit/`, dijalankan lokal di CPU, tanpa training MT).
Output model demo yang diminta Krama Alus diedit dengan leksikon dari train; pemicu memakai reranker, bukan juri.
Mode dipilih di valid lewat simulasi (referensi Krama sebagai output meleset, referensi Krama Alus sebagai output benar):
di sana `trig_gain` menaikkan ketepatan Krama Alus output meleset dari 0,00 ke 0,65 dan chrF++ dari 55,24 ke 70,66.

| Input (prediksi tersimpan v2) | Mode | chrF++ | BLEU | Ketepatan tingkat | Krama Alus | Output diubah |
|---|---|---|---|---|---|---|
| `rerank2.0` (demo) | tanpa post-editing | 58,62 | 33,17 | 0,85 | 0,57 | - |
| | `trig_gain` (pilihan valid) | 58,62 | 33,17 | 0,85 | 0,57 | 0 dari 75 |
| `beam4` | tanpa post-editing | 58,13 | 33,13 | 0,81 | 0,48 | - |
| | `trig_gain` | 58,26 | 33,29 | 0,81 | 0,48 | 2 dari 75 |

- **Hasil negatif.** Post-editing hampir tidak pernah aktif di test: output Krama Alus yang meleset tidak mengandung
  satu pun dari 57 kata sumber leksikon (model tidak menulis *sampeyan*/*kula* saat diminta Krama Alus), dan 19 dari 32
  output yang dinilai Krama sudah memakai kosakata alus (*adalem*, *panjenenganipun*, *dhahar*, *tindak*). Kesalahannya
  ada di struktur kalimat, makna, atau penilaian juri, bukan kata Krama yang tertinggal.
- Simulasi valid dari referensi manusia tidak mewakili jenis kesalahan model, jadi hasil valid yang bagus tidak terbawa ke test.
- **Noise juri lebih besar dari perkiraan.** Prediksi yang sama persis dengan model demo dinilai 0,85 oleh juri run ini
  (dilatih di CPU) dan 0,92 oleh juri kernel v2 (GPU). Ketepatan Krama Alus 0,57 vs 0,77.

**Self-training dengan rerank** (`kaggle/final_3lv_selftrain/`, kernel `mdevian/unggah-ungguh-final-3lv-selftrain`).
Teacher = resep demo (output beam4 identik 100% dengan model demo). Teacher menerjemahkan kalimat Indonesia yang belum
punya referensi Krama Alus (16 kandidat, rerank λ=2); kandidat disimpan jika P(Krama Alus) reranker ≥ 0,9.
Student = resep demo + pseudo-data. Ketepatan di tabel memakai **juri tetap** (lihat di bawah).

| Varian | Pseudo-data | Skor valid | chrF++ | BLEU | Ketepatan tingkat | Krama Alus |
|---|---|---|---|---|---|---|
| teacher (rerank 4.0) | - | **92,08** | 58,59 | 33,10 | 0,947 | 0,867 |
| `st_train` (rerank 0.5) | 185 dari 385 kalimat train | 90,91 | **59,46** | 33,09 | 0,951 | 0,880 |
| `st_train_nusax` (rerank 0.5) | + 159 dari 333 kalimat NusaX | 89,96 | 58,02 | 32,26 | **0,960** | **0,920** |

- **Teacher tetap menang di valid**; tidak ada student yang dipilih. Selisih ketepatan di test hanya 1–4 dari 75
  kalimat, di bawah noise satu juri (0,02–0,05).
- Penyebab utama: reranker hanya menilai *seberapa alus*, bukan *apakah benar*. Banyak pseudo-data lolos filter
  (P rata-rata 0,96) tetapi maknanya rusak, misalnya "tenaga kerja" → *garwa* (istri), "uangnya" → *panjenengan
  dhahar* (makan). Perbaikan berikutnya perlu filter makna (mis. ambang log-prob model atau kesamaan dengan
  terjemahan Krama), bukan hanya filter tingkat.
- `pseudo_nusax.csv` (turunan NusaX, CC BY-SA) tidak disimpan di repo.

**Juri tetap** (`scripts/juri_tetap.py`, hasil lengkap di `analisis/juri_tetap/README.md`). Juri DistilBERT dilatih
sekali dengan 3 seed (42, 13, 2024) memakai resep v2; juri tetap = ensemble (rata-rata probabilitas). Semua output
kontrol 3 tingkat dinilai ulang. Akurasi ensemble pada referensi test: 0,949 (Krama Alus 0,889).

| Output | Juri kernel (lama) | **Juri tetap** | Krama Alus (lama → tetap) |
|---|---|---|---|
| Model demo (v2 `m2m_base` rerank 2.0) | 0,916 | **0,947** | 0,773 → **0,867** |
| Model demo tanpa rerank (beam4) | 0,858 | 0,898 | 0,613 → 0,747 |
| v2 `m2m_na_ls` rerank 1.0 | 0,911 | 0,964 | 0,787 → 0,920 |
| merge `aug_nusax` rerank 0.5 | 0,920 | 0,938 | 0,813 → 0,853 |
| NLLB `nllb_na` rerank 2.0 | 0,844 | 0,916 | 0,600 → 0,787 |

- Output identik (model demo di kernel v2, merge `base`, post-editing, teacher self-training) sekarang mendapat angka
  yang sama persis (0,947); sebelumnya 0,85–0,94 tergantung kernel.
- Satu juri seed tunggal berbeda 0,02–0,06 dari juri seed lain untuk output yang sama. Angka ketepatan lama di README
  ini (juri kernel) sebaiknya dibaca sebagai perkiraan kasar.
- Pemilihan varian di setiap kernel tetap memakai juri kernel di valid; juri tetap hanya dipakai untuk pelaporan, jadi
  tidak boleh dipakai untuk memilih ulang varian berdasarkan test.

**Catatan perbandingan:** classifier penilai dilatih ulang di setiap kernel. Prediksi `base` di kernel merge
identik 100% dengan model demo, tetapi ketepatannya 0,94 di sana dan 0,92 di kernel v2 (13 dari 225 output diberi
label berbeda). Ketepatan tingkat hanya adil dibandingkan di dalam satu kernel. Antar run, selisih ketepatan bisa mencapai
0,07 (Krama Alus 0,20) untuk output yang identik (lihat post-editing di atas), jadi selisih antar run tidak bermakna.

## Menjalankan demo di cloud (Kaggle + link Gradio)
Hosting Gradio di Hugging Face Spaces (cpu-basic) sekarang butuh langganan PRO, jadi demo dijalankan
di kernel CPU Kaggle `mdevian/unggah-ungguh-demo-gradio` (folder `kaggle/demo/`) dengan `share=True`.
Link publik `*.gradio.live` dikirim ke topik ntfy di `kaggle/demo/ntfy_topic.txt`.

```bash
cd kaggle/demo && kaggle kernels push -p .
```

Tunggu 2-3 menit, lalu ambil link dari field `message`:

```bash
curl -s "https://ntfy.sh/$(cat kaggle/demo/ntfy_topic.txt)/json?poll=1"
```

- Demo hidup selama kernel berjalan (maksimal sekitar 11,5 jam), lalu harus dijalankan ulang.
- Kecepatan di CPU Kaggle dengan rerank (diukur di model 3 tingkat v2): sekitar 4 detik untuk satu tingkat,
  sekitar 17 detik untuk tiga tingkat. Panggilan pertama tanpa pemanasan butuh ~100 detik, jadi `app.py`
  menjalankan satu terjemahan pemanasan saat start. Tanpa rerank (model 4 tingkat lama): 2,5 detik / 3,4 detik.
- Kernel memakai CPU, jadi tidak memakai kuota GPU mingguan. Akun Kaggle dibatasi 2 batch session
  sekaligus, jadi jangan menyalakan demo saat dua kernel training sedang berjalan.
- File ini tetap berformat Space card Hugging Face; jika berlangganan PRO, folder `demo/` (ditambah
  folder `model/` berisi weights) bisa langsung di-upload sebagai Space.

## File
- `app.py`: aplikasi Gradio (lokasi model diambil dari variabel lingkungan `MODEL_DIR`, default `./model`).
- `requirements.txt`: dependensi untuk Space.
- `../kaggle/demo/demo_kaggle.ipynb`: versi kernel Kaggle (isi `app.py` + launcher `share=True` + kirim link ke ntfy).

## Contoh output
Model 3 tingkat v2 + rerank (diambil dari demo yang berjalan di Kaggle):

| Tingkat | "Bapak sudah pulang dari kantor." | "Anakmu perutnya sakit, katanya akan buang air besar." |
|---|---|---|
| Ngoko | Bapak wis mulih saka kantor. | Anakmu wetenge lara, jare arep ngising. |
| Krama | Bapak sampun mantuk saking kantor. | Anak sampeyan wetengipun sakit, criyosipun badhe wawratan. |
| Krama Alus | Ibu panjenengan sampun kondur saking kantor. | Ingkang putra padharanipun gerah, ngandikanipun badhe bobotan. |

Output Krama Alus untuk kalimat pertama memakai kosakata hormat yang tepat (*panjenengan, kondur*), tetapi
"Bapak" berubah menjadi "Ibu panjenengan". Ini contoh kesalahan makna yang tidak tertangkap oleh metrik tingkat.

## Keterbatasan
- Tingkat alus paling sering meleset: permintaan Krama Alus kadang menghasilkan Krama biasa.
- Dilatih pada sekitar 1.600 kalimat pendek (rata-rata 8 kata) dari kamus dan buku unggah-ungguh;
  kalimat panjang, istilah teknis, dan dialek bisa diterjemahkan kurang tepat.
- Output belum diperiksa penutur Jawa.
- Dataset berlisensi CC BY-NC 4.0; demo hanya untuk keperluan riset dan kuliah.
