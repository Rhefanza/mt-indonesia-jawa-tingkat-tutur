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
