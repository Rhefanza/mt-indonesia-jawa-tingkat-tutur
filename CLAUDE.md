# CLAUDE.md — konteks proyek (baca ini dulu di sesi baru)

Proyek UAS NLP (semester 5): **MT Indonesia → Jawa dengan kontrol tingkat tutur**
(Ngoko, Krama, Krama Alus; versi lama juga Ngoko Alus). Tingkat dipilih lewat source prefix tag
(`<ngoko>`, `<krama>`, `<krama_alus>`) di input.

Repo publik: https://github.com/Rhefanza/mt-indonesia-jawa-tingkat-tutur

## Status terakhir
- Implementasi dan evaluasi **sudah selesai**; demo Gradio sudah jalan di Kaggle.
- `IDE_MT_Jawa_Tingkat_Tutur.md` adalah dokumen ide awal. Baris "Status: belum implementasi" di dalamnya
  **sudah usang**. Bagian urgensi dan riset terdahulu di sana masih berguna untuk paper/PPT.
- `referensi/`: related work yang sudah diverifikasi (`README.md` berisi metode dan angka sebelum/sesudah per paper,
  `referensi.bib` untuk sitasi). Pakai ini, bukan daftar riset di dokumen ide (ada angka Wibawa yang keliru di sana).
  Entri bertanda ⚠️ atau `note = {CEK: ...}` belum diverifikasi dari full text.
- Topik lama (MT Jerman → Inggris, EMEA) ditolak. File `Proposal_MT_De-En_*.pptx` dan `data/emea_*.csv`
  adalah sisa topik itu, bukan bagian proyek aktif.
- `Proposal_MT_Jawa_Tingkat_Tutur.pptx` adalah proposal topik Jawa.

## Model final (demo)
- **M2M100-418M** full fine-tuning, 3 tingkat (v2), dilatih di Kaggle kernel
  `mdevian/unggah-ungguh-final-3lv-v2` (seed 42, fp16). Skrip: `kaggle/final_3lv_v2/final_3lv_v2.ipynb`.
- Teknik: init embedding tag dari kata penanda, oversampling Krama Alus ×3, multi-task JV Krama → Krama Alus,
  15 epoch, pilih checkpoint dengan skor valid = chrF++ + 30 × ketepatan tingkat.
- Rerank saat decoding: 8 kandidat beam, skor = log-prob model + 2 × log P(tingkat diminta) dari
  `w11wo/javanese-roberta-small`. Classifier penilai sengaja berbeda (`w11wo/javanese-distilbert-small`)
  supaya ketepatan tingkat tidak naik palsu. **Jangan samakan keduanya.**
- Hasil test (1 seed): chrF++ 58,62 · BLEU 33,17 · ketepatan tingkat 0,92
  (Ngoko 1,00 · Krama 0,97 · Krama Alus 0,77). Tanpa rerank: ketepatan 0,86.
- Hasil multi-seed (3 seed), tabel 4 tingkat dan 3 tingkat: lihat `demo/README.md`.

## Eksperimen sesudah model demo (1 seed, model demo belum diganti)
- `kaggle/final_3lv_merge/`: M2M100 + data tambahan. Terbaik `aug_nusax` (chrF++ 60,44, ketepatan 0,92);
  NusaX membantu Ngoko, Krama Alus sintetis dari leksikon justru menurunkan ketepatan Krama Alus.
- `kaggle/final_3lv_cendol*/`: Cendol mT5-base. chrF++ ~62 (lebih tinggi dari M2M100) tetapi Krama Alus lebih lemah;
  pemenang valid `B_os5_ls` (61,96, ketepatan 0,90). Varian `C_merge` gagal karena CUDA OOM. Weights tidak tersimpan.
- `kaggle/final_3lv_postedit/` (lokal, CPU): post-editing leksikon Krama → Krama Alus pada prediksi tersimpan v2.
  **Hasil negatif**: hampir tidak aktif di test (0–3 output diubah), karena output Krama Alus yang meleset tidak memuat
  kata sumber leksikon dan sebagian besar sudah berkosakata alus. Jangan dicoba ulang dalam bentuk yang sama.
- `kaggle/final_3lv_selftrain/`: self-training dengan rerank (pseudo Krama Alus dari teacher, filter P reranker ≥ 0,9).
  **Teacher tetap menang di valid**; student hanya beda 1–4 dari 75 kalimat di test. Pseudo-data lolos filter tingkat
  tetapi banyak yang maknanya rusak (reranker tidak menilai makna). Perlu filter makna jika dicoba lagi.
- **Juri tetap** (`scripts/juri_tetap.py`): juri DistilBERT 3 seed + ensemble, weights di `models/juri_tetap/`
  (di-gitignore; skrip melatih ulang jika tidak ada). Hasil di `analisis/juri_tetap/`. **Pakai ini untuk melaporkan
  ketepatan tingkat lintas eksperimen.** Model demo dengan juri tetap: ketepatan **0,947**, Krama Alus **0,867**
  (juri kernel v2: 0,92 / 0,77). Juri kernel tiap run berbeda 0,02–0,07 untuk output identik.
- Pemilihan varian tetap di valid dengan juri kernel; juri tetap hanya untuk pelaporan (jangan memilih ulang di test).
- Simpan prediksi valid di setiap kernel baru (sudah dilakukan di `final_3lv_selftrain`).
- **Diagnosis makna (2026-11-22, diskusi Claude ↔ Codex):** masalah dominan model demo adalah **makna**, bukan tingkat.
  Dari 27 output Krama Alus test: 18 salah makna, 5 makna + arah hormat, 1 tingkat saja, 3 lolos. Rerank tidak punya sinyal makna.
  Grid λ v2 datar di valid (chrF++ 62,42–62,66), jadi **jangan kalibrasi λ lagi**.
- `kaggle/pool_makna_v1/`: pool beam8/beam16 + skor reverse jv→id. **Hasil negatif, dikunci baseline** (`pilihan.json`).
  Reverse lolos uji fakta 19/20 tetapi menurunkan chrF++ Krama Alus valid di semua μ. Audit cakupan (draf Claude):
  top-1 kehilangan fakta di 15/22 baris, hanya 1 yang punya kandidat pulih. **Rerank (λ maupun skor makna) sudah buntu;**
  perbaikan harus dari training/generator. Detail di `kaggle/pool_makna_v1/README.md`. Weights reverse di-gitignore.
- `kaggle/training_makna_tingkat_v1/`: **hasil negatif, dikunci demo** (`pilihan.json`). Pagar zero-shot 1.3B gagal
  (Ngoko 42,90 vs 600M 41,67, p5 −0,60) jadi 1.3B tidak dilatih. NLLB-600M paket demo: Ngoko 60,07 (demo 57,28) tetapi
  Krama Alus 64,82 (demo 68,83), CI90 −7,97. Konverter Krama→KA input-bersih: chrF++ 72,84 dan juri 22/26 tetapi audit draf
  merusak fakta inti 13/26 (menebak kata alus seperti `pundhut`). Detail di README folder itu. Test tidak dijalankan.
  Metrik tingkat + chrF++ bisa naik sambil makna turun: **selalu audit makna sebelum klaim perbaikan KA.**
- `kaggle/multiseed_demo_v2/` (protokol C dari diskusi `2026-11-22-langkah-sesudah-hasil-negatif`): **gerbang reproduksi gagal,
  test tidak dijalankan.** Training seed 42 terulang persis (chrF++ greedy per epoch identik), tetapi juri kernel baru memilih
  epoch 12, bukan 14, sehingga valid 62,17 / KA 67,81 (demo 62,66 / 68,83). Seed 43/44: 62,65 / 62,40, KA 67,82 / 67,50.
  Demo asli ada di ujung atas sebaran valid; variasi terbesar berasal dari juri pemilih checkpoint. Juri tetap valid 0,98–0,99 untuk semua run.
  Demo **dibekukan**; tidak ada eksperimen model lagi yang direncanakan. Sisa tenaga untuk laporan/PPT.
- `data/external/` (NusaX, CC BY-SA) dan `data/merged/` di-gitignore karena lisensi; buat ulang dengan
  `scripts/build_lexicon_augment.ipynb`. Tabel lengkap di `demo/README.md`.

## Dataset
- **Unggah-Ungguh** (HF: `JavaneseHonorifics/Unggah-Ungguh`, CC BY-NC 4.0) di `data/unggah_ungguh/`
  (`translation.csv`, `conversation.csv`, `split/`). Untuk 3 tingkat: train 1.395 / valid 175 / test 176.
- Split dibuat oleh `scripts/split_data.py`. 176 kalimat test yang sama dipakai di semua perbandingan.
- `data/east_java/manual_east_java.csv`: data manual dialek Jawa Timur.
- Lisensi non-komersial: demo dan model hanya untuk riset/kuliah.

## Struktur folder
- `kaggle/<nama>/`: satu folder per eksperimen (notebook `.ipynb`, `kernel-metadata.json`, `output/` berisi
  log, metrics, prediksi). `baseline_zeroshot` → `finetune*` → `finetune_r2*` → `finetune_r3*` →
  `finetune_3lv*` → `final_*` adalah urutan kronologis eksperimen. `*_nllb` dan `*_m2m100` = dua model dibanding.
- `kaggle/demo/`: kernel demo Kaggle (`demo_kaggle.ipynb` = isi `demo/app.py` + launcher `share=True` + notifikasi ntfy).
- `demo/`: `app.py` (Gradio), `requirements.txt`, `README.md` (juga Space card HF, berisi hasil lengkap).
- `analisis/`: EDA dataset. `scripts/`: utilitas data.

## Cara menjalankan demo
Detail di `demo/README.md` (bagian "Menjalankan demo di cloud"). Ringkas: push kernel dengan
`cd kaggle/demo && kaggle kernels push -p .`, lalu ambil link `*.gradio.live` dari topik ntfy.
- **`kaggle/demo/ntfy_topic.txt` TIDAK ada di repo** (di-gitignore karena sifatnya seperti secret).
  Di mesin lain, buat sendiri file itu berisi satu topik ntfy acak.
- **Weights model tidak ada di repo** (hanya config/tokenizer). Weights ada di output kernel Kaggle
  `mdevian/unggah-ungguh-final-3lv-v2`; `app.py` memuat dari `MODEL_DIR` (default `./model`).
- Hugging Face Spaces cpu-basic butuh PRO, makanya demo di Kaggle (CPU, maksimal ~11,5 jam per run).
- Akun Kaggle maksimal 2 batch session sekaligus.

## Keterbatasan yang diketahui (jangan diklaim lebih dari ini)
- Krama Alus paling sering meleset (kadang keluar Krama biasa).
- Data kecil (~1.600 kalimat pendek, rata-rata 8 kata); kalimat panjang/teknis/dialek lemah.
- Output **belum diperiksa penutur Jawa**.
- Classifier penilai akurasinya 0,70–0,81 pada referensi alus, jadi angka alus cenderung under-estimate.
- Kesalahan makna (mis. "Bapak" → "Ibu panjenengan") tidak tertangkap metrik tingkat.
- Hasil model demo hanya 1 seed.

## Preferensi pengguna
- Bahasa: Indonesia. **Istilah teknis Inggris jangan diterjemahkan** (open-source, fine-tuning, rerank, dll.).
- Commit/PR **atas nama pemilik repo saja**: jangan tambahkan `Co-Authored-By` Claude atau "Generated with Claude"
  (ada hook yang menolaknya).
- Sebelum menambah file ke repo publik, cek secret dan lisensi data; jangan upload weights/data berlisensi di luar CC BY-NC.

## Belum dikerjakan / ide lanjutan
- Belum ada file `LICENSE` di repo (dataset CC BY-NC 4.0).
- Evaluasi oleh penutur Jawa; paper/laporan dan PPT final (belum ada di repo).
- Tambahkan di sini setiap keputusan penting baru supaya sesi berikutnya tidak mengulang pekerjaan.
