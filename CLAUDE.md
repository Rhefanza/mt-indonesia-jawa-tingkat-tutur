# CLAUDE.md — konteks proyek (baca ini dulu di sesi baru)

Proyek UAS NLP (semester 5): **MT Indonesia → Jawa dengan kontrol tingkat tutur**
(Ngoko, Krama, Krama Alus; versi lama juga Ngoko Alus). Tingkat dipilih lewat source prefix tag
(`<ngoko>`, `<krama>`, `<krama_alus>`) di input.

Repo publik: https://github.com/Rhefanza/mt-indonesia-jawa-tingkat-tutur

## Status terakhir
- Implementasi dan evaluasi **sudah selesai**; demo Gradio sudah jalan di Kaggle.
- `IDE_MT_Jawa_Tingkat_Tutur.md` adalah dokumen ide awal. Baris "Status: belum implementasi" di dalamnya
  **sudah usang**. Bagian urgensi dan riset terdahulu di sana masih berguna untuk paper/PPT.
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
- Juri dilatih ulang tiap kernel, jadi ketepatan tingkat **tidak bisa dibandingkan antar kernel** (noise ~0,02).
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
- Evaluasi oleh penutur Jawa; multi-seed untuk model demo v2; paper/laporan dan PPT final (belum ada di repo).
- Tambahkan di sini setiap keputusan penting baru supaya sesi berikutnya tidak mengulang pekerjaan.
