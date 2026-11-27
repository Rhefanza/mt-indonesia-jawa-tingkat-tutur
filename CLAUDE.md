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
- `kaggle/learning_curve_v2/` (valid saja, 7 run, subset per grup stratified 25/50/75/100 %): **kurva belum jenuh**.
  chrF++ valid 48,1 → 55,6 → 59,7 → 62,7 (+7,3 per penggandaan; KA 50,0 → 68,8, +9,2). Juri tetap KA datar 0,93–0,96 sejak 25 %,
  OOV token referensi 38 → 23 %. Jadi data membatasi **kosakata/makna**, bukan kontrol tingkat. Run 100 % = demo persis (cek lolos).
  Batasan: epoch tetap (subset kecil = langkah lebih sedikit), 1 seed; ekstrapolasi hanya ilustrasi.
- **Diskusi 2026-11-23 `menambah-data-sesudah-learning-curve` (SELESAI):** menambah data = **ya bersyarat** (paralel ID–KA
  ditulis/diperiksa penutur, berizin). Kurva v2 **belum mengisolasi efek data**: langkah update & jadwal LR ikut berubah
  (kritik Codex; Sennrich & Zhang 2019). Jangan tulis "kualitas dibatasi data" di PPT sebelum kontrol selesai; pakai redaksi
  deskriptif di Ringkasan Akhir diskusi. Kandidat: NgokoKrama (1.000 pasang "Krama (Alus)", repo **tanpa lisensi** → butuh izin
  tertulis + audit label), NusaX (Ngoko saja), KA sintetis (negatif). Dataset publik Unggah-Ungguh hanya 2.009 baris (HF = lokal).
  Diputuskan satu kontrol valid-saja `lc_kontrol_langkah_v1`: f050_s0/s1 vs f100 ulang, 1.905 langkah terjadwal, warmup 190,
  rerank di titik akhir, catat update optimizer aktual; aturan baca gap ≥3,5 / ±1,0 [asumsi].
  **Hasil (`kaggle/lc_kontrol_langkah_v1/`):** cek f100 lolos; pada 1.905 langkah f100 62,50 vs f050 54,92 / 57,62 →
  gap 7,58 / 4,88 (bootstrap grup CI90 5,70–9,51 / 2,97–6,87) → **"keunggulan f100 bertahan pada anggaran langkah yang sama"**.
  Gandakan langkah f050 hanya +0,7. Klaim "data membatasi kualitas (bukan kontrol tingkat)" kini boleh, dengan batasan 1 seed,
  token tidak disamakan, valid saja, tanpa ekstrapolasi.
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
- **Demo dua arah (2026-11-27):** `kaggle/demo_gradio_live/` (kernel private `mdevian/unggah-ungguh-demo-gradio-live`), notebook
  interaktif bertahap. Tab Indonesia → Jawa (model demo v2 + rerank) dan Jawa → Indonesia (`model_semua` dari
  `reverse-jv-id-v1`, beam 5, tanpa tag; pilihan user meski memuat NgokoKrama, hanya output yang terekspos). `NTFY_TOPIC` di repo
  kosong; saat push, isi di salinan scratchpad dari `kaggle/demo/ntfy_topic.txt`. Push versi baru **tidak** menghentikan versi lama.
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
- **Data gabungan eksternal (2026-11-25, belum dilatih):** `scripts/gabung_data_eksternal.py` → `data/merged/train_eksternal_3lv.csv`
  (di-gitignore). Unggah-Ungguh train + NgokoKrama (3 tingkat) + NusaX + NusaTranslation (Ngoko). Label tingkat eksternal dari classifier
  TF-IDF terpisah (bukan juri/reranker); Krama Alus eksternal masih berisik dan belum diaudit penutur. NgokoKrama **tanpa lisensi**:
  lokal saja, jangan disebarkan (pengguna meminta diambil; opsi `--tanpa-ngokokrama`). Lisensi NusaTranslation tidak konsisten (Apache-2.0/CC BY-SA).
- `kaggle/data_eksternal_v1/` (di-push 2026-11-25 atas permintaan pengguna, **tanpa uji lokal**, private, SELESAI 2026-11-25): 4 lengan
  `base`/`ngokokrama`/`tanpa_ngokokrama`/`semua`, valid saja, 1 seed. Menang = skor valid tertinggi DAN KA chrF++ tidak turun > 1,0 dari base.
  Data diunduh saat run, tidak disimpan di output. Weights pemenang mungkin dilatih NgokoKrama (tanpa lisensi): jangan dibagikan.
  **Hasil: `semua` menang (aturan terkunci), cek base identik lolos.** chrF++ valid base 62,17 → ngokokrama 63,25 / tanpa_ngokokrama 63,41 / semua 64,95
  (+2,78, bootstrap grup CI90 1,14–4,59; nilai demo asli di valid 62,66). Kenaikan jelas hanya **Ngoko** (+5,13; 2,87–7,28); Krama +1,42 dan KA +1,31 CI memuat 0.
  Juri tetap base 0,994 → semua 0,977 (Krama 0,987 → 0,960; KA 1,00 → 0,963). Makna KA: baca sekilas draf Claude (bukan audit, bukan penutur) masih banyak rusak
  (mis. dimandikan → dipunagem). Jangan klaim perbaikan KA. Batasan: 1 seed, langkah ikut naik (1.905 → 3.960), pilih 1 dari 4 di valid, test belum dijalankan,
  demo **belum diganti**. Weights `model_semua` hanya di output kernel private (NgokoKrama tanpa lisensi). Analisis: `kaggle/data_eksternal_v1/analisis_valid.py`.
- `kaggle/reverse_jv_id/` (2026-11-25, valid saja, CPU): model reverse jv→id dari `pool_makna_v1` diukur sebagai penerjemah.
  chrF++ 50,41 / BLEU 30,57 (Ngoko 51,66 · Krama 48,02 · KA 53,61, n=27); zero-shot M2M100 24,42 / 6,13; salin input 23,65 / 6,01.
  Belajar, tetapi **belum layak demo**: banyak kata isi diganti kata lain. Hanya 5 epoch, 1 seed, tanpa data eksternal; `ukur_valid.py` untuk mengulang.
  Training ulang 15 epoch (`mdevian/unggah-ungguh-reverse-jv-id-v1`, private, 2026-11-25, SELESAI): lengan `base` vs `semua`, valid saja.
  `semua` menang jika chrF++ beam 5 > base DAN tidak ada tingkat turun > 1,0. Weights `semua` memuat NgokoKrama: jangan dibagikan.
  **Hasil 15 epoch: `semua` menang (aturan terkunci).** chrF++ / BLEU valid beam 5: base 51,85 / 32,91, semua 55,72 / 35,07 (Ngoko 55,58 · Krama 54,58 · KA 59,24).
  Epoch terbaik base 10, semua 7. Interval kepercayaan belum dihitung; langkah ikut naik (1.320 → 3.375); makna masih sering salah (baca 25 sampel), belum layak demo.
  Varian 10 epoch (permintaan pengguna): `kaggle/reverse_jv_id_ep10/`, kernel `mdevian/unggah-ungguh-reverse-jv-id-ep10`, SELESAI. chrF++ / BLEU: base 52,44 / 32,66,
  semua 55,31 / 33,34 (KA 57,72 / 33,90). Tidak lebih baik dari 15 epoch `semua` (55,72 / 35,07); selisih kecil, 1 seed, valid saja.
- **20 epoch lengan `semua` (2026-11-26, permintaan pengguna, tanpa uji lokal, SELESAI):** `kaggle/data_eksternal_ep20/`
  (Indonesia → Jawa, pembanding semua 15 epoch 64,95) dan `kaggle/reverse_jv_id_ep20/` (Jawa → Indonesia, pembanding 55,72 / 35,07). Valid saja, 1 seed.
  **Hasil: 20 epoch tidak lebih baik dari 15 epoch** (bootstrap grup CI90, `scripts/banding_epoch.py`). Indonesia → Jawa 64,92 vs 64,95 (−0,03; −0,86 s.d. 0,81),
  epoch terpilih 19, KA +1,51 (−0,11 s.d. 3,53), juri tetap total sama 0,977. Jawa → Indonesia 56,76 / 36,36 vs 55,72 / 35,07 (+1,04; −0,70 s.d. 2,70), epoch
  terpilih 20. Tetap pakai 15 epoch; jangan tambah epoch lagi tanpa seed tambahan.
- **Diskusi 2026-11-26 `metode-terbaru-kualitas-terjemahan-jawa` (SELESAI, belum dijalankan):** literatur 2024–2026. JAIC April 2026
  (Prayoga dkk.): Gemini 2.5 Flash zero-shot chrF++ 67,08 vs M2M100 FT 56,39 pada Unggah-Ungguh (kontaminasi mungkin). Uji murah: Wikipedia Jawa
  ~280 ribu kalimat, juri tetap Krama 10–18 %, KA 0,7–2,5 % (`scripts/cek_tingkat_wiki_jv.py`); audit Codex reverse hanya 8/27 KA fakta utuh → **BT ditunda**;
  retrieval top-3 menutup 50 % token referensi KA, 19 % tidak ada di train (`scripts/cek_retrieval.py`). Rencana: (1) pagar few-shot retrieval Qwen3-4B-Instruct
  (± Gemma-3-4B bila user menerima lisensi), 1 GPU-jam; (2) QLoRA pemenang pagar, 4 GPU-jam + replikasi seed 43. Keputusan pakai **audit fakta** (modalitas
  termasuk fakta), bukan recall token. Detail: `D:/claude/diskusi bro 1 dan bro 2/2026-11-26-metode-terbaru-kualitas-terjemahan-jawa.md`.
  **Pagar LLM dijalankan (`kaggle/pagar_llm_v1/`, 2026-11-26, 4 menit T4, VRAM 3,6 GB): GAGAL, jalur QLoRA Qwen ditutup.** chrF++ valid
  zero-shot Ngoko 24,64 · Krama 19,59 · KA 20,45; few-shot 3 contoh 37,56 · 33,40 · 37,66 (demo valid 56,70 · 64,34 · 67,81). Output banyak masih
  bahasa Indonesia / kata Jawa ngawur; bukan masalah format (0 output terpotong ganda). Few-shot +13 s.d. +17 poin tetapi jauh di bawah ambang (turun maks 1 poin).
  Audit fakta tidak diperlukan karena gagal pagar chrF++. Sisa opsi jalur LLM: Gemma-3-4B (butuh user menerima lisensi) atau LLM API (keputusan user).
  **Pagar Gemma-3-4B-it (`kaggle/pagar_llm_gemma/`, Kaggle Models, lisensi disetujui user, FP32 compute, 9 menit): juga GAGAL.** chrF++ valid
  zero 28,21 · 21,83 · 23,79; few-shot 39,44 · 36,39 · 36,04 (Ngoko · Krama · KA). Jalur LLM open-source 4B (prompting dan QLoRA) ditutup;
  sisa jalur LLM hanya API (keputusan user).
