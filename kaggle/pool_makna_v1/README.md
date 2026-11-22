# pool-makna-v1: pool kandidat + skor makna

Rancangan disepakati di diskusi Claude ↔ Codex `2026-11-22-kualitas-terjemahan-tingkat-tutur.md`
(folder `D:\claude\diskusi bro 1 dan bro 2\`). Model demo v2 tidak dilatih ulang.

**Diagnosis yang mendasari:** dari 27 output Krama Alus test (model demo), 18 salah makna saja, 5 salah makna dan arah hormat,
1 salah tingkat saja, 3 lolos (audit Claude + Codex, belum dicek penutur). Skor rerank v2 (log-prob + 2 × log P(tingkat))
tidak punya sinyal makna, dan kandidat n-best v2 tidak pernah disimpan.

## Alur (jangan diubah urutannya)

1. **Kunci uji fakta** (sudah): `fakta_valid.csv`, 20 pasangan referensi Krama Alus valid vs versi dengan satu fakta diubah
   (objek 8, predikat 4, negasi 3, subjek 3, waktu 2). Isinya ditanam di notebook dan dicek SHA-256
   (`f948ec86…46c3f83`). Disusun Claude, belum dicek penutur Jawa. Jangan diubah setelah run A dimulai.
2. **Run A (valid, GPU):** `cd kaggle/pool_makna_v1 && kaggle kernels push -p .`, lalu unduh output ke `kaggle/pool_makna_v1/output/`
   (`kaggle kernels output mdevian/unggah-ungguh-pool-makna-v1 -p output`). Isinya pool beam8 dan beam16 (decode terpisah) untuk 175 valid,
   skor forward/reranker/juri valid/reverse, uji fakta reverse, weights reverse. Estimasi 1,35–2 jam T4.
   Cek di log: baseline valid chrF++ harus ≈ 62,66 (angka v2).
3. **Seleksi (lokal, CPU):** `python scripts/pilih_pool_makna.py seleksi`.
   - Jalankan pertama → menulis `audit_cakupan.csv`. Isi tiga kolom 0/1 per kandidat dengan sumber Indonesia sebagai patokan fakta.
   - Jalankan kedua → menghitung varian, memilih pemenang, **mengunci** `pilihan.json`, dan membuat kernel run B di `kaggle/pool_makna_v1_test/`.
   - Setelah terkunci, seleksi ulang ditolak.
4. **Run B (test, GPU):** `cd kaggle/pool_makna_v1_test && kaggle kernels push -p .` (estimasi 0,3–0,5 jam; lebih lama bila pemenang memakai beam16 + reverse).
   Test 176 pasangan + 225 permintaan kontrol, baseline vs pilihan. Unduh ke `kaggle/pool_makna_v1_test/output/`.
5. **Pelaporan:** `python scripts/juri_tetap.py` (ketepatan tingkat dengan juri tetap; file `control_*.csv` ikut terbaca),
   lalu `python scripts/pilih_pool_makna.py audit-buta` → `audit_buta.csv` untuk penutur (15 grup: 10 Krama Alus, 3 Krama, 2 Ngoko).
   Tahan promosi bila audit buta menunjukkan regresi fakta inti lebih banyak daripada pemulihan.

## Aturan seleksi (dipatok sebelum run A)

- Baseline: beam8, λ = 2. Varian: pool gabungan beam8+beam16, λ = 2, μ ∈ {0; 0,5; 1} (μ = bobot reverse log P(Indonesia | Jawa)).
- Lolos bila batas bawah CI 90% (2.000 paired bootstrap per `group`) selisih chrF++ Krama Alus > 0, kehilangan ≤ 1 keputusan benar
  Krama Alus (juri valid), dan Ngoko/Krama turun ≤ 1 poin chrF++.
- μ > 0 hanya bila uji fakta reverse ≥ 16/20, gerbang cakupan ≥ 3 baris pulih fakta inti tanpa tingkat turun, dan mengungguli μ = 0.
- Pilih chrF++ Krama Alus tertinggi; seri → μ terkecil; tidak ada yang lolos → baseline.

## Penyimpangan dari spesifikasi diskusi (disengaja)

- `fakta_valid.csv` ditanam di notebook + SHA-256, bukan dataset Kaggle terpisah (tidak perlu membuat dataset baru).
- Run B memakai kernel ID terpisah (`…-pool-makna-v1-test`, notebook yang sama dengan `MODE = "test"`), karena output run A
  harus di-mount sebagai input dan kernel tidak bisa diandalkan me-mount output versinya sendiri.
- Kolom skor reranker disimpan sebagai log-prob (`rr_logp_*`), bukan probabilitas, supaya presisi tidak hilang.

## Hasil run A (2026-11-22) — **hasil negatif, dikunci: pemenang = baseline**

- Run A selesai dalam 443 detik (T4). Baseline valid chrF++ **62,66**, sama persis dengan v2. Juri valid akurasi 0,96 pada referensi.
- Uji fakta reverse: **19/20** (gagal hanya g21 "jatuh → makan", selisih 0,002). Model reverse kemungkinan overfit
  (loss train 0,14, skor valid median −1,13 per token).
- Celah oracle sentence chrF++ Krama Alus valid: beam8 rerata 3,36 (8/27 baris ≥3), beam16 rerata 4,05 (10/27 baris ≥3).
- Audit cakupan (`audit_cakupan.csv`, **draf Claude, bukan penutur Jawa**): top-1 kehilangan fakta inti di 15/22 baris,
  tetapi hanya **1 baris** (g207, sebagian) yang punya kandidat pulih tanpa tingkat turun. Syaratnya ≥3, jadi gerbang gagal.
  Kenaikan chrF++ oracle kebanyakan berasal dari kemiripan kata, bukan pemulihan fakta.
- Seleksi (`pilihan.json`): pool gabungan μ=0 menurunkan chrF++ Krama Alus (68,83 → 68,19, lb90 −1,71). Pemenang **baseline**.
  Run B tidak dijalankan, karena baseline vs baseline = hasil v2.
- Eksplorasi sesudah penguncian (tidak dipakai untuk memilih): reverse μ ∈ {0,5; 1; 2; 4} **menurunkan** chrF++ Krama Alus valid
  di semua setelan (selisih rerata −0,7 sampai −3,4, lb90 semuanya negatif), walaupun lolos uji fakta 19/20.
- **Kesimpulan:** perbaikan tidak bisa datang dari rerank. Pool M2M100 jarang memuat kandidat yang menjaga fakta.
  Sisi training/generator yang perlu diubah. Menurut aturan, berikutnya intervensi 3 (pool Cendol), dengan catatan
  Cendol dulu juga lemah di Krama Alus dan data Krama Alus train hanya 210 kalimat.
