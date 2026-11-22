# training-makna-tingkat-v1: model dasar lebih besar + konverter Krama → Krama Alus

Rancangan disepakati di diskusi Claude ↔ Codex `2026-11-22-perbaikan-sisi-training-data.md`
(folder `D:\claude\diskusi bro 1 dan bro 2\`). Latar: semua perbaikan di tahap decoding buntu (`kaggle/pool_makna_v1/`),
karena pool kandidat jarang memuat terjemahan yang benar. 22,5% token sumber test tidak ada di train; NusaX hanya menutup
25/136 token OOV valid. Tuas yang tersisa: pengetahuan model dasar.

## Yang dijalankan (MODE valid, hanya train + valid)

1. Preflight: weights demo v2 + reranker, hash CSV, 1.395/175 baris, tanpa irisan grup/sumber train–valid.
2. Demo v2 di kernel yang sama (beam8, λ=2) → harus mereproduksi chrF++ valid 62,66 dan Krama Alus 68,83.
3. Pagar zero-shot: NLLB-600M vs NLLB-1.3B pada 74 Ngoko valid (tanpa tag, beam4). Lanjut 1.3B bila persentil ke-5
   bootstrap selisih (1.3B − 600M) > 0.
4. Preflight memori 1.3B: 3 update pada blok terpanjang + beam8, microbatch 2 (turun ke 1 bila OOM). Lewati bila gagal
   atau proyeksi melewati pagu 6 jam. Tidak ada pergantian otomatis ke LoRA/multi-GPU.
5. `nllb600_base` dan (bila lolos) `nllb13_base`: paket data demo (1.815 pasangan ID→JV dengan oversampling Krama Alus ×3,
   206 auxiliary Krama→Krama Alus, init tag, 15 epoch, 127 blok/epoch), Adafactor di kedua arm, fp16 AMP,
   gradient checkpointing, checkpoint dipilih dengan greedy chrF++ valid + 30 × ketepatan juri valid. Decode beam8/λ=2.
6. Konverter input-bersih: NLLB-600M dilatih hanya dari 206 pasangan Krama→Krama Alus (15 epoch tetap), diuji dari
   Krama referensi pada 26 pasangan valid, dibandingkan dengan salinan Krama. Diagnosis saja, tidak ikut seleksi.

Push: `cd kaggle/training_makna_tingkat_v1 && kaggle kernels push -p .`
Unduh: `kaggle kernels output mdevian/unggah-ungguh-training-makna-tingkat-v1 -p kaggle/training_makna_tingkat_v1/output`

## Seleksi (lokal) — aturan dipatok sebelum run

`python scripts/pilih_training_makna.py seleksi`
- Berhenti bila reproduksi demo gagal.
- Gerbang angka: batas bawah CI 90% selisih chrF++ Krama Alus vs demo > 0 (2.000 paired bootstrap per grup),
  kehilangan ≤1 keputusan benar Krama Alus (juri valid), Ngoko/Krama turun ≤1 poin.
- Gerbang audit (untuk arm yang lolos angka): `audit_buta_valid.csv` berisi output demo dan arm untuk 27 Krama Alus valid,
  diacak, sistem disembunyikan. Pulih − regresi ≥ 2 (semua fakta inti terjaga) dan jumlah lolos makna+tingkat tidak berkurang.
- Pilih chrF++ Krama Alus tertinggi → jumlah lolos dua sumbu → `nllb600_base`. Tidak ada yang lolos → demo.
- Mengunci `pilihan.json` dan membuat kernel test bila ada pemenang baru; lalu `audit-buta-test` (15 grup test).

## Lembar fakta (sudah diisi sebelum prediksi arm baru ada)

`fakta_ka_valid.csv`: fakta inti 27 Krama Alus valid dari sumber Indonesia, `bergantung_kata_oov` (12/27),
`referensi_bermasalah` (1/27, g669). SHA-256 `ef451588…245dc58`. **Draf Claude, bukan penutur Jawa.** Prediksi demo valid
sudah pernah terlihat di audit pool-makna, jadi lembar ini tidak sepenuhnya buta terhadap demo.

## Klaim untuk laporan

Test sudah dibaca untuk diagnosis, jadi hasil test berikutnya adalah evaluasi retrospektif. Satu seed = satu run.
Audit oleh Claude/Codex adalah draf non-penutur; makna, keluwesan, dan arah hormat yang ambigu tetap perlu penutur Jawa.
