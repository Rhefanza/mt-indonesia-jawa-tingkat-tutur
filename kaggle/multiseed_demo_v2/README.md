# multiseed_demo_v2 — sensitivitas seed resep demo (protokol C)

Hasil diskusi Claude ↔ Codex `2026-11-22-langkah-sesudah-hasil-negatif` (folder ruang diskusi). Model demo **dibekukan**
(M2M100-418M v2 + rerank, seed 42). Kernel ini hanya mengukur seberapa besar angka demo berubah bila seed penerjemah
diganti; **tidak ada pemilihan antar-seed** dan model demo tidak diganti apa pun hasilnya.

## Protokol (dikunci sebelum push)

- Notebook `multiseed_demo_v2.ipynb` SHA-256 `ca493322aee89340a6ccd53d000116d0c81a0dadaed036e49e064987d3383f72`.
- Resep `m2m_base` dari `kaggle/final_3lv_v2` identik (fungsi translate/nbest/seq_scores/juri diverifikasi sama teks).
  Seed penerjemah 42 (reproduksi), 43, 44. Split, data, hparams sama.
- Dikunci untuk semua seed: juri pemilih checkpoint (DistilBERT, dilatih sekali dengan kode v2, seed 42); reranker =
  weights demo dari output v2 (`model/reranker`, dimuat); decoder beam 8 + rerank λ = 2 tanpa grid; pemilihan checkpoint
  chrF++ + 30 × ketepatan juri kernel (epoch pertama bila seri).
- Fase valid ketiga seed dulu. **Gerbang:** fase test hanya jalan bila seed 42 mereproduksi valid chrF++ 62,66 dan
  Krama Alus 68,83 (dua desimal) dan ketiga seed lengkap. Bila gagal: dicatat, angka demo tetap hasil historis satu seed,
  dan hasil valid multi-seed tetap boleh dilaporkan sebagai sensitivitas valid.
- Pelaporan: chrF++ dan BLEU pada 176 pasangan test, per tingkat dan keseluruhan, nilai per seed + rerata + min–maks.
  Ketepatan tingkat pada 225 permintaan kontrol dengan **juri tetap** lokal (`python scripts/juri_tetap.py`).
  Selisih > 2 poin antara seed 42 dan seed lain hanya penanda deskriptif, bukan uji. Test pernah dipakai untuk diagnosis,
  jadi hasil ini retrospektif.

## Hasil (2026-11-23, T4, 38 menit) — gerbang reproduksi GAGAL, fase test tidak dijalankan

| Run | Epoch terpilih | Valid chrF++ all | Ngoko | Krama | Krama Alus | Juri tetap valid (KA) |
|---|---|---|---|---|---|---|
| Demo v2 asli (seed 42) | 14 | **62,66** | 57,28 | 64,54 | **68,83** | 0,983 (0,963) |
| Ulang seed 42 | 12 | 62,17 | 56,70 | 64,34 | 67,81 | 0,994 (1,000) |
| Seed 43 | 14 | 62,65 | 57,88 | 64,42 | 67,82 | 0,983 (1,000) |
| Seed 44 | 14 | 62,40 | 58,08 | 63,85 | 67,50 | 0,983 (1,000) |

Ketepatan juri tetap dinilai lokal (ensemble `models/juri_tetap/`) pada 175 output valid. Prediksi demo asli diambil
dari `kaggle/training_makna_tingkat_v1/output/valid_demo.csv`.

**Mengapa gerbang gagal.** Training seed 42 terulang **persis**. chrF++ greedy valid per epoch identik dengan
`kaggle/final_3lv_v2/output/metrics_m2m_base.json`, misalnya epoch 11–15 61,10 / 61,69 / 61,62 / 61,73 / 61,77. Yang berbeda
hanya juri kernel. Juri baru akurasinya 0,9543 di referensi valid (KA 0,778), sedangkan juri v2 0,96 (KA 0,815). Juri
ini memberi epoch 12 skor 90,84 dan epoch 14 skor 90,53, padahal di v2 epoch 14 menang tipis (90,88 vs 90,84). Juri
classifier di-inisialisasi sebelum `seed_all`, jadi tidak bisa diulang walaupun kodenya sama. Sesuai protokol, test
tidak disentuh dan angka test demo tetap hasil historis satu seed.

**Bacaan.**
- Rentang valid resep demo pada tiga seed baru adalah chrF++ 62,17–62,65 dan Krama Alus 67,50–67,82. Run demo asli
  (62,66 / 68,83) ada di ujung atas. Untuk Krama Alus, run asli sekitar 1 poin di atas ketiga run lain.
- Sumber variasi terbesar adalah **pemilihan checkpoint oleh juri kernel**, bukan seed penerjemah. Model yang sama
  (seed 42) bisa jadi demo 62,66 atau 62,17, bergantung pada juri mana yang memilih epoch.
- Ketepatan tingkat menurut juri tetap stabil tinggi di semua run (0,98–0,99 di valid). Tingkat bukan sumber ketidakstabilan;
  angka chrF++ Krama Alus yang lebih rendah sejalan dengan diagnosis bahwa masalahnya makna/kosakata.
- Untuk laporan: angka test demo (58,62) adalah satu run yang di valid berada di ujung atas sebaran. Laporkan rentang
  valid di atas sebagai sensitivitas, dan jangan klaim ketelitian di bawah ~0,5 poin chrF++ total atau ~1 poin Krama Alus.
