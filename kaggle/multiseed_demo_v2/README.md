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
