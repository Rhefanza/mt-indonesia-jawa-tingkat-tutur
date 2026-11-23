# lc_kontrol_langkah_v1 — apakah keunggulan f100 bertahan bila langkah disamakan?

Kontrol untuk `kaggle/learning_curve_v2`. Protokol diputuskan di diskusi Claude ↔ Codex
`2026-11-23-menambah-data-sesudah-learning-curve.md` (giliran [3]–[6]). Hanya **valid**; test tidak dibaca; demo tidak diganti.

## Masalah yang dikontrol

Di learning curve v2 epoch tetap 15, jadi subset kecil juga mendapat langkah update dan jadwal LR yang lebih pendek
(f050: 64 batch/epoch × 15 = 960 langkah; f100: 127 × 15 = 1.905). Kemiringan kurva bisa berasal dari langkah, bukan data.

## Protokol (dikunci sebelum push)

- Lengan: f100 (ulang), f050_s0, f050_s1. Grup identik dengan v2 (assert `grup_sha256`).
- Semua lengan **1.905 langkah terjadwal**, warmup 190, jadwal linear sama; f050 berhenti di 29 epoch + 49 batch.
  Seed training 42, batch, LR, OS, multi-task, reranker, decoder beam 8 + λ 2 sama dengan v2.
- **Angka utama:** chrF++ valid rerank pada bobot **titik akhir** (tidak bergantung juri kernel).
  Sekunder: checkpoint pilihan aturan demo (`*_pilih.csv`). Juri tetap dinilai lokal.
- Update optimizer aktual dicatat (langkah yang dilewati GradScaler). Jika berbeda dari 1.905, kontrol hanya setara
  langkah *terjadwal*.
- Cek kewarasan: greedy chrF++ per epoch f100 identik dengan f100 v2.
- Aturan baca atas G = chrF++ total titik akhir f100 − f050_sX ([asumsi], bukan uji signifikansi):
  kedua G ≥ 3,5 → "keunggulan f100 bertahan pada anggaran langkah yang sama"; kedua |G| ≤ 1,0 → "kedua subset mendekati
  f100 pada kontrol ini"; kedua G < −1,0 → pembalikan; selain itu → tidak ada klaim mekanisme.
  Angka KA (n = 27) hanya dilaporkan.
- Batasan: tidak menyamakan jumlah token; satu seed training; tidak menguji penambahan khusus KA.

SHA-256 notebook saat dikunci: `f0c6281c9cefb93c34af23fb85d1b7c052b1d27317255eac112bfac34cd5e0a1`
