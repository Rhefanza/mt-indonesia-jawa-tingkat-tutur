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

## Hasil (run 2026-11-23, valid saja, 2.507 detik di T4)

Cek kewarasan **lolos**: greedy chrF++ per epoch f100 identik dengan v2 (15 nilai). Grup f050 cocok dengan hash v2.
Langkah: semua lengan 1.905; update optimizer aktual f100 1.904 (1 dilewati GradScaler), f050 1.905.

| Lengan | Kalimat | Epoch | chrF++ titik akhir: semua | Ngoko | Krama | KA | Juri tetap semua / KA | v2 (15 epoch, pilih) |
|---|---|---|---|---|---|---|---|---|
| f100 | 1.395 | 15 | **62,50** | 57,53 | 64,22 | 68,27 | 0,983 / 0,963 | 62,66 |
| f050_s0 | 700 | 29 + 49 batch | **54,92** | 50,58 | 57,15 | 58,04 | 0,977 / 0,963 | 55,18 |
| f050_s1 | 697 | 29 + 49 batch | **57,62** | 54,67 | 58,86 | 60,45 | 0,977 / 0,889 | 55,96 |

Gap titik akhir G = f100 − f050: **s0 7,58** (bootstrap berpasangan per grup valid, B = 1.000, CI90 5,70–9,51),
**s1 4,88** (CI90 2,97–6,87); rata-rata 6,23. P(G ≤ 0) = 0 pada kedua subset.
Aturan baca yang dikunci: kedua G ≥ 3,5 → **"keunggulan f100 bertahan pada anggaran langkah yang sama."**
Checkpoint pilihan (sekunder): f100 62,17 (juri kernel memilih epoch 12), f050_s0 54,92, f050_s1 57,69.
Angka per lengan: `output/manifest.json`, `analisis_lokal.json`.

## Bacaan

- Menggandakan langkah untuk f050 (960 → 1.905) hanya menaikkan rata-rata f050 dari 55,57 ke 56,27 (+0,7);
  gap terhadap f100 tetap 6,23 dari 7,09 di v2. Jadi sebagian besar kemiringan learning curve v2 **bukan** efek langkah.
- Kalimat PPT yang sekarang didukung: "Dengan jumlah langkah training yang sama (1.905), model yang dilatih pada seluruh
  train (1.395 kalimat) unggul 4,9–7,6 chrF++ valid atas model pada separuh data (dua subset acak)." Tetap dengan batasan:
  satu seed training, jumlah token tidak disamakan, valid saja, dan tidak memprediksi skor pada data baru atau 2× data.
- Juri tetap tidak berbeda berarti (KA 0,89–0,96 di semua lengan): data menambah kualitas terjemahan, bukan kontrol tingkat.
- Selisih antar-subset f050 (2,7) masih besar; isi subset ikut menentukan, jadi angka gap jangan dikutip tanpa rentangnya.
