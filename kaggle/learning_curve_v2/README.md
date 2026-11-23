# learning_curve_v2 — apakah skor masih naik bila data ditambah?

Pertanyaan: apakah kelemahan terjemahan (terutama Krama Alus) karena data train kurang? Kernel ini melatih ulang resep
demo `m2m_base` pada subset train 25 / 50 / 75 / 100 % dan hanya mengukur **valid**. Test tidak dibaca, dan model demo
tidak diganti.

## Desain (ditulis sebelum push)

- Subset diambil per **grup** (semua tingkat satu kalimat sumber ikut), stratified: 210 grup yang punya Krama Alus dan
  386 grup lain diambil terpisah, jadi proporsi KA tetap. Subset bertingkat: 25 ⊂ 50 ⊂ 75 ⊂ 100.
  Ukuran (permutasi 0): 347 / 700 / 1.051 / 1.395 kalimat, KA 52 / 105 / 158 / 210, pasangan Krama→KA 51 / 104 / 156 / 206.
- Dua permutasi subset (0 dan 1) untuk fraksi < 100 %, jadi 7 run. Seed training 42 untuk semua run.
- Resep, juri kernel (satu set weights), reranker demo (dimuat dari output v2), decoder beam 8 + λ 2, dan aturan pilih
  checkpoint identik dengan `kaggle/multiseed_demo_v2`. Jumlah epoch tetap 15, jadi subset kecil mendapat lebih sedikit
  langkah update (konvensi learning curve biasa).
- Cek kewarasan: run 100 % harus memberi chrF++ greedy per epoch identik dengan ulang seed 42 di `multiseed_demo_v2`.
- Ketepatan tingkat dilaporkan dengan juri tetap lokal.

## Cara membaca

- Kurva Krama Alus masih naik jelas dari 75 % ke 100 % (melebihi selisih antar-permutasi) → data KA kurang;
  menambah data KA yang benar kemungkinan membantu.
- Kurva sudah datar → masalahnya jenis data, cakupan kosakata, atau metrik, bukan jumlah.
- Satu titik 100 % hanya satu run; sebaran seed dari `multiseed_demo_v2` (±0,5 chrF++ total, ±1 KA) dipakai sebagai
  ukuran derau.

## Hasil (run 2026-11-23, valid saja, 3.288 detik di T4)

Cek kewarasan **lolos**: chrF++ greedy epoch 11–15 run 100 % identik dengan seed 42 `multiseed_demo_v2`
([61,10, 61,69, 61,62, 61,73, 61,77]). Juri kernel memilih epoch 14, jadi run 100 % sama persis dengan demo
(valid 62,66, KA 68,83). Subset lokal direkonstruksi dan `grup_sha256` cocok untuk ketujuh run.

chrF++ valid (rerank), rata-rata dua permutasi untuk fraksi < 100 % (s0 / s1 di kurung):

| Fraksi | Kalimat | Semua | Ngoko | Krama | Krama Alus | Juri tetap semua | Juri tetap KA | OOV token ref |
|---|---|---|---|---|---|---|---|---|
| 25 % | 347 | 48,09 (47,60 / 48,58) | 45,57 | 49,29 | 50,03 (48,47 / 51,58) | 0,946 | 0,944 | 0,383 |
| 50 % | ~700 | 55,57 (55,18 / 55,96) | 51,17 | 57,29 | 60,07 (60,45 / 59,68) | 0,980 | 0,944 | 0,302 |
| 75 % | ~1.050 | 59,65 (58,75 / 60,54) | 55,53 | 61,02 | 64,46 (64,77 / 64,14) | 0,986 | 0,944 | 0,257 |
| 100 % | 1.395 | 62,66 | 57,28 | 64,54 | 68,83 | 0,983 | 0,963 | 0,229 |

Fit `chrF++ = a + b·log2(n)` (7 titik): kenaikan per penggandaan data **+7,3** (Ngoko +6,0, Krama +7,5, KA +9,2),
RMSE 0,4–1,2. OOV = proporsi token referensi valid yang tidak pernah muncul di sisi Jawa train subset.
Angka per run: `analisis_lokal.json`; plot: `learning_curve.png`.

## Bacaan

- **Kurva belum jenuh.** Dari 75 % ke 100 % chrF++ total naik +3,0 dan KA +4,4, jauh di atas selisih antar-permutasi
  (≤1,8) dan sebaran seed (±0,5 total, ±1 KA). Jadi, ya: kualitas terjemahan masih dibatasi jumlah data.
- **Yang dibatasi data adalah kosakata/makna, bukan kontrol tingkat.** Juri tetap KA sudah 0,93–0,96 sejak 25 %
  (347 kalimat) dan tidak bergerak; yang naik adalah chrF++. Bahkan dengan seluruh train, 23 % token referensi valid
  tidak pernah muncul di train, dan angka ini turun mulus seiring data (38 → 30 → 26 → 23 %). Ini cocok dengan
  diagnosis makna sebelumnya (18 dari 27 output KA test salah makna, bukan salah tingkat).
- KA paling curam (+9,2 per penggandaan) karena datanya paling sedikit (52 → 210 kalimat).
- **Batasan:** epoch tetap 15, jadi subset kecil juga mendapat langkah update lebih sedikit; sebagian kemiringan bisa
  berasal dari itu. Satu seed training. Ekstrapolasi (2× data ≈ chrF++ 70) hanya ilustrasi: kurva log biasanya
  melandai, dan data tambahan dari sumber lain (mis. NusaX, KA sintetis) terbukti tidak setara (`final_3lv_merge`).
  Implikasinya: yang paling mungkin membantu adalah **data paralel KA berkualitas yang ditulis/diperiksa penutur**,
  bukan trik decoding.
