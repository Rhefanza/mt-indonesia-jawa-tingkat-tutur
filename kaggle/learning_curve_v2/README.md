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
