# Juri tetap: ketepatan tingkat dinilai ulang dengan juri yang sama

Dibuat oleh `scripts/juri_tetap.py`. Juri = `w11wo/javanese-distilbert-small`, resep v2, dilatih di train split
dengan seed 42, 13, 2024; **juri tetap = ensemble** (rata-rata probabilitas). Weights di
`models/juri_tetap/` (tidak di-commit; jalankan ulang skrip untuk membuatnya).

## Akurasi juri pada referensi test asli

| Juri | Semua | Ngoko | Krama | Krama Alus |
|---|---|---|---|---|
| seed42 | 0.9432 | 0.9733 | 0.9324 | 0.8889 |
| seed13 | 0.9489 | 0.9733 | 0.9730 | 0.8148 |
| seed2024 | 0.9489 | 0.9733 | 0.9459 | 0.8889 |
| ensemble | 0.9489 | 0.9733 | 0.9459 | 0.8889 |

## Ketepatan tingkat output (75 kalimat test × 3 tingkat = 225 permintaan)

`juri kernel` = angka asli yang tercatat di kernel masing-masing (juri berbeda tiap run).
`sebaran seed` = selisih terbesar antar juri seed tunggal untuk output yang sama.

| Eksperimen | Output | Juri kernel | **Ensemble** | Ngoko | Krama | **Krama Alus** | Sebaran seed |
|---|---|---|---|---|---|---|---|
| final_3lv_cendol/output_run1_bug | cendol_lr3e-4_beam4 | 0.5956 | **0.5600** | 0.6267 | 0.0533 | **1.0000** | 0.0400 |
| final_3lv_cendol/output_run1_bug | cendol_lr3e-4_rerank1.0 | 0.4978 | **0.4311** | 0.1867 | 0.1067 | **1.0000** | 0.1245 |
| final_3lv_cendol/output_run1_bug | cendol_lr5e-5_beam4 | 0.3822 | **0.3867** | 0.4133 | 0.7467 | **0.0000** | 0.0178 |
| final_3lv_cendol/output_run1_bug | cendol_lr5e-5_rerank0.5 | 0.4222 | **0.4089** | 0.4400 | 0.7867 | **0.0000** | 0.0178 |
| final_3lv_merge | aug_beam4 | 0.8578 | **0.8933** | 1.0000 | 0.9733 | **0.7067** | 0.0445 |
| final_3lv_merge | aug_nusax_beam4 | 0.8622 | **0.8844** | 1.0000 | 0.9600 | **0.6933** | 0.0444 |
| final_3lv_merge | aug_nusax_rerank0.5 | 0.9200 | **0.9378** | 1.0000 | 0.9600 | **0.8533** | 0.0533 |
| final_3lv_merge | aug_rerank4.0 | 0.9111 | **0.9378** | 1.0000 | 0.9867 | **0.8267** | 0.0267 |
| final_3lv_merge | base_beam4 | 0.8933 | **0.8978** | 0.9867 | 0.9600 | **0.7467** | 0.0489 |
| final_3lv_merge | base_rerank2.0 | 0.9378 | **0.9467** | 1.0000 | 0.9733 | **0.8667** | 0.0312 |
| final_3lv_merge | nusax_beam4 | 0.8889 | **0.9111** | 1.0000 | 0.9600 | **0.7733** | 0.0444 |
| final_3lv_merge | nusax_rerank2.0 | 0.9333 | **0.9467** | 1.0000 | 0.9600 | **0.8800** | 0.0267 |
| final_3lv_postedit | beam4_pe_trig_gain | 0.8133 | **0.8978** | 0.9867 | 0.9600 | **0.7467** | 0.0489 |
| final_3lv_postedit | rerank2.0_pe_trig_gain | 0.8489 | **0.9467** | 1.0000 | 0.9733 | **0.8667** | 0.0312 |
| final_3lv_selftrain | st_train_beam4 | 0.9156 | **0.9289** | 1.0000 | 0.9600 | **0.8267** | 0.0267 |
| final_3lv_selftrain | st_train_nusax_beam4 | 0.9067 | **0.9244** | 1.0000 | 0.9467 | **0.8267** | 0.0356 |
| final_3lv_selftrain | st_train_nusax_rerank0.5 | 0.9422 | **0.9600** | 1.0000 | 0.9600 | **0.9200** | 0.0177 |
| final_3lv_selftrain | st_train_rerank0.5 | 0.9422 | **0.9511** | 1.0000 | 0.9733 | **0.8800** | 0.0267 |
| final_3lv_selftrain | teacher_beam4 | 0.9067 | **0.8978** | 0.9867 | 0.9600 | **0.7467** | 0.0489 |
| final_3lv_selftrain | teacher_rerank4.0 | 0.9467 | **0.9467** | 1.0000 | 0.9733 | **0.8667** | 0.0356 |
| final_3lv_v2 | m2m_base_beam4 | 0.8578 | **0.8978** | 0.9867 | 0.9600 | **0.7467** | 0.0489 |
| final_3lv_v2 | m2m_base_rerank2.0 | 0.9156 | **0.9467** | 1.0000 | 0.9733 | **0.8667** | 0.0312 |
| final_3lv_v2 | m2m_na_beam4 | 0.8844 | **0.9200** | 1.0000 | 0.9600 | **0.8000** | 0.0577 |
| final_3lv_v2 | m2m_na_ls_beam4 | 0.8578 | **0.9111** | 1.0000 | 0.9733 | **0.7600** | 0.0444 |
| final_3lv_v2 | m2m_na_ls_rerank1.0 | 0.9111 | **0.9644** | 1.0000 | 0.9733 | **0.9200** | 0.0400 |
| final_3lv_v2 | m2m_na_rerank0.5 | 0.9156 | **0.9511** | 1.0000 | 0.9600 | **0.8933** | 0.0355 |
| final_3lv_v2 | nllb_na_beam4 | 0.8044 | **0.8844** | 1.0000 | 0.9333 | **0.7200** | 0.0223 |
| final_3lv_v2 | nllb_na_rerank2.0 | 0.8444 | **0.9156** | 1.0000 | 0.9600 | **0.7867** | 0.0400 |

`final_3lv_cendol/output_run1_bug` adalah run yang rusak (chrF++ ~15); disertakan hanya sebagai pembanding.
