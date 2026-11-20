# Referensi: penelitian terkait

Daftar penelitian yang mirip dengan proyek ini (MT Indonesia → Jawa dengan kontrol tingkat tutur), metode yang
dipakai, dan angka sebelum/sesudah metode. Kunci sitasi BibTeX ada di [`referensi.bib`](referensi.bib).

**Status verifikasi**
- ✅ = angka diambil langsung dari tabel di full text paper.
- ⚠️ = baru dari abstrak atau halaman web; baca full text sebelum mengutip angkanya.

Disusun November 2026. Pencarian literatur belum exhaustive.

## 1. Ringkasan: posisi proyek ini

| Aspek | Riset Jawa sebelumnya | Riset formality bahasa lain | Proyek ini |
|---|---|---|---|
| Arah | Jawa↔Jawa (Wibawa), Indonesia→Jawa tanpa kontrol (Prayoga), LLM zero-shot (Farhansyah) | Inggris→Jerman/Jepang/Korea/dll. | Indonesia→Jawa |
| Kontrol tingkat di input | Tidak ada (kecuali nama tingkat di prompt LLM) | Ada (tag, rerank) | Ada: tag + rerank |
| Jumlah tingkat | 3–4 | Umumnya 2; Jepang 3 | 3 (versi lama 4) |
| Ketepatan tingkat diukur | Tidak (MT) | Ya (formality accuracy) | Ya, dengan juri terpisah dari reranker |

Celah yang diisi: belum ditemukan riset yang melakukan fine-tuning model MT dengan kontrol tingkat tutur Jawa di input
dan mengukur ketepatan tingkatnya. Teknik-tekniknya sendiri **bukan baru**; yang baru adalah penerapan dan evaluasinya
pada tingkat tutur Jawa.

## 2. Penelitian bahasa Jawa

### Prayoga dkk. (2026), JAIC ✅ `prayoga2026evaluating`
- **Paling mirip**: dataset Unggah-Ungguh, model NLLB-200 600M dan M2M100 418M, tingkat Ngoko/Krama/Krama Alus.
- Setup: 3.434 pasang untuk training (split 90:10 train/valid), 3 epoch, AdamW lr 2e-4, batch 4, max 128 token.
  Test 1.500 kalimat (500 per tingkat), dipilih tanpa split berbasis group.
- Tidak ada kontrol tingkat di input dan tidak ada pengukuran ketepatan tingkat.

| Model | BLEU | chrF++ | Per tingkat (BLEU) |
|---|---|---|---|
| NLLB-200 zero-shot | 6,89 | 27,72 | Ngoko 13,54 · Krama 2,84 · Krama Alus 3,02 |
| M2M100 zero-shot | 2,98 | 21,14 | Ngoko 3,71 · Krama 2,36 · Krama Alus 2,78 |
| NLLB-200 fine-tuned | 6,89 | 27,72 | **identik dengan zero-shot** (lihat catatan) |
| M2M100 fine-tuned | **32,30** | **56,39** | Ngoko 22,24 · Krama 40,86 · Krama Alus 33,38 |
| Gemini (prompt, nama tingkat disebut) | 37,49 | 67,08 | Ngoko 42,95 · Krama 24,92 · Krama Alus 42,85 |

Catatan:
- Angka NLLB "fine-tuned" (Tabel VI/VII) **sama persis hingga 6 desimal** dengan zero-shot (Tabel II/III).
  Kemungkinan salah tabel atau model fine-tuned tidak tersimpan.
- Test set dipilih acak dari dataset yang kalimat Indonesianya berulang di beberapa tingkat, jadi berisiko bocor.
  Pada data ini, random split 80/20 membuat 95% kalimat test juga ada di train (lihat `IDE_MT_Jawa_Tingkat_Tutur.md`).
- chrF++ M2M100 mereka (56,39) **tidak bisa dibandingkan langsung** dengan proyek ini (58,62 pada 176 kalimat test
  dengan split berbasis group).

### Farhansyah dkk. (2025), ACL ✅ `farhansyah2025javanese`
- Sumber dataset **Unggah-Ungguh** (4.024 baris translation, 4 tingkat; sebagian dari buku yang di-OCR lalu
  dikoreksi penutur asli). Lisensi CC BY-NC 4.0.
- Tidak ada fine-tuning model MT. Yang ada: klasifikasi tingkat, style change, terjemahan dengan LLM zero-shot.
- Klasifikasi tingkat (Tabel 5): dictionary-based 88,37% → LSTM 93,47% → **DistilBERT fine-tuned 95,65%**.
  Classifier DistilBERT inilah yang ditiru proyek ini sebagai juri.
- LLM zero-shot Indonesia → Jawa (Tabel 10 BLEU / Tabel 22 chrF++):

| Model | Ngoko | Ngoko Alus | Krama | Krama Alus |
|---|---|---|---|---|
| GPT-4o | 27,81 / 54,14 | 14,52 / 36,99 | 14,31 / 42,59 | 15,56 / 46,16 |
| Gemini-1.5-pro | 16,76 / 47,49 | 6,99 / 32,33 | 14,86 / 45,55 | 7,05 / 31,66 |
| SahabatAI (Gemma2 9B) | 6,27 / 31,02 | 3,49 / 25,68 | 1,06 / 21,47 | 1,59 / 23,26 |

- Temuan: LLM bias ke Ngoko; Ngoko Alus paling sulit; Indonesia → Jawa lebih sulit daripada Jawa → Indonesia.

### Yuliawati, Alfina & Budi (2025), IALP ⚠️ `yuliawati2025ngokokrama`
- Korpus **NgokoKrama**: 1.000 pasang Indonesia–Krama. Fine-tuning IndoBART-v2, Indo-T5, NLLB-200; NLLB terbaik
  di dua arah. Metrik hanya SacreBLEU. Angka belum didapat (hanya abstrak).
- Bedanya dengan proyek ini: hanya satu tingkat (Krama).

### Wibawa, Nafalski & Mahmudy (2014), ECTI-CIT ✅ `wibawa2014augmented`
- **Pra-neural**. Hybrid corpus-based MT (memory-based + fitur statistik), alignment dengan *edit shifting distance*
  dan aturan *impossible pair*, pemilihan pasangan dengan Dice coefficient.
- **Arah terjemahan Jawa → Jawa antar tingkat**, bukan Indonesia → Jawa.
- **Data**: korpus paralel buatan penulis dari literatur tata bahasa Jawa (bukan dataset publik), 126 kalimat per
  tingkat × 4 tingkat (7.168 kata). Eksperimen tambahan: train 504/824/1.144 kalimat, test 504 kalimat.
- **Metrik**: A = % terjemahan benar secara leksikal *dan* pragmatis; Q = skor bertingkat 100/75/50/25/0.
- Hasil: Ngoko → Krama Alus terendah (A = 69,0%); Krama → Krama Alus 96,8%. Rata-rata A 82,7% → 87,9% saat data
  latih ditambah; 93,8% setelah aturan impossible pair (abstrak).
- Catatan: baris "akurasi rata-rata 72,3%" di `IDE_MT_Jawa_Tingkat_Tutur.md` tidak cocok dengan paper ini.

### Primandhika dkk. (2020), CONAPLIN, bahasa Sunda ⚠️ `primandhika2020sundanese`
- Transformer Indonesia–Sunda, 50.000 pasang, BLEU 42,72%. Output didominasi *basa loma* (56%), pola bias ke tingkat
  rendah yang mirip dengan Jawa.

## 3. Teknik kontrol tingkat/formality (sumber metode proyek ini)

### Tag di source (side constraint)

**Sennrich, Haddow & Birch (2016), NAACL** ✅ `sennrich2016politeness`
- Asal-usul teknik prefix tag. Inggris → Jerman, 2 tingkat (du/Sie), label otomatis dari kata ganti.
- Tabel 2 (2.000 kalimat OpenSubtitles yang mengandung *you*):

| Tag | Output sopan | Output informal | BLEU |
|---|---|---|---|
| Tanpa tag | 351 | 1.471 | 20,7 |
| `polite` | **1.728** | 64 | 17,9 |
| `informal` | 28 | **1.831** | 20,2 |
| Oracle (sesuai referensi) | 567 | 1.272 | **23,9 (+3,2)** |

**Feely, Hasler & de Gispert (2019), WAT** ✅ `feely2019japanese`
- **Analogi terdekat**: Inggris → Jepang, **3 tingkat** (informal, polite, formal), tag di source, label tingkat dari
  heuristik bentuk kata kerja (classifier F1 0,83–0,86).
- Data latih sangat timpang: tingkat formal hanya 0,7–1,6%.

| Dataset | BLEU keseluruhan | BLEU bagian formal | F1 formal | F1 polite |
|---|---|---|---|---|
| ASPEC | 43,0 → 43,3 | 33,0 → 43,9 | 0,19 → 0,96 | 0,67 → 1,00 |
| JESC | 18,8 → 20,3 | 22,1 → 29,3 | 0,28 → 0,96 | 0,50 → 0,99 |
| KFTT | 24,6 → 25,5 | 11,4 → 19,3 | 0,00 → 0,74 | 0,41 → 0,96 |
| Proprietary | 24,6 → 25,5 | 17,4 → 22,3 | 0,29 → 0,91 | 0,82 → 0,99 |

- Konversi berbasis aturan (post-processing) kalah dari NMT dengan tag di semua dataset.

### Rerank kandidat dengan skor/classifier formality

**Niu, Martindale & Carpuat (2017), EMNLP** ✅ `niu2017formality`
- Rerank n-best (1.000 kandidat) phrase-based MT Prancis → Inggris dengan skor formality leksikal.
- BLEU: +0,5 (informal), +0,3 (netral), tetap (formal, 47,97).

**Wu dkk. / HW-TSC (2023), IWSLT** ✅ `wu2023hwtsc`
- *Reranking-based transductive learning* (RTL): pilih kandidat yang sesuai tingkat dari 100 hasil beam dengan
  classifier formality, lalu pakai sebagai data latih tambahan.
- Inggris → Korea, tingkat formal (Tabel 4):

| Tahap | M-Acc | C-F | BLEU |
|---|---|---|---|
| Pre-training | 100% | 65,33% | 5,5 |
| + Domain adaptation (fine-tuning berlabel) | 100% | 97,49% | 24,5 |
| + RTL | 100% | 97,65% | 25,8 |
| + Iterative RTL | 100% | 99,83% | 25,0 |

### Augmentasi data dan post-editing

**Zhang dkk. / Amazon Alexa AI (2022), IWSLT** ✅ `zhang2022amazon`
- Fine-tuning mBART: akurasi formal Inggris → Jepang 0,585 → 0,939; BLEU 19,4 → 27,8.
- Augmentasi weakly-labelled: Inggris → Jepang naik (0,939 → 0,962), Inggris → Hindi turun (BLEU formal 33,7 → 29,6).
- Post-editing berbasis aturan: akurasi informal Inggris → Hindi (Transformer) 0,844 → 0,978, BLEU tetap.

**Lee dkk. (2023), arXiv 2306.14514** ✅ `lee2023datadriven`
- Augmentasi sintetis ChatGPT **merugikan**: Inggris → Korea formal BLEU 26,60 → 17,09 (akurasi 87,0 → 79,4%);
  Inggris → Vietnam formal BLEU 47,00 → 41,57.

### Benchmark dan metrik

**Nădejde dkk. (2022), CoCoA-MT, NAACL Findings** ⚠️ `nadejde2022cocoa`
- Benchmark kontrastif formal/informal untuk 6 bahasa; sumber konsep *formality accuracy*.

## 4. Model dan benchmark bahasa Indonesia/daerah
- **Cahyawijaya dkk. (2024), Cendol, ACL** ⚠️ `cahyawijaya2024cendol`: LLM instruction-tuned 560M–13B untuk bahasa
  Indonesia/daerah, Apache-2.0. Varian `cendol-mt5-base-inst` diuji di `kaggle/final_3lv_cendol*`.
- **Cahyawijaya dkk. (2021), IndoNLG, EMNLP** ⚠️ `cahyawijaya2021indonlg`: benchmark MT Jawa↔Indonesia (IndoBART),
  tanpa tingkat tutur.
- **Susanto dkk. (2023), arXiv 2311.00998** ⚠️ `susanto2023replicable`: benchmark NMT Jawa, Sunda, Minang, Bali,
  tanpa tingkat tutur.
- **NusaX (Winata dkk., 2023)** ⚠️ `winata2023nusax`: sumber data tambahan di `kaggle/final_3lv_merge` (CC BY-SA).

## 5. Perbandingan pola hasil dengan proyek ini

| Metode | Proyek ini | Paper |
|---|---|---|
| Tag + fine-tuning | Ketepatan tingkat 0,86 (Krama Alus 0,61) | Feely: F1 formal 0,00–0,29 → 0,74–0,96 |
| + Rerank classifier | 0,86 → 0,92; Krama Alus 0,61 → 0,77; chrF++ +0,49 | HW-TSC: C-F +0,2–2%; Niu: BLEU +0,3–0,5 |
| Augmentasi sintetis | Krama Alus 0,85 → 0,79 (turun) | ChatGPT: BLEU −9,5; Amazon: turun di Hindi |

Pola yang sama di semua paper: fine-tuning dengan label tingkat memberi lompatan terbesar, rerank memberi tambahan
kecil tetapi konsisten, dan augmentasi sintetis bersifat untung-untungan. Rerank proyek ini memberi efek lebih besar
karena baseline kontrolnya masih jauh dari jenuh (61% di Krama Alus, sedangkan HW-TSC sudah 97% sebelum rerank).

## 6. Penelitian dengan lebih dari dua tingkat

Hampir semua riset formality control bersifat biner (formal/informal). Yang multi-tingkat:

| Paper | Bahasa | Tingkat | Metode |
|---|---|---|---|
| Wibawa dkk. 2014 | Jawa → Jawa | 4 | Corpus-based, tabel padanan per tingkat |
| Farhansyah dkk. 2025 | Indonesia ↔ Jawa | 4 | LLM zero-shot, classifier 4 kelas |
| Prayoga dkk. 2026 | Indonesia → Jawa | 3 | Fine-tuning tanpa pemilih tingkat; Gemini dengan prompt |
| Feely dkk. 2019 | Inggris → Jepang | 3 | Tag di source |
| Ohtake dkk. 2004 ⚠️ | Jepang → Korea | 6 | MT berbasis aturan transfer |
| Proyek ini | Indonesia → Jawa | 3 (lama 4) | Tag + init embedding + oversampling + rerank 3 kelas |

## 7. Tautan
- Prayoga dkk. 2026: https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/12326
- Farhansyah dkk. 2025: https://aclanthology.org/2025.acl-long.1296/ · dataset https://github.com/JavaneseHonorifics/Unggah-Ungguh
- Yuliawati dkk. 2025: https://dara.ui.ac.id/research-output/122df25b-37aa-45d2-91dd-5d171660daed
- Wibawa dkk. 2014: https://ph01.tci-thaijo.org/index.php/ecticit/article/view/54387
- Primandhika dkk. 2020: https://www.atlantis-press.com/proceedings/conaplin-20/125956026
- Sennrich dkk. 2016: https://aclanthology.org/N16-1005/
- Feely dkk. 2019: https://aclanthology.org/D19-5203/
- Niu dkk. 2017: https://aclanthology.org/D17-1299/
- HW-TSC 2023: https://aclanthology.org/2023.iwslt-1.13
- Amazon Alexa AI 2022: https://aclanthology.org/2022.iwslt-1.32
- Lee dkk. 2023: https://arxiv.org/abs/2306.14514
- CoCoA-MT: https://aclanthology.org/2022.findings-naacl.47
- Cendol: https://aclanthology.org/2024.acl-long.796
- IndoNLG: https://arxiv.org/abs/2104.08200
- Susanto dkk. 2023: https://arxiv.org/abs/2311.00998
- Ohtake dkk. 2004: https://www2.nict.go.jp/direct/members/ohtake/publishing/CICLing2004.pdf
- Context-aware NMT honorific Korea (MDPI Electronics 2021): https://mdpi.com/2079-9292/10/13/1589
- LoraxBench (Krama sebagai register): https://arxiv.org/abs/2508.12459
