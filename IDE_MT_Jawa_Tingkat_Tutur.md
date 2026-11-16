# Ide: MT Indonesia → Jawa dengan Kontrol Tingkat Tutur

Status: **topik dipilih (Jawa tingkat tutur), belum implementasi.** Topik lama (MT Jerman → Inggris, EMEA) ditolak. PPT tidak dikerjakan dulu. Dosen belum memutuskan soal deployment.

## Ringkasan
Satu model machine translation Indonesia → Jawa yang tingkat tuturnya bisa dipilih pengguna (Ngoko, Ngoko Alus, Krama, Krama Alus). Tingkat dikirim sebagai **source prefix tag** di input.

```
Pengguna : "aku makan nasi" + pilih [Krama Alus]
Aplikasi : "<krama_alus> aku makan nasi"  ->  model
Output   : kalimat Jawa Krama Alus
```

Contoh output (ilustrasi dari pengetahuan umum, **bukan hasil model, perlu dicek penutur Jawa**): Ngoko *Aku mangan sega.* / Krama Alus *Kula nedha sekul.*

## Urgensi

### Inti (paling kuat)
> Pengguna butuh memilih tingkat tutur, tetapi model machine translation open-source belum punya kontrol tingkat tutur. LLM bisa lewat prompt, tetapi tidak reproducible, berbayar, dan akurasi tingkatnya jarang diukur.

### Bukti teknis
| Klaim | Bukti |
|---|---|
| Model open-source gagal di tingkat tinggi | NLLB-200 zero-shot BLEU Ngoko 13,54 vs Krama 2,84 (JAIC 2026) |
| NMT yang ada tidak punya kontrol tingkat | JAIC 2026: NMT "without explicit register control in the input" |
| LLM bias ke Ngoko | GPT-4o ID → Jawa BLEU Ngoko 27,81 vs Krama Alus 15,56; akurasi klasifikasi tingkat 53,5%, F1 Ngoko Alus 0 (ACL 2025) |
| LLM open-source berbahasa Indonesia/Jawa pun lemah | SahabatAI (Gemma2 9B) zero-shot ID → Ngoko BLEU 6,27, ID → Krama 1,06 (ACL 2025) |
| LLM bukan solusi operasional | Gemini di JAIC dijalankan manual lewat interface, diakui hanya sebagai reference |

### Bukti sosial (pendukung)
- Penguasaan Krama di kalangan muda menurun. Survei 127 siswa SMP: 41,7% lancar Ngoko, hanya **4,7% lancar Krama**. Studi Purworejo: hanya 8% anak masih memakai Krama (Dewi, 2017, dikutip jurnal Kandai). Sampel kecil dan tidak acak, jadi angka bersifat indikatif.
- Salah tingkat tutur dapat dianggap tidak sopan (ethical consideration, ACL 2025).
- Wibawa dkk. (2016) sudah memotivasi MT sebagai cara melestarikan tingkat tutur, karena sebagian guru pun memakai tingkat yang tidak tepat.
- Jawa dituturkan lebih dari 98 juta orang (ACL 2025).

## Riset terdahulu

| Tahun | Riset | Yang dikerjakan | Keterbatasan |
|---|---|---|---|
| 2013-2016 | Wibawa dkk. (ECTI-CIT 2016, IAPGOS 2016) | MT Jawa dengan tingkat tutur, hybrid corpus-based / statistical memory-based | Pra-neural; akurasi rata-rata 72,3% (paper pendamping, belum dicek) |
| 2016 / 2022 | Sennrich dkk. 2016; CoCoA-MT 2022; IWSLT 2022 formality track | Kontrol formality/politeness dengan side constraint atau source prefix; metrik formality accuracy | Bahasa Eropa/Asia lain, bukan Jawa; hanya 2 tingkat (formal/informal) |
| 2023 | Susanto dkk. (arXiv 2311.00998) | Benchmark NMT Jawa, Sunda, Minang, Bali | Tanpa tingkat tutur |
| 2025 | Farhansyah dkk. (ACL 2025), Unggah-Ungguh | Dataset 4 tingkat; classifier; LLM zero-shot | Tidak fine-tune model MT |
| 2025 | Yuliawati dkk. (IALP 2025), NgokoKrama | 1.000 pasang ID–Krama; fine-tune IndoBART-v2, Indo-T5, NLLB-200; NLLB terbaik | Hanya Krama; hanya SacreBLEU (baru dibaca abstraknya) |
| 2026 | Prayoga dkk. (JAIC 2026) | NLLB-200 600M dan M2M100 418M, zero-shot + fine-tuning pada Unggah-Ungguh; Gemini via prompt | Tanpa kontrol tingkat di input; tanpa Ngoko Alus; skor NLLB fine-tuned identik dengan zero-shot (anomali); angka data tidak sinkron (3.434 train + 1.500 test vs total ±4.000) |

### Posisi project ini (pembeda)
1. **Kontrol eksplisit** lewat source prefix tag untuk **empat** tingkat, termasuk Ngoko Alus (dilewati JAIC).
2. **Level accuracy** sebagai metrik utama selain BLEU/chrF++, mengikuti praktik formality accuracy di CoCoA-MT, plus confusion matrix tingkat diminta vs dihasilkan.
3. **Group-based split** per kolom `group` agar kalimat Indonesia yang sama tidak bocor dari train ke test.
4. **Ablation dengan vs tanpa tag** pada model dan split yang sama.
5. **Factorized tag** (sumbu dasar + sumbu hormat) dibanding 4 atomic tag, berdasarkan temuan EDA bahwa tingkat tutur terbentuk dari dua sumbu.

Teknik tag sendiri bukan hal baru; yang baru adalah penerapan dan evaluasinya pada sistem empat tingkat Jawa. Pencarian literatur belum exhaustive: bisa ada riset lain yang belum ditemukan.

## Dataset: Unggah-Ungguh
- Paper: Farhansyah dkk., ACL 2025 (arXiv 2502.20864). Hugging Face: `JavaneseHonorifics/Unggah-Ungguh`. Lisensi **CC BY-NC 4.0** (riset saja).
- File lokal: `data/unggah_ungguh/translation.csv`, `conversation.csv`.

| Hal | Temuan (diperiksa langsung) |
|---|---|
| Baris translation | **2.009** (paper menyebut 4.024; JAIC menyebut ±4.000; selisih belum jelas) |
| Kolom | `index`, `label`, `javanese sentence`, `group`, `indonesian sentence`, `english sentence` |
| Label | 0 Ngoko 742, 1 Ngoko Alus 263, 2 Krama 740, 3 Krama Alus 264 |
| `group` | 746 kelompok makna, 2-4 tingkat per kelompok; kalimat Indonesia identik di semua tingkat |
| Panjang | rata-rata 7,6 kata, maksimum 16 |
| Conversation | 80 skenario |
| Domain | umum (kamus dan buku unggah-ungguh), hampir tanpa kesehatan |

## Split data (sudah dibuat)
Script: `scripts/split_data.py` (StratifiedGroupKFold 10 fold, seed 42: fold 0 = test, fold 1 = valid, sisanya train). Output: `data/unggah_ungguh/split/{train,valid,test}.csv`, kolom `index, group, label, level, base, honor, indonesian, javanese, english`. Normalisasi ringan (spasi, kutip miring); typo tidak diubah.

| Split | Rows | Groups | Ngoko | Ngoko Alus | Krama | Krama Alus |
|---|---|---|---|---|---|---|
| train | 1.605 | 596 | 593 | 210 | 592 | 210 |
| valid | 202 | 75 | 74 | 27 | 74 | 27 |
| test | 202 | 75 | 75 | 26 | 74 | 27 |

Overlap group dan kalimat Indonesia antar split: 0.

## Evaluasi 3 tingkat (setara setup JAIC)
JAIC hanya memakai Ngoko, Krama, Krama Alus (Ngoko Alus tidak dievaluasi). Dataset Unggah-Ungguh sendiri berlabel 4 tingkat. Angka di bawah dihitung ulang dari output ronde 3 (tanpa retraining), hanya pada 176 baris test berlevel Ngoko/Krama/Krama Alus; control accuracy dari permintaan 3 tingkat tersebut. Mean ± std 3 seed.

| Model | Konfigurasi | chrF++ | BLEU | Control acc (3 tingkat) |
|---|---|---|---|---|
| NLLB-600M | base | 54,93 ± 0,74 | 28,40 ± 0,61 | 0,631 ± 0,004 |
| NLLB-600M | +init+os+mt | **58,67 ± 0,25** | **32,28 ± 0,30** | 0,824 ± 0,022 |
| M2M100-418M | base | 57,29 ± 0,19 | 30,87 ± 0,45 | 0,781 ± 0,024 |
| M2M100-418M | +init+os | 57,98 ± 0,23 | 31,63 ± 0,43 | **0,839 ± 0,007** |

Pembanding JAIC (M2M100 fine-tuned, 3 tingkat, test 1.500 kalimat, kemungkinan leakage): BLEU 32,30, chrF++ 56,39. Test set berbeda, jadi hanya pembanding kasar.

### Model yang dilatih khusus 3 tingkat
Baris Ngoko Alus dibuang dari train/valid/test (1.395 / 175 / 176), 3 tag, classifier 3 kelas, multi-task hanya Krama↔Krama Alus (412 pasangan), oversampling hanya Krama Alus. Script `kaggle/finetune_3lv/finetune_3lv.ipynb`; output `kaggle/finetune_3lv_*/output/`. Test 176 baris sama persis dengan subset di tabel atas, jadi chrF++/BLEU bisa dibandingkan langsung (control accuracy tidak, karena classifier berbeda).

| Model | Dilatih pada | Konfigurasi | chrF++ | BLEU | Control acc | Krama Alus | Identik krama=krama_alus |
|---|---|---|---|---|---|---|---|
| NLLB-600M | 3 tingkat | base | 56,91 ± 0,41 | 30,23 ± 0,32 | 0,80 | 0,60 | 16% |
| NLLB-600M | 3 tingkat | +init+os+mt | 56,92 ± 0,78 | 30,36 ± 1,11 | 0,76 | 0,60 | 27% |
| NLLB-600M | **4 tingkat** | +init+os+mt | **58,67 ± 0,25** | **32,28 ± 0,30** | (0,82)* | 0,64 | 10% |
| M2M100-418M | 3 tingkat | base | 55,42 ± 1,17 | 29,76 ± 0,99 | 0,69 | 0,18 | 64% |
| M2M100-418M | 3 tingkat | +init+os | 57,82 ± 0,44 | 31,22 ± 0,69 | 0,85 | 0,60 | 8% |
| M2M100-418M | **4 tingkat** | +init+os | 57,98 ± 0,23 | 31,63 ± 0,43 | (0,84)* | 0,58 | 5% |

\* classifier 4 kelas, tidak sebanding langsung dengan classifier 3 kelas.

Temuan:
1. **Melatih dengan 4 tingkat tidak merugikan tugas 3 tingkat, malah sama atau lebih baik.** NLLB: 58,67 vs 56,92 chrF++ (selisih > variasi seed). M2M100: 57,98 vs 57,82 (setara). Dugaan: Ngoko Alus berbagi kosakata hormat dengan Krama Alus, sehingga sumbu hormat lebih sering terlihat saat training.
2. Pada versi 3 tingkat, efek teknik berbeda per model: NLLB base sudah baik dan teknik tidak menambah; M2M100 base tidak stabil (Krama Alus 0,12-0,29, identik 36-80% antar seed) dan +init+os memperbaikinya besar (Krama Alus 0,60, identik 8%).
3. Hasil terbaik untuk setup 3 tingkat ala JAIC tetap model 4 tingkat NLLB +init+os+mt (chrF++ 58,67, BLEU 32,28), setara BLEU JAIC (32,30) tetapi pada split tanpa leakage dan dengan kontrol tingkat.

## Demo website (Kaggle + Gradio)
- Model demo: M2M100-418M +init+os, seed 42, dilatih ulang di kernel `mdevian/unggah-ungguh-final-m2m100` (weights fp16 di output kernel).
- Aplikasi: `demo/app.py` (Gradio: input Indonesia, pilihan 4 tingkat, tombol "Bandingkan 4 tingkat", catatan evaluasi dan keterbatasan).
- Hosting: kernel CPU private `mdevian/unggah-ungguh-demo-gradio` (`kaggle/demo/`) menjalankan app dengan `share=True`; link publik `*.gradio.live` dikirim ke topik ntfy di `kaggle/demo/ntfy_topic.txt`. Demo hidup selama kernel berjalan (maks ±11,5 jam).
- Hugging Face Spaces (Gradio) tidak dipakai: hosting Gradio di cpu-basic sekarang butuh langganan PRO. Login HF (akun mdevian1) sudah ada bila nanti ingin pindah.
- Menjalankan ulang (mis. sebelum presentasi):
  ```bash
  cd kaggle/demo && kaggle kernels push -p .
  curl -s "https://ntfy.sh/$(cat ntfy_topic.txt)/json?poll=1"
  ```
  Link muncul di field `message` sekitar 2-3 menit setelah push. Satu terjemahan ±2,5 detik, empat tingkat ±3,4 detik (CPU Kaggle).
- Contoh output nyata: "Bapak sudah pulang dari kantor." → Ngoko *Bapak wis mulih saka kantor.* / Ngoko Alus *Bapak wis kondur saka kantor.* / Krama *Bapak sampun mantuk saking kantor.* / Krama Alus *Bapak sampun kondur saking kantor.*

## Ronde 3: konfirmasi 3 seed (hasil utama)
Seed 42, 13, 2024; setup sama dengan ronde 2. Script `kaggle/finetune_r3/finetune_r3.ipynb`; output `kaggle/finetune_r3_*/output/` (`summary_*.json` berisi bagian `aggregate`). Nilai = mean ± std (3 seed).

| Model | Konfigurasi | chrF++ all | BLEU all | Control acc | Ngoko Alus | Krama Alus | Identik krama=krama_alus |
|---|---|---|---|---|---|---|---|
| NLLB-600M | base (atomic) | 54,73 ± 0,54 | 28,38 ± 0,59 | 0,584 ± 0,002 | 0,44 ± 0,02 | 0,20 ± 0,03 | 77,3% ± 4,8 |
| NLLB-600M | **+init+os+mt** | **58,68 ± 0,22** | **32,28 ± 0,25** | 0,768 ± 0,016 | 0,60 ± 0,04 | **0,64 ± 0,03** | 9,8% ± 1,5 |
| M2M100-418M | base (atomic) | 56,65 ± 0,56 | 30,19 ± 0,16 | 0,702 ± 0,036 | 0,47 ± 0,07 | 0,44 ± 0,04 | 19,1% ± 3,1 |
| M2M100-418M | **+init+os** | 58,14 ± 0,11 | 31,67 ± 0,34 | **0,787 ± 0,007** | **0,63 ± 0,01** | 0,58 ± 0,03 | **4,9% ± 2,0** |

chrF++ per tingkat (mean): NLLB terbaik Ngoko 56,40 / Ngoko Alus 58,70 / Krama 58,98 / Krama Alus 62,81; M2M100 terbaik 53,59 / 59,21 / 60,16 / 61,96.

Temuan:
1. **Perbaikan konsisten di ketiga seed**: rentang base dan konfigurasi terbaik tidak bersinggungan (NLLB chrF++ base maks 55,30 < terbaik min 58,55; M2M100 base maks 57,29 < terbaik min 58,04).
2. NLLB: chrF++ +3,95, control +0,18, Krama Alus 0,20 → 0,64, output krama = krama_alus 77% → 10%.
3. M2M100: chrF++ +1,49, control +0,09, Krama Alus 0,44 → 0,58, identik 19% → 5%.
4. Variansi antar seed kecil (std chrF++ 0,1-0,6). Seed 42 mereproduksi skor ronde 2 persis (chrF++ sama); control accuracy sedikit berbeda karena classifier dilatih ulang tiap kernel.
5. Dua konfigurasi terbaik hampir setara: NLLB unggul di chrF++/BLEU dan Krama Alus; M2M100 unggul di control acc keseluruhan, lebih kecil (418M), dan output tingkatnya paling jarang identik. Untuk deployment, M2M100 lebih ringan.

Keterbatasan: classifier evaluator (akurasi per kelas alus 0,70-0,81 pada referensi) membatasi control accuracy alus; belum ada manual check penutur Jawa; test 202 baris.

## Fine-tuning ronde 2 (sudah dijalankan)
Tujuan: memperbaiki sumbu hormat. Semua varian 4 atomic tag, 10 epoch, lr 5e-5, seed 42. Varian kumulatif: `base` → `+init` (embedding tag dari rata-rata embedding kata penanda) → `+os` (oversampling contoh alus x2, 1.605 → 2.025 pasangan) → `+mt` (multi-task 826 pasangan JV→JV sumbu hormat dari train split). Script: `kaggle/finetune_r2/finetune_r2.ipynb`; output `kaggle/finetune_r2_*/output/`.

| Model | Varian | Test chrF++ all (CI95) | BLEU | Control acc | Ngoko Alus | Krama Alus | Identik krama=krama_alus |
|---|---|---|---|---|---|---|---|
| NLLB-600M | base | 54,22 (52,0-56,4) | 27,71 | 0,57 | 0,47 | 0,21 | 79% |
| NLLB-600M | +init | 55,88 (53,6-58,2) | 28,61 | 0,60 | 0,44 | 0,25 | 47% |
| NLLB-600M | +init+os | 56,62 (54,3-59,0) | 29,72 | 0,68 | 0,56 | 0,44 | 15% |
| NLLB-600M | **+init+os+mt** | **58,55** (56,2-60,9) | **32,50** | 0,73 | **0,63** | **0,53** | 11% |
| M2M100-418M | base | 56,22 (53,9-58,5) | 30,27 | 0,66 | 0,40 | 0,31 | 17% |
| M2M100-418M | +init | 57,08 (54,9-59,3) | 31,76 | 0,72 | 0,52 | 0,43 | 4% |
| M2M100-418M | **+init+os** | **58,12** (55,7-60,5) | 31,80 | **0,75** | 0,60 | 0,44 | 5% |
| M2M100-418M | +init+os+mt | 57,01 (54,8-59,4) | 29,99 | 0,75 | 0,56 | 0,51 | 7% |

Akurasi classifier evaluator per kelas pada referensi test: NLLB-run Ngoko 0,95 / Ngoko Alus 0,81 / Krama 0,96 / Krama Alus 0,81; M2M-run 0,95 / 0,73 / 0,99 / 0,70. Jadi control accuracy alus dibatasi (under-estimate) oleh classifier.

Temuan:
1. **Ketiga teknik membantu sumbu hormat.** NLLB: Krama Alus 0,21 → 0,53; output krama identik dengan krama_alus 79% → 11%; chrF++ +4,3. M2M100: Krama Alus 0,31 → 0,44-0,51; identik 17% → 5%.
2. Kontribusi per teknik berbeda per model: pada NLLB setiap tahap menambah (mt paling besar untuk Krama Alus); pada M2M100 init+os sudah cukup, mt tidak menambah chrF++.
3. **Sisa kelemahan**: permintaan Krama Alus masih dinilai Krama pada 31-40 dari 75 kalimat; Ngoko Alus → Ngoko 22-25 dari 75.
4. **Variansi antar run besar**: M2M100 atomic ronde 1 (8 epoch) identik 61%, ronde 2 base (10 epoch) 17%. Klaim per teknik perlu beberapa seed.
5. Konfigurasi terbaik: NLLB +init+os+mt (chrF++ 58,55, BLEU 32,50) dan M2M100 +init+os (chrF++ 58,12, control 0,75); CI keduanya overlap.

## Fine-tuning ronde 1 (sudah dijalankan)
Kaggle T4, kernel private `mdevian/unggah-ungguh-finetune-nllb` dan `mdevian/unggah-ungguh-finetune-m2m100`. Script: `kaggle/finetune/finetune.ipynb`; output di `kaggle/finetune_*/output/`. Full fine-tuning, 8 epoch, batch 16, lr 5e-5, warmup 10%, seed 42 (satu seed), epoch terbaik dipilih dari valid chrF++. Classifier evaluator dilatih ulang di train split (`w11wo/javanese-distilbert-small`, akurasi pada referensi test 91-92%).

Test chrF++ (BLEU) dan control accuracy (tiap kalimat Indonesia test diminta ke 4 tingkat, 300 output, dinilai classifier):

| Model | Varian | Ngoko | Ngoko Alus | Krama | Krama Alus | All (CI95) | Control acc |
|---|---|---|---|---|---|---|---|
| NLLB-600M | zero-shot | 38,01 | 29,09 | 21,37 | 19,91 | 27,84 | - |
| NLLB-600M | no_tag | 38,08 | 40,20 | 45,93 | 42,85 | 42,05 (39,4-44,8) | 0,25 |
| NLLB-600M | **atomic** | **56,39** | **51,76** | 55,14 | 49,74 | **54,36** (52,2-56,7) | 0,56 |
| NLLB-600M | factorized | 54,42 | 47,08 | 53,82 | 49,67 | 52,59 (50,4-54,8) | 0,49 |
| M2M100-418M | zero-shot | 20,91 | 21,81 | 17,69 | 19,39 | 19,57 | - |
| M2M100-418M | no_tag | 35,21 | 34,44 | 47,63 | 42,65 | 40,96 (38,6-44,1) | 0,25 |
| M2M100-418M | **atomic** | 53,72 | 49,09 | 57,15 | **50,34** | 54,00 (51,7-56,5) | **0,58** |
| M2M100-418M | factorized | 51,30 | 45,30 | **58,08** | 48,01 | 52,72 (50,4-55,2) | 0,49 |

Control accuracy per tingkat (atomic): NLLB Ngoko 0,93 / Ngoko Alus 0,37 / Krama 0,80 / Krama Alus 0,13; M2M100 0,97 / 0,25 / 0,87 / 0,23.

Temuan:
1. **Fine-tuning bekerja** (tidak ada anomali seperti JAIC): valid chrF++ NLLB 29,6 → 56,3; M2M100 17,0 → 57,3.
2. **Tag terbukti penting**: atomic vs no_tag +12-13 chrF++ (CI tidak overlap); control accuracy 0,25 → 0,56-0,58.
3. **Sumbu dasar (ngoko/krama) terkontrol baik, sumbu hormat (alus) gagal.** Output identik untuk krama vs krama_alus: NLLB atomic 96%, M2M100 atomic 61%. Confusion: permintaan Krama Alus sebagian besar dinilai Krama; Ngoko Alus sering dinilai Ngoko.
4. **Hipotesis factorized tag tidak didukung**: chrF++ sedikit di bawah atomic (CI overlap) dan control accuracy lebih rendah (0,49); tag hormat hampir diabaikan (output ngoko = ngoko_alus 84%, krama = krama_alus 88-91%).
5. NLLB dan M2M100 setara setelah fine-tuning (CI overlap). NLLB konvergen lebih cepat (valid 48,5 setelah epoch 1); M2M100 masih naik di epoch 8.
6. No_tag M2M100 Krama BLEU 23,29 vs JAIC 40,86: konsisten dengan dugaan leakage di JAIC, tetapi setup berbeda (belum bukti).

Keterbatasan: satu seed; classifier evaluator akurasi ~92% sehingga control accuracy per tingkat alus bisa bias (akurasi per kelas classifier belum dihitung); test kecil.

Ide ronde 2 (memperbaiki sumbu hormat): oversampling contoh alus, learning rate lebih besar untuk embedding tag baru atau inisialisasi tag dari kata penanda (mis. panjenenganipun), multi-task pasangan JV→JV (krama → krama_alus), epoch lebih banyak untuk M2M100, beberapa seed.

## Baseline zero-shot (sudah dijalankan)
Kaggle T4, kernel private `mdevian/unggah-ungguh-zero-shot-baseline`, dataset private `mdevian/unggah-ungguh-id-jv-split`. Script dan output: `kaggle/baseline_zeroshot/`. Test split 202 baris (75 kalimat Indonesia unik), beam 4.

chrF++ (BLEU) per tingkat, CI95 bootstrap untuk chrF++:

| Sistem | Ngoko | Ngoko Alus | Krama | Krama Alus | All |
|---|---|---|---|---|---|
| Copy | 25,30 (2,25) | 22,97 (3,45) | 21,62 (1,98) | 21,07 (3,17) | 22,96 (1,74) |
| M2M100-418M | 20,91 (2,66) | 21,81 (4,47) | 17,69 (2,06) | 19,39 (3,36) | 19,57 (2,82) |
| NLLB-600M | **38,01 (12,82)** | 29,09 (9,59) | 21,37 (2,46) | 19,91 (3,42) | **27,84 (7,73)** |

CI95 chrF++ NLLB: Ngoko 34,7-41,7 vs Krama 19,5-23,4 (tidak overlap).

Temuan:
- **NLLB zero-shot hampir selalu menghasilkan Ngoko**: classifier menilai 74 dari 75 output sebagai Ngoko; tidak satu pun output memakai penanda alus. Skor Krama dan Krama Alus NLLB bahkan di bawah copy baseline.
- **M2M100 zero-shot lebih buruk dari copy baseline** dan sering degenerate (pengulangan kata, mis. "ngomong ngomong ngomong ..."), sebagian kata tidak diterjemahkan.
- Angka konsisten dengan JAIC (NLLB Ngoko BLEU 13,54 vs 12,82 di sini; Krama 2,84 vs 2,46; M2M100 overall 2,98 vs 2,82): hasil JAIC untuk zero-shot berhasil direplikasi.
- Classifier DistilBERT JavaneseHonorifics akurasi 98,5% pada referensi test; kemungkinan karena referensi itu ada di data training classifier (leakage). Untuk evaluasi akhir, latih ulang classifier di train split. Mapping LABEL_k = label k diasumsikan (didukung akurasi 98,5%).
- Implikasi model: NLLB titik awal jauh lebih baik (chrF++ 27,8 vs 19,6). Tetap fine-tune keduanya karena JAIC melaporkan M2M100 fine-tuned lebih baik.

## Analisis dataset (EDA)
Script: `analisis/eda_unggah.py` dan `analisis/eda_unggah2.py` (jalankan dari folder project).

| Temuan | Angka | Implikasi |
|---|---|---|
| **Random split bocor parah** | Random split 80/20: 312 dari 328 kalimat Indonesia di test juga ada di train (95%). Group-based split: 0 | Group-based split wajib. Skor riset yang memakai random split (kemungkinan JAIC) bisa inflated |
| Dua struktur group | 481 group hanya Ngoko+Krama; 258 group lengkap 4 tingkat; 7 lainnya | Ngoko Alus dan Krama Alus hampir seluruhnya dari 258 group lengkap |
| Dua sumbu tingkat | Token berubah: Ngoko↔Ngoko Alus 0,44, Krama↔Krama Alus 0,42, Ngoko↔Krama 0,72. Jaccard kosakata Ngoko Alus vs Krama Alus 0,39 (tertinggi) | Tingkat = sumbu dasar (ngoko/krama) x sumbu hormat (biasa/alus). Dasar factorized tag |
| Sumbu seimbang | ngoko 1.005 vs krama 1.004; alus 527 vs biasa 1.482 | Factorized tag menggabungkan 527 contoh alus, bukan 263 per tingkat |
| Penanda kuat berupa pronomina | Ngoko: kowe, aku; Krama: sampeyan (218/218), kula; Ngoko Alus: panjenengane, adalem; Krama Alus: panjenenganipun | Model bisa "curang" hanya mengganti pronomina; evaluasi tingkat perlu lebih dari satu cara |
| Copy baseline rendah | ID→JV BLEU 3,4-4,3, chrF++ 21,9-26,0 | Benar-benar butuh translation, bukan copy |
| Style copy | Ngoko→Ngoko Alus BLEU 29,4; Ngoko→Krama 9,0; Ngoko→Krama Alus 8,0 | Perubahan ke alus relatif kecil; ke krama besar |
| Kalimat pendek | 7,3-8,0 kata; maksimum 43 subword token | max_length 64 cukup; training cepat |
| Tokenizer | Fertility JV: M2M100 2,13, NLLB 1,96 token/kata; unk M2M100 0, NLLB 19 token | Keduanya layak; chrF++ cocok karena morfologi (-ipun, dipun-, -aken) |
| Test set kecil | Group split 80/10/10: 1.613 / 200 / 196 baris; test Ngoko Alus 23, Krama Alus 24 | Pakai 5-fold GroupKFold atau bootstrap confidence interval |
| Noise ringan | 1.321 kalimat Indonesia berspasi di tepi; kutip miring di 19 tempat; kandidat typo OCR (dhdwuh, inkang, sasampuninpun), banyak kandidat lain sebenarnya kata sah | Normalisasi spasi dan kutip; koreksi typo secukupnya |
| Pasangan antar tingkat | 4.076 pasangan JV→JV terarah dari seluruh data | Bisa dipakai untuk multi-task / augmentation |
| Conversation | 80 baris, didominasi Ngoko | Terlalu kecil untuk training; opsional untuk evaluasi kualitatif |

## Model

| Peran | Model | Alasan | Catatan |
|---|---|---|---|
| **Utama** | **M2M100-418M** | Terbukti naik besar setelah fine-tuning di Unggah-Ungguh (BLEU 2,98 → 32,30, JAIC); kecil, full fine-tuning muat di Kaggle T4 | Lisensi belum dicek |
| **Pembanding utama** | **NLLB-200-distilled-600M** | Zero-shot terbaik di JAIC; terbaik setelah fine-tuning di NgokoKrama | Hasil fine-tuning di JAIC anomali, perlu dibuktikan ulang; lisensi CC BY-NC |
| Opsional | SahabatAI (Llama3 8B / Gemma2 9B), QLoRA | Model card mencantumkan Jawa | Zero-shot di ACL 2025 lemah; berat untuk T4; cukup sebagai zero-shot reference |
| Evaluator | Classifier DistilBERT | Akurasi 95,65% di ACL 2025 | **Latih ulang hanya pada train split** agar tidak leakage |

Model utama dan pembanding dipilih berdasarkan bukti riset; keduanya tetap dibandingkan di eksperimen.

## Workflow
```
1. Data preparation
   translation.csv -> normalisasi (spasi, kutip) -> "<tag> kalimat Indonesia" -> target kalimat Jawa
   -> group-based split 80/10/10 (SUDAH DIBUAT: 1.605 / 202 / 202) atau 5-fold GroupKFold
2. Baseline: copy baseline, zero-shot M2M100 & NLLB
3. Fine-tuning: M2M100 & NLLB, masing-masing:
   (a) tanpa tag
   (b) 4 atomic tag: <ngoko> <ngoko_alus> <krama> <krama_alus>
   (c) factorized tag: <ngoko>/<krama> + <biasa>/<alus>
   (d) opsional: (c) + multi-task pasangan JV->JV antar tingkat (hanya dari train split)
4. Evaluation: chrF++ (utama), BLEU per tingkat + level accuracy
   (classifier dilatih ulang di train split + rule-based dictionary) + confusion matrix
   + bootstrap confidence interval
5. Error analysis: salah tingkat, campur tingkat, hanya pronomina berubah, salah kata, makna berubah
6. (Opsional) Demo Gradio: textbox + dropdown tingkat
```

Matriks eksperimen: 2 model x 3-4 varian = 6-8 run. Dengan ±1.600 kalimat pendek, tiap run diperkirakan cepat di Kaggle T4 (belum diukur). GPU lokal RTX 3050 4 GB cukup untuk inference, kurang ideal untuk fine-tuning.

Detail:
- Tag sebagai special token (`add_special_tokens` + `resize_token_embeddings`). Kode bahasa: NLLB `ind_Latn` → `jav_Latn`, M2M100 `id` → `jv` (dari JAIC dan ingatan, konfirmasi saat implementasi).
- Hyperparameter awal mengikuti JAIC sebagai referensi: 3 epoch, lr 2e-4, batch 4, max length 128, fp16; tune di validation set.
- **Sanity check**: pastikan output fine-tuned berbeda dari zero-shot (pelajaran dari anomali NLLB di JAIC).
- Manual check penutur Jawa pada ±50 kalimat bila memungkinkan.

## Pertanyaan terbuka
- Selisih 2.009 vs 4.024 baris: cek halaman dataset / GitHub.
- Lisensi M2M100.
- Ada penutur Jawa untuk manual check?
- Deployment masuk atau tidak (keputusan dosen)?
- Baca full text NgokoKrama (IEEE) dan paper Wibawa sebelum menulis klaim kebaruan.

## Sumber
- Farhansyah dkk. (2025). Do Language Models Understand Honorific Systems in Javanese? ACL. https://arxiv.org/pdf/2502.20864v2
- Dataset: https://huggingface.co/datasets/JavaneseHonorifics/Unggah-Ungguh
- Prayoga dkk. (2026). Evaluating MT Models and LLMs for Indonesian–Javanese Translation Across Speech Levels. JAIC. https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/12326
- Yuliawati dkk. (2025). From Corpus to Benchmark: Indonesian-Javanese Krama Translation. IALP. https://dara.ui.ac.id/research-output/122df25b-37aa-45d2-91dd-5d171660daed
- Wibawa dkk. (2016). Augmented Javanese Speech Levels Machine Translation. ECTI-CIT. https://ph01.tci-thaijo.org/index.php/ecticit/article/view/54387
- Susanto dkk. (2023). Replicable Benchmarking of NMT on Low-Resource Local Languages in Indonesia. https://arxiv.org/abs/2311.00998v1
- Nădejde dkk. (2022). CoCoA-MT. https://arxiv.org/abs/2205.04022
- Studi sikap bahasa remaja (4,7% lancar Krama): https://paperity.org/p/365435331/exploring-teenagers-language-attitudes-towards-javanese-language
- Pergeseran bahasa anak Purworejo (Dewi, 2017), dikutip di: https://ojs.badanbahasa.kemdikbud.go.id/jurnal/index.php/kandai/article/download/7632/2318
- SahabatAI model card: https://huggingface.co/GoToCompany/llama3-8b-cpt-sahabatai-v1-base
