# Paper UAS: Kontrol Tingkat Tutur MT Indonesia → Jawa

Output: `output/Paper_MT_Jawa_Tingkat_Tutur.docx` dan `.pdf` (13 halaman, 6 gambar, 3 tabel, 38 referensi).
Template: `script/template/format_uas.docx` (salinan "Format UAS FP Paper.docx", gaya JISEBI).

## Aturan utama
- **Jangan edit `output/` langsung.** Semua perubahan lewat `script/`, lalu jalankan ulang build.
- **Jangan ketik angka hasil di `content.py`.** Angka diambil dari `script/data/results.json` lewat helper `c()`, `a()`,
  `ci()`, `lc_mean()`. `results.json` dibuat oleh `collect.py` dari `kaggle/*/output` (sumber mentah).
- Rujukan gambar/tabel pakai `[[fig:kunci]]` / `[[tab:kunci]]`; nomornya dihitung otomatis dari urutan di `BODY`.
  Sitasi: `[@kunci; @kunci2]` (dalam kurung) dan `@{kunci}` (naratif).

## Cara regenerate
```powershell
cd paper/script
pip install -r requirements.txt
python build.py              # figures -> checks -> DOCX -> PDF (Word lewat docx2pdf) -> tmp_render/page*.png
python build.py --collect    # hitung ulang results.json dari kaggle/*/output dulu (±15 menit, bootstrap di CPU)
python build.py --no-pdf     # tanpa Word
```
PDF dikonversi oleh **Microsoft Word** (docx2pdf), jadi PDF = DOCX persis. Tanpa Word, pakai LibreOffice
(`soffice --headless --convert-to pdf`); font Calibri bisa beda metrik dan halaman bergeser.

## Peta file
| File | Isi |
|---|---|
| `script/content.py` | judul, penulis, abstract (Inggris), seluruh isi bab (`BODY`), baris Tabel 2 (`ORDER`, `NAME`, `NOTE`) dan Tabel 3 |
| `script/collect.py` | hitung ulang chrF++/BLEU test dari `pred*.csv`, ketepatan tingkat dengan juri tetap dari `control*.csv`, paired bootstrap per grup, angka valid dari manifest |
| `script/figures.py` | 6 gambar (`fig_flowchart`, `fig_fishbone`, `fig_eda`, `fig_tradeoff`, `fig_learning_curve`, `fig_audit`) |
| `script/references.py` | formatter APA 7 dari `refs_raw/acl.bib` (BibTeX resmi ACL Anthology) + entri manual (Crossref/JMLR/arXiv) |
| `script/checks.py` | em dash, placeholder, istilah teknis yang diterjemahkan, ≥30 referensi & ≤5 sebelum 2021, semua referensi disitasi, flowchart + fishbone ada, teks tumpang-tindih di gambar, semua gambar/tabel dirujuk |
| `script/build.py` | isi template: placeholder judul/penulis/abstract diganti, isi dari "Introduction" ke bawah dihapus lalu dibangun ulang dari `BODY` |

## Keputusan yang sudah diambil
- Bahasa: isi Bahasa Indonesia, **abstract bahasa Inggris** (desimal titik via `en()`), judul bagian mengikuti template.
  Istilah teknis Inggris tidak diterjemahkan (dicek `checks.py`).
- Alur: struktur template (Introduction, Related Works, Methods [Material, Methodology], Result, Discussion, Conclusion, References).
- Tabel "Comparison Method" template (kolom P/R/F1) diganti metrik MT: chrF++, BLEU, ketepatan tingkat, ketepatan KA,
  Δ chrF++ vs model final (CI90 bootstrap). Alasan ditulis di catatan tabel.
- **Tabel TEST dan VALID dipisah.** Eksperimen sesudah model dibekukan hanya dilaporkan di valid (Tabel 3).
- Ketepatan tingkat = **juri tetap** (ensemble DistilBERT). Baris Cendol memakai juri kernel (ditandai ᶜ) karena prediksi
  kontrolnya tidak tersimpan.
- Gaya: formal akademik, tanpa em dash, APA 7.
- Penulis masih placeholder `Nama Penulis 1, 2, 3` (`AUTHORS` di `content.py`), **isi sendiri**.

## Catatan teknis
- Penomoran subjudul (1. Material, 2. Methodology, …) memakai list `numId` template; tiap bab membuat `w:num` baru dengan
  `startOverride=1` (`Builder._restart_num`) supaya nomor mulai lagi dari 1.
- Template default font Calibri 12 (docDefaults), heading Normal 14 bold, abstract style `JISEBI Abstract text` (TNR 9).
- Label Gambar 3 (trade-off) memakai nomor baris Tabel 2; panel (b) memperbesar klaster resep v2. Baris 20 (post-edit)
  identik dengan model final sehingga tidak digambar terpisah.
- `references.py`: judul ACL diubah ke sentence case otomatis (akronim/kata berhuruf kapital di tengah dipertahankan,
  daftar `KEEP`). Judul IWSLT 2023 di BibTeX asli huruf kapital semua, jadi ditimpa manual.
- `collect.py` meng-assert chrF++/BLEU model final = 58,62/33,17 (angka kernel v2) sebagai cek kewarasan.
- Audit 27 output Krama Alus (18/5/1/3) bersumber dari diskusi Claude ↔ Codex
  `D:\claude\diskusi bro 1 dan bro 2\2026-11-22-kualitas-terjemahan-tingkat-tutur.md` (Ringkasan Akhir), bukan dari file di repo.
  Konverter "fakta inti rusak 13/26" dari `kaggle/training_makna_tingkat_v1/README.md`; pool "15/22, 19/20" dari
  `kaggle/pool_makna_v1/README.md`.
- Halaman rawan bergeser: Gambar 2 (flowchart) dan Gambar 5 (fishbone) cukup tinggi; menambah teks di Methods/Result
  bisa memindahkannya ke halaman berikutnya dan meninggalkan ruang kosong.

## Checklist sebelum selesai
1. `python build.py` → "checks OK".
2. Buka tiap PNG di `script/figures/`.
3. Lihat semua `script/tmp_render/page*.png`: gambar tidak terpotong, caption menempel, tidak ada halaman kosong.
4. Cek nomor Gambar/Tabel di teks = caption (otomatis, tapi lihat sekilas).
5. Isi nama penulis.

## Sumber
- Angka: `kaggle/*/output/` (lihat `collect.py` untuk path per baris), `analisis/juri_tetap/`, `kaggle/learning_curve_v2/`,
  `kaggle/lc_kontrol_langkah_v1/`.
- Referensi: 38 entri, semua diverifikasi lewat ACL Anthology / DOI / arXiv (2026-11-23). Lima entri sebelum 2021:
  Sennrich 2016, Feely 2019, Papineni 2002, Popović 2017, Wibawa 2014.
