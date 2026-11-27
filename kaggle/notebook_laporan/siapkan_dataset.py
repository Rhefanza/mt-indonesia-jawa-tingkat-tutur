"""Salin bahan mentah yang dibutuhkan notebook laporan ke kaggle/notebook_laporan/dataset/ (di-gitignore).

Isi folder dataset meniru struktur repo, jadi notebook bisa jalan di Kaggle maupun lokal:
  data/unggah_ungguh/...                     dataset Unggah-Ungguh (CC BY-NC 4.0)
  kaggle/<eksperimen>/output/...             prediksi, kontrol, metrics, log kernel
  models/juri_tetap/seed*/                   weights juri tetap (classifier penilai tingkat)
Jalankan dari root repo:  python kaggle/notebook_laporan/siapkan_dataset.py
Lalu (sekali):  kaggle datasets create -p kaggle/notebook_laporan/dataset --dir-mode zip
Versi baru:     kaggle datasets version -p kaggle/notebook_laporan/dataset --dir-mode zip -m "pesan"
"""
import glob, json, os, shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DST = os.path.join(os.path.dirname(__file__), "dataset")

POLA = [
    "data/unggah_ungguh/translation.csv",
    "data/unggah_ungguh/split/*.csv",
    "kaggle/baseline_zeroshot/output/predictions.csv",
    "kaggle/finetune_m2m100/output/pred_m2m100_418m_*.csv",
    "kaggle/finetune_m2m100/output/control_m2m100_418m_*.csv",
    "kaggle/finetune_nllb/output/pred_nllb_600m_atomic.csv",
    "kaggle/finetune_nllb/output/control_nllb_600m_atomic.csv",
    "kaggle/finetune_3lv_m2m100/output/pred3_*.csv",
    "kaggle/finetune_3lv_m2m100/output/control3_*.csv",
    "kaggle/finetune_3lv_nllb/output/pred3_*.csv",
    "kaggle/finetune_3lv_nllb/output/control3_*.csv",
    "kaggle/final_3lv_v2/output/*.csv",
    "kaggle/final_3lv_v2/output/*.json",
    "kaggle/final_3lv_postedit/output/pred_rerank2.0_pe_trig_gain.csv",
    "kaggle/final_3lv_postedit/output/control_rerank2.0_pe_trig_gain.csv",
    "kaggle/final_3lv_selftrain/output/pred_st_train_rerank0.5.csv",
    "kaggle/final_3lv_selftrain/output/control_st_train_rerank0.5.csv",
    "kaggle/final_3lv_merge/output/metrics_*.json",
    "kaggle/final_3lv_merge/output/control_*.csv",
    "kaggle/final_3lv_cendol_tune/output/kernel_log_v1.json",
    "kaggle/learning_curve_v2/analisis_lokal.json",
    "kaggle/learning_curve_v2/output/manifest.json",
    "kaggle/learning_curve_v2/output/valid_*.csv",
    "kaggle/lc_kontrol_langkah_v1/analisis_lokal.json",
    "kaggle/lc_kontrol_langkah_v1/output/manifest.json",
    "kaggle/lc_kontrol_langkah_v1/output/valid_*.csv",
    "models/juri_tetap/seed*/*",
]


def main():
    if os.path.exists(DST):
        shutil.rmtree(DST)
    n = 0
    for pola in POLA:
        hits = sorted(glob.glob(os.path.join(ROOT, pola)))
        assert hits, f"tidak ada file untuk pola {pola}"
        for src in hits:
            rel = os.path.relpath(src, ROOT)
            dst = os.path.join(DST, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            n += 1
    meta = {"title": "Unggah-Ungguh Bahan Notebook Laporan", "id": "mdevian/unggah-ungguh-bahan-notebook-laporan",
            "licenses": [{"name": "other"}],
            "description": "Bahan mentah notebook laporan UAS NLP (MT Indonesia-Jawa tingkat tutur): split Unggah-Ungguh "
                           "(CC BY-NC 4.0, JavaneseHonorifics/Unggah-Ungguh), prediksi dan log eksperimen, weights juri tetap. "
                           "Hanya untuk riset/kuliah, non-komersial."}
    json.dump(meta, open(os.path.join(DST, "dataset-metadata.json"), "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print("disalin", n, "file ke", DST)


if __name__ == "__main__":
    main()
