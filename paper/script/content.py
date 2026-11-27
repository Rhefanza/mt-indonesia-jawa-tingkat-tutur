"""SATU-SATUNYA sumber isi paper. Angka diambil dari data/results.json lewat helper di bawah (jangan ketik angka
hasil secara manual kecuali yang bersumber dari README eksperimen dan disebut path-nya di komentar).

Markup: [@a; @b] = sitasi dalam kurung, @{a} = sitasi naratif, *miring*, **tebal**.
Blok BODY: ("h1", teks) ("h2", teks) ("para", teks) ("bullets", [..]) ("figure", file, caption)
("table", judul, header, rows, widths, notes) ("refs",)
"""
import json, os

R = json.load(open(os.path.join(os.path.dirname(__file__), "data", "results.json"), encoding="utf-8"))
T, V, E = R["test"], R["valid"], R["eda"]


def n(x, d=2):
    """angka format Indonesia: 58,62 / 1.395"""
    if isinstance(x, int) or (isinstance(x, float) and d == 0):
        return f"{int(round(x)):,}".replace(",", ".")
    return f"{x:.{d}f}".replace(".", ",")


def c(rid, lv="all", m="chrF++", d=2): return n(T[rid]["scores"][lv][m], d)
def a(rid, lv="all", d=2): return n(T[rid]["level_acc"][lv], d)
def en(s): return s.replace(",", "§").replace(".", ",").replace("§", ".")  # 58,62 -> 58.62 ; 1.746 -> 1,746
def ci(rid):
    lo, hi = T[rid]["boot_vs_demo"]["ci90"]; return f"[{n(lo)}; {n(hi)}]"


LV = V["learning_curve"]; KL = V["kontrol_langkah"]
def lc_mean(frac, lv="all"):
    xs = [r["chrF++"][lv] for k, r in LV.items() if not k.startswith("_") and abs(r["frac"] - frac) < 1e-9]
    return sum(xs) / len(xs)


TITLE = "Kontrol Tingkat Tutur pada Machine Translation Indonesia ke Jawa: Source Tag, Rerank, dan Batas Data Kecil"
AUTHORS = "Nama Penulis 1, Nama Penulis 2, Nama Penulis 3"
KEYWORDS = "Javanese speech levels; controllable machine translation; low-resource NMT; reranking; learning curve"

ABSTRACT = {
    "Background": "Javanese has speech levels (Ngoko, Krama, Krama Alus) whose choice depends on the social relation between "
                  "speaker and addressee. Indonesian-to-Javanese machine translation systems ignore this choice, and "
                  "zero-shot multilingual models barely translate into Javanese at all, especially into the honorific levels.",
    "Objective": "We build a controllable Indonesian-to-Javanese translator in which the target speech level is selected "
                 "with a source prefix tag, and we compare every method family we tried under one evaluation protocol.",
    "Methods": f"Using the Unggah-Ungguh corpus ({en(n(E['n3'], 0))} three-level sentences, group-based split), we fine-tune "
               "M2M100-418M with level tags initialised from marker words, Krama Alus oversampling, and an auxiliary "
               "Krama-to-Krama Alus task, then rerank 8 beam candidates with a RoBERTa speech-level classifier. Seventeen "
               "variants (NLLB, Cendol mT5, extra data, self-training, post-editing, meaning-aware reranking, a converter) "
               "are compared with chrF++, BLEU, and a fixed DistilBERT judge, plus a learning curve with a step-matched control.",
    "Results": f"The final system reaches chrF++ {en(c('demo'))} and BLEU {en(c('demo', m='BLEU'))} on the test set, with "
               f"speech-level accuracy {en(a('demo'))} (Krama Alus {en(a('demo', 'krama_alus'))}), up from chrF++ "
               f"{en(c('zs_nllb'))} for the best zero-shot model. No later variant was significantly better on test "
               "(paired bootstrap per source group). "
               f"With the number of updates held at 1,905, the full training set still outperforms half of it by "
               f"{en(n(KL['gap']['f050_s1']))} to {en(n(KL['gap']['f050_s0']))} chrF++.",
    "Conclusion": "Level control is largely solved by tagging plus reranking; the remaining errors are meaning errors in "
                  "Krama Alus, which decoding tricks cannot fix and which shrink with more parallel data.",
}

# ---------------------------------------------------------------- tabel perbandingan TEST
ORDER = ["copy", "zs_m2m", "zs_nllb", "ft_notag", "ft_tag_m2m", "ft_tag_nllb", "r3_base", "r3_initos", "r3_nllb",
         "v2_beam", "demo", "v2_na_ls", "v2_nllb", "nusax", "aug", "aug_nusax", "st", "cendol_A", "cendol_B", "pe"]
NAME = {"copy": "Salin input (tanpa terjemahan)", "zs_m2m": "M2M100-418M zero-shot", "zs_nllb": "NLLB-200-600M zero-shot",
        "ft_notag": "M2M100 fine-tuning tanpa tag", "ft_tag_m2m": "M2M100 + source tag", "ft_tag_nllb": "NLLB-600M + source tag",
        "r3_base": "M2M100 + tag (khusus 3 tingkat)", "r3_initos": "+ init embedding tag + oversampling KA",
        "r3_nllb": "NLLB-600M + init + OS + multi-task", "v2_beam": "Resep v2 (M2M100), beam 4",
        "demo": "**Resep v2 + rerank (model final)**", "v2_na_ls": "v2 + data Ngoko Alus + label smoothing + rerank",
        "v2_nllb": "NLLB-600M resep v2 + rerank", "nusax": "v2 + NusaX (Ngoko) + rerank", "aug": "v2 + KA sintetis leksikon + rerank",
        "aug_nusax": "v2 + KA sintetis + NusaX + rerank", "st": "v2 + self-training pseudo-KA + rerank",
        "cendol_A": "Cendol mT5-base, nama tingkat di prompt", "cendol_B": "Cendol mT5-base + OS + LS + rerank",
        "pe": "Model final + post-editing leksikon"}
NOTE = {"ft_notag": "a", "ft_tag_m2m": "a", "ft_tag_nllb": "a", "r3_base": "b", "r3_initos": "b", "r3_nllb": "b",
        "cendol_A": "c", "cendol_B": "c", "nusax": "d", "aug": "d", "aug_nusax": "d"}


def _row(i, rid):
    r = T[rid]; s = r["scores"]["all"]; la = r["level_acc"]
    acc = "-" if la is None else n(la["all"]); ka = "-" if la is None else n(la["krama_alus"])
    if la is not None and r["judge"] == "kernel":
        acc += "ᶜ"; ka += "ᶜ"
    d = "" if rid == "demo" or "boot_vs_demo" not in r else ci(rid)
    name = NAME[rid] + (f" ⁽{NOTE[rid]}⁾" if rid in NOTE else "")
    return [str(i), name, n(s["chrF++"]), n(s["BLEU"]), acc, ka, d or ("acuan" if rid == "demo" else "n/a")]


TEST_ROWS = [_row(i, rid) for i, rid in enumerate(ORDER, 1)]
DEMO_ROW = ORDER.index("demo")

# ---------------------------------------------------------------- tabel VALID
def _vrow(label, d, note=""):
    return [label, n(d["all"]), n(d["ngoko"]), n(d["krama"]), n(d["krama_alus"]), note]


ms = [V[f"ms_seed{s}"] for s in (42, 43, 44)]
VALID_ROWS = [
    _vrow("Model final (epoch 14)", V["demo"], f"juri KA {V['demo']['KA_juri_benar']}/27"),
    _vrow("Ulang resep, seed 42", ms[0], f"epoch {ms[0]['best_epoch']} dipilih juri baru"),
    _vrow("Ulang resep, seed 43", ms[1], f"epoch {ms[1]['best_epoch']}"),
    _vrow("Ulang resep, seed 44", ms[2], f"epoch {ms[2]['best_epoch']}"),
    _vrow("NLLB-600M, resep sama", V["nllb600"], f"juri KA {V['nllb600']['KA_juri_benar']}/27"),
    ["Pool beam 8+16, rerank", "-", n(V["pool"]["ngoko"]), n(V["pool"]["krama"]), n(V["pool"]["krama_alus"]),
     f"CI90 bawah {n(V['pool']['lb90'])}"],
]

# ---------------------------------------------------------------- isi
BODY = [
("h1", "Introduction"),
("para", "Bahasa Jawa termasuk bahasa daerah dengan penutur terbanyak di Indonesia, tetapi tetap tergolong bahasa dengan "
         "sumber daya rendah untuk NLP [@aji; @nusacrowd]. Kekhasan yang membuatnya sulit bagi machine translation (MT) adalah *unggah-ungguh*, "
         "yaitu tingkat tutur yang dipilih menurut hubungan sosial penutur dan lawan bicara. Kalimat yang sama, misalnya "
         "\"Ibu sudah makan\", ditulis *Ibu wis mangan* dalam Ngoko, *Ibu sampun nedha* dalam Krama, dan "
         "*Ibu sampun dhahar* dalam Krama Alus. Memilih tingkat yang salah bukan sekadar salah gaya: di masyarakat Jawa "
         "hal itu dibaca sebagai tidak sopan. Sistem MT Indonesia ke Jawa yang ada tidak memberi pengguna cara untuk memilih tingkat ini."),
("para", "Masalahnya bertingkat. Secara umum, model multilingual besar seperti M2M100 dan NLLB-200 mencakup bahasa Jawa "
         f"[@fan2021; @nllb2024], tetapi pada data uji kami keduanya hanya mencapai chrF++ {c('zs_m2m')} dan {c('zs_nllb')} "
         f"secara zero-shot. Angka M2M100 bahkan lebih rendah dari sekadar menyalin kalimat Indonesia ({c('copy')}), dan "
         f"NLLB hanya mencapai chrF++ {c('zs_nllb', 'krama_alus')} untuk Krama Alus. Secara khusus, "
         "Krama Alus paling jarang muncul di data: dari 1.395 kalimat latih tiga tingkat, hanya 210 berlabel Krama Alus. "
         "Studi LLM pada dataset yang sama juga menemukan bias kuat ke Ngoko dan kinerja rendah pada tingkat hormat "
         "[@farhansyah]. Urgensinya praktis: alat bantu terjemahan yang bisa diminta tingkatnya berguna bagi pembelajar "
         "bahasa Jawa dan bagi penutur yang perlu menulis dalam ragam hormat untuk komunikasi formal."),
("para", "Penelitian kontrol formalitas di MT umumnya biner (formal vs informal) dan berangkat dari side constraint berupa "
         "tag di sisi sumber [@sennrich], yang kemudian diperluas ke tiga tingkat hormat bahasa Jepang [@feely] dan ke "
         "shared task IWSLT [@iwslt22; @iwslt23]. Untuk bahasa Jawa, karya terdahulu menerjemahkan antar tingkat di dalam "
         "bahasa Jawa dengan pendekatan pra-neural [@wibawa2014], atau mengevaluasi model tanpa mekanisme pemilih tingkat "
         "[@prayoga2026; @yuliawati2025]. Sepengetahuan kami, belum ada studi yang melakukan fine-tuning model MT Indonesia "
         "ke Jawa dengan kontrol tingkat di input dan mengukur ketepatan tingkatnya dengan penilai yang terpisah."),
("para", "Kontribusi paper ini ada tiga. Pertama, resep fine-tuning dengan source tag dan rerank berbasis classifier tingkat "
         f"yang menaikkan chrF++ test dari {c('zs_nllb')} (zero-shot terbaik) menjadi {c('demo')} dengan ketepatan tingkat "
         f"{a('demo')}. Kedua, perbandingan {len(ORDER) - 3} varian dari tujuh keluarga metode di bawah satu protokol "
         "evaluasi yang sama, termasuk hasil negatif yang biasanya tidak dilaporkan. Ketiga, analisis yang memisahkan dua "
         "jenis kesalahan: kesalahan tingkat yang sudah dapat dikendalikan, dan kesalahan makna yang tersisa dan baru "
         "berkurang bila data paralel bertambah."),

("h1", "Related Works"),
("para", "**Kontrol tingkat dan formalitas.** @{sennrich} memperkenalkan tag kesopanan di sisi sumber untuk Inggris ke "
         "Jerman, sementara @{feely} memakai tag tiga tingkat untuk honorifik Jepang dan menunjukkan tag mengungguli konversi "
         "berbasis aturan. @{schioppa} mengendalikan beberapa atribut sekaligus dengan additive interventions pada embedding. "
         "Shared task formality control di IWSLT 2022 dan 2023 [@iwslt22; @iwslt23] memicu banyak pendekatan, antara lain "
         "fine-tuning model multilingual pralatih [@rippeth], augmentasi data berlabel lemah dan post-editing berbasis aturan "
         "[@amazon], serta rerank kandidat dengan classifier formalitas yang hasilnya dipakai lagi sebagai data latih "
         "[@hwtsc]. @{cocoa} menyediakan benchmark kontrastif yang menjadi asal metrik ketepatan formalitas."),
("para", "**Reranking dan decoding sadar kualitas.** Rerank n-best dengan model diskriminatif [@discrr], quality-aware "
         "decoding [@qad], dan minimum Bayes risk decoding [@mbr] menunjukkan bahwa memilih ulang kandidat beam bisa "
         "menaikkan kualitas tanpa melatih ulang model. Rerank kami adalah varian sederhana: skor log-prob model ditambah "
         "probabilitas tingkat dari classifier RoBERTa bahasa Jawa [@wongso2021]."),
("para", "**MT bahasa Jawa dan bahasa daerah Indonesia.** IndoNLG [@indonlg], NusaX [@nusax], NusaWrites [@nusawrites], "
         "benchmark NMT bahasa daerah [@susanto2023], dan Cendol [@cendol] menyediakan data dan model untuk bahasa daerah "
         "Indonesia, tetapi tanpa label tingkat tutur. Dataset Unggah-Ungguh [@farhansyah] adalah sumber pertama yang "
         "melabeli tingkat tutur untuk pasangan Indonesia-Jawa, dan @{prayoga2026} melakukan fine-tuning NLLB serta M2M100 "
         "pada data itu tanpa pemilih tingkat. Korpus NgokoKrama [@yuliawati2025] hanya mencakup satu tingkat hormat."),
("para", "**MT data kecil.** Survei MT sumber daya rendah [@haddow; @ranathunga2023] menekankan bahwa jumlah data paralel "
         "dan setelan pelatihan sama-sama menentukan. Augmentasi data [@dasurvey] dan self-training [@selftrain] adalah "
         "dua cara umum menambah data, tetapi augmentasi sintetis untuk formalitas dilaporkan dapat merugikan [@lee2023]. "
         "Model multilingual yang menjadi titik awal kami adalah M2M100 [@fan2021], NLLB-200 [@nllb2024], mBART-50 "
         "[@mbart50], dan mT5 [@mt5]."),

("h1", "Methods"),
("para", "Bagian ini menjelaskan data yang dipakai dan alur metodologi. Semua eksperimen dijalankan sebagai kernel Kaggle "
         "(GPU T4) dengan notebook, log, dan prediksi tersimpan, sehingga setiap angka di bagian Result dapat dihitung ulang "
         "dari berkas keluarannya."),
("h2", "Material"),
("para", f"Kami memakai dataset Unggah-Ungguh [@farhansyah] versi publik di Hugging Face (lisensi CC BY-NC 4.0) yang berisi "
         f"{n(E['rows_total'], 0)} baris terjemahan Indonesia-Jawa berlabel empat tingkat. Mengikuti @{{prayoga2026}}, kami memakai "
         f"tiga tingkat (Ngoko, Krama, Krama Alus), sehingga tersisa {n(E['n3'], 0)} kalimat. Kalimat sumber Indonesia yang sama "
         "muncul di beberapa tingkat, sehingga split acak akan membocorkan kalimat uji ke data latih. Karena itu split dibuat "
         "**per grup kalimat sumber** dengan rasio sekitar 80/10/10 ([[tab:data]] dan [[fig:eda]]). Kalimatnya pendek, rata-rata "
         f"{n(E['len_words_id'], 1)} kata di sisi Indonesia, dan Krama Alus hanya mencakup sekitar 15% data."),
("table", "data", "Statistik dataset tiga tingkat setelah split per grup",
 ["Split", "Grup", "Ngoko", "Krama", "Krama Alus", "Total"],
 [[sp, n(E[sp + "_groups"], 0), n(E[sp]["ngoko"], 0), n(E[sp]["krama"], 0), n(E[sp]["krama_alus"], 0),
   n(E[sp]["ngoko"] + E[sp]["krama"] + E[sp]["krama_alus"], 0)] for sp in ["train", "valid", "test"]],
 [2.5, 2.0, 2.2, 2.2, 2.4, 2.2],
 "Jumlah grup dihitung pada split empat tingkat; baris Ngoko Alus tidak dipakai kecuali pada varian yang disebut."),
("figure", "eda", "fig03_eda.png", "Distribusi tingkat per split dan rata-rata panjang kalimat Jawa per tingkat."),
("h2", "Methodology"),
("para", "Alur penelitian ditunjukkan pada [[fig:flow]]. Setiap varian baru dipilih di set validasi, dan set uji hanya dibaca "
         "untuk varian yang lolos gerbang validasi atau untuk pelaporan akhir. Eksperimen sesudah model final dibekukan "
         "hanya memakai set validasi."),
("figure", "flow", "fig01_flowchart_metodologi.png", "Flowchart metodologi: dari data, fine-tuning, pemilihan checkpoint, dan rerank "
           "sampai evaluasi; varian yang tidak lolos gerbang validasi dicatat sebagai hasil negatif."),
("para", "**Source tag.** Tingkat yang diminta ditulis sebagai token khusus di depan kalimat sumber, misalnya "
         "\"<krama_alus> Ibu sudah makan.\" [@sennrich]. Embedding tiga token baru ini diinisialisasi dengan rata-rata "
         "embedding kata penanda tingkatnya (misalnya *panjenenganipun, dalem, dhahar, tindak, sare* untuk Krama Alus), "
         "bukan secara acak, supaya model langsung punya petunjuk arah sejak langkah pertama."),
("para", "**Resep fine-tuning (v2).** Model dasar M2M100-418M [@fan2021] dilatih full fine-tuning selama 15 epoch "
         "(batch 16, AdamW, learning rate 5e-5, warmup 10%, fp16). Contoh Krama Alus di-oversampling tiga kali, dan "
         "ditambahkan tugas multi-task Jawa ke Jawa yang memetakan kalimat Krama ke pasangan Krama Alus-nya (206 pasangan), "
         "karena dua tingkat ini paling sering tertukar. Checkpoint dipilih dengan skor validasi chrF++ + 30 × ketepatan "
         "tingkat."),
("para", "**Rerank tingkat.** Saat decoding, model menghasilkan 8 kandidat beam *y*, lalu kandidat dipilih dengan skor "
         "log *p*(*y*|*x*) + λ log *P*(tingkat diminta | *y*), λ = 2, dengan *P* dari classifier RoBERTa bahasa Jawa "
         "[@wongso2021] yang di-fine-tuning pada label tingkat data latih. Pendekatan ini sejalan dengan rerank berbasis "
         "classifier formalitas [@hwtsc] dan rerank diskriminatif [@discrr]."),
("para", "**Evaluasi.** Kualitas terjemahan diukur dengan chrF++ [@chrf] dan BLEU [@bleu] memakai sacreBLEU "
         f"{R['sacrebleu']} pada 176 kalimat uji berreferensi; chrF++ dijadikan metrik utama karena lebih stabil untuk "
         "bahasa beraglutinasi dan data kecil [@kocmi]. Ketepatan tingkat diukur pada set kontrol: 75 kalimat sumber uji "
         "diterjemahkan ke ketiga tingkat (225 permintaan), lalu dinilai oleh **juri tetap**, yaitu ensemble tiga "
         "classifier DistilBERT bahasa Jawa [@wongso2021] (seed 42, 13, 2024). Juri ini sengaja berbeda dari classifier "
         "rerank supaya ketepatan tingkat tidak naik palsu, dan bobotnya sama untuk semua eksperimen. Akurasinya pada "
         "referensi uji adalah 0,95 secara keseluruhan dan 0,89 untuk Krama Alus. Selisih chrF++ terhadap model final "
         "diuji dengan paired bootstrap per grup (1.000 sampel, interval 90%)."),
("para", "**Varian pembanding.** Selain resep utama, kami menguji tujuh keluarga metode yang diringkas pada [[tab:cmp]]: "
         "(1) zero-shot M2M100 dan NLLB-200; (2) fine-tuning dengan dan tanpa tag, termasuk NLLB-600M; (3) resep v2 dengan "
         "variasi data Ngoko Alus dan label smoothing; (4) data tambahan: NusaX bahasa Jawa [@nusax], Krama Alus sintetis "
         "hasil substitusi leksikon Krama ke Krama Alus, dan self-training dengan pseudo-label Krama Alus yang disaring "
         "reranker [@selftrain]; (5) model lain, yaitu Cendol mT5-base [@cendol; @mt5]; (6) pasca-proses berupa "
         "post-editing leksikon [@amazon]; dan (7) dua upaya menambahkan sinyal makna di validasi, yaitu rerank pool "
         "beam 8 dan 16 dengan skor terjemahan balik Jawa ke Indonesia serta konverter Krama ke Krama Alus."),

("h1", "Result"),
("para", "Subbab ini menyajikan perbandingan di set uji ([[tab:cmp]], [[fig:trade]]), eksperimen yang hanya dinilai di set "
         "validasi ([[tab:valid]]), analisis kesalahan ([[fig:audit]] dan [[fig:fish]]), dan learning curve ([[fig:lc]])."),
("h2", "Perbandingan metode di set uji"),
("table", "cmp", "Comparison Method (set uji: chrF++ dan BLEU pada 176 kalimat; ketepatan tingkat pada 225 permintaan kontrol)",
 ["No", "Method", "chrF++", "BLEU", "Ketepatan tingkat", "Ketepatan KA", "Δ chrF++ vs final (CI90)"],
 TEST_ROWS, [0.8, 6.0, 1.3, 1.3, 1.7, 1.6, 3.0],
 "(a) dilatih pada empat tingkat, dinilai pada tiga tingkat; (b) rata-rata tiga seed; (c) angka log kernel, ketepatan "
 "tingkat dari juri kernel, bukan juri tetap; (d) angka metrik kernel karena prediksi berreferensi tidak disimpan. "
 "Δ = chrF++ varian dikurangi model final pada sampel bootstrap per grup; n/a bila prediksi tidak tersimpan. "
 "Juri tetap tidak dipakai untuk zero-shot karena model tanpa tag tidak menerima permintaan tingkat."),
("para", f"[[tab:cmp]] menunjukkan tiga lompatan besar. Fine-tuning saja menaikkan chrF++ dari {c('zs_nllb')} (NLLB zero-shot) "
         f"ke {c('ft_notag')}, tetapi tanpa tag model hanya mengenai tingkat yang diminta pada {a('ft_notag')} permintaan, "
         f"kira-kira setara menebak. Menambah source tag menaikkan ketepatan tingkat ke {a('ft_tag_m2m')}, dan resep v2 "
         f"(init embedding tag, oversampling, multi-task) menaikkannya ke {a('v2_beam')}. Rerank menambah ketepatan Krama "
         f"Alus dari {a('v2_beam', 'krama_alus')} ke {a('demo', 'krama_alus')} dan chrF++ dari {c('v2_beam')} ke {c('demo')}; "
         f"selisih chrF++ ini kecil tetapi interval bootstrap-nya tidak memuat nol ({ci('v2_beam')}). "
         "Pola ini sama dengan temuan pada bahasa Jepang dan Korea: tag memberi lompatan terbesar, rerank memberi tambahan "
         "kecil tetapi konsisten [@feely; @hwtsc]."),
("para", "Tidak ada varian lanjutan yang lebih baik secara meyakinkan dari model final: semua varian sesudah resep v2 yang "
         "prediksinya tersimpan punya interval bootstrap selisih chrF++ yang memuat nol, sedangkan varian yang lebih lemah "
         "(fine-tuning tanpa resep v2) berada jelas di bawahnya. Beberapa varian memang sedikit lebih tinggi di satu atau dua "
         f"sumbu. Contohnya, v2 dengan data Ngoko Alus dan label smoothing mencapai chrF++ {c('v2_na_ls')} dan ketepatan "
         f"Krama Alus {a('v2_na_ls', 'krama_alus')}, dan NLLB-600M resep v2 mencapai chrF++ {c('v2_nllb')}. Varian dipilih di "
         "set validasi dengan juri kernel sebelum set uji dibaca, dan memilih ulang berdasarkan angka uji akan membuat angka "
         f"uji bias, jadi model final tidak diganti. Cendol mT5-base menerjemahkan lebih baik (chrF++ {c('cendol_A')}) "
         f"tetapi lebih jarang menuruti tag Krama Alus ({a('cendol_A', 'krama_alus')} menurut juri kernel). [[fig:trade]] "
         "memperlihatkan trade-off ini: varian dengan chrF++ tertinggi justru paling lemah di Krama Alus, dan sebagian besar "
         "varian resep v2 mengumpul di sekitar model final. Data tambahan NusaX menaikkan chrF++ "
         f"terutama di Ngoko ({c('demo', 'ngoko')} ke {c('nusax', 'ngoko')}), sesuai isinya yang berbentuk Ngoko, sedangkan "
         f"Krama Alus sintetis menurunkan ketepatan Krama Alus ({a('aug', 'krama_alus')}). Post-editing leksikon hampir tidak "
         "pernah aktif di set uji karena output Krama Alus yang meleset tidak memuat kata sumber leksikon."),
("figure", "trade", "fig04_tradeoff_test.png", "Trade-off kualitas terjemahan (chrF++ uji) dan ketepatan Krama Alus per varian. "
           "Tanda belah ketupat memakai juri kernel."),
("h2", "Eksperimen di set validasi"),
("para", "Setelah model final dibekukan, lima eksperimen dijalankan hanya di set validasi (175 kalimat) agar set uji tidak "
         "dipakai berulang ([[tab:valid]]). Mengulang resep yang sama dengan tiga seed menghasilkan chrF++ "
         f"{n(min(m_['all'] for m_ in ms))} sampai {n(max(m_['all'] for m_ in ms))}; seed 42 mengulang training secara "
         "identik (chrF++ greedy per epoch sama persis) tetapi juri kernel yang dilatih ulang memilih epoch lain. Jadi variasi "
         "terbesar berasal dari juri pemilih checkpoint, dan model final berada di ujung atas sebaran validasi."),
("table", "valid", "Eksperimen sesudah model final (set validasi, chrF++ dengan rerank)",
 ["Varian", "Semua", "Ngoko", "Krama", "Krama Alus", "Catatan"],
 VALID_ROWS, [4.6, 1.5, 1.5, 1.5, 1.8, 4.6],
 f"NLLB-600M lebih baik di Ngoko tetapi lebih buruk di Krama Alus. Pool beam 8 dan 16 menurunkan chrF++ Krama Alus; "
 f"skor terjemahan balik juga menurunkannya pada semua bobot yang dicoba. Konverter Krama ke Krama Alus dengan input "
 f"referensi Krama mencapai chrF++ {n(V['konverter']['chrF_konverter'])} (salinan Krama {n(V['konverter']['chrF_salinan'])}) "
 f"dan juri Krama Alus {V['konverter']['juri_KA_konverter']}/{V['konverter']['n']}, tetapi audit kami menemukan fakta inti rusak "
 f"pada 13 dari 26 kalimat."),
("para", f"[[tab:valid]] juga mencatat dua hasil negatif yang penting. NLLB-600M dengan resep yang sama mencapai chrF++ Ngoko "
         f"{n(V['nllb600']['ngoko'])}, lebih tinggi dari model final ({n(V['demo']['ngoko'])}), tetapi Krama Alus-nya turun ke "
         f"{n(V['nllb600']['krama_alus'])} ({n(V['demo']['krama_alus'])} pada model final); NLLB-1.3B tidak dilatih karena "
         "keunggulan zero-shot-nya atas 600M tidak konsisten. Konverter Krama ke Krama Alus menaikkan chrF++ dan ketepatan "
         "tingkat, tetapi audit makna menunjukkan konverter sering *menebak* kata alus (misalnya *pundhut*) yang mengubah "
         "fakta. Ini contoh nyata bahwa chrF++ dan metrik tingkat bisa naik sementara makna rusak."),
("h2", "Analisis kesalahan"),
("para", f"Juri tetap menilai {a('demo', 'krama_alus')} permintaan Krama Alus model final tepat tingkat, tetapi ketepatan tingkat "
         "tidak menjamin makna. Kami mengaudit manual 27 output Krama Alus berreferensi di set uji ([[fig:audit]]). Hanya 3 yang "
         "lolos; 18 salah makna (misalnya *mandi* menjadi *dhahar* \"makan\", *cincin* menjadi *ingkang rayi* \"adik\"), "
         "5 salah makna sekaligus salah arah hormat (krama andhap dipakai untuk orang yang dihormati), dan hanya 1 yang "
         "salah tingkat saja. Audit ini dilakukan penulis, bukan penutur asli, dan dicek silang oleh pemeriksa kedua, sehingga angkanya "
         "adalah taksonomi kerja. [[fig:fish]] merangkum penyebabnya dalam diagram fishbone."),
("figure", "audit", "fig06_audit_ka.png", "Hasil audit manual 27 output Krama Alus model final pada set uji."),
("figure", "fish", "fig02_fishbone_krama_alus.png", "Diagram fishbone penyebab output Krama Alus yang salah makna atau salah tingkat."),
("h2", "Learning curve"),
("para", "Untuk menguji apakah kesalahan makna dibatasi jumlah data, resep final dilatih ulang pada subset 25, 50, 75, dan "
         "100% grup latih (diambil per grup, stratified menurut keberadaan Krama Alus, dua permutasi). chrF++ validasi naik "
         f"dari {n(lc_mean(0.25))} ke {n(lc_mean(0.5))}, {n(lc_mean(0.75))}, dan {n(lc_mean(1.0))} ([[fig:lc]]a); Krama Alus "
         f"naik dari {n(lc_mean(0.25, 'krama_alus'))} ke {n(lc_mean(1.0, 'krama_alus'))}. Sebaliknya, ketepatan Krama Alus "
         "menurut juri tetap sudah 0,93 sampai 0,96 sejak subset terkecil. Proporsi token referensi validasi yang tidak "
         "pernah muncul di data latih turun dari 38% ke 23%."),
("para", "Karena epoch dibuat tetap, subset kecil juga mendapat langkah update lebih sedikit [@haddow]. Kami menjalankan "
         "kontrol dengan jumlah langkah terjadwal disamakan (1.905) dan rerank pada bobot titik akhir. Model 100% tetap unggul "
         f"{n(KL['gap']['f050_s1'])} dan {n(KL['gap']['f050_s0'])} chrF++ atas dua subset 50% (CI90 bootstrap "
         f"[{n(KL['boot']['f050_s1']['gap_boot_CI90'][0])}; {n(KL['boot']['f050_s1']['gap_boot_CI90'][1])}] dan "
         f"[{n(KL['boot']['f050_s0']['gap_boot_CI90'][0])}; {n(KL['boot']['f050_s0']['gap_boot_CI90'][1])}]), sedangkan "
         "menggandakan langkah untuk subset 50% hanya menaikkan chrF++ sekitar 0,7 ([[fig:lc]]b)."),
("figure", "lc", "fig05_learning_curve.png", "(a) Learning curve chrF++ validasi per tingkat (titik kecil: tiap permutasi subset). "
           "(b) Kontrol dengan jumlah langkah update disamakan."),

("h1", "Discussion"),
("para", "**Kontrol tingkat sudah tercapai; makna belum.** Hasil terpenting kami adalah pemisahan dua jenis kesalahan. "
         "Source tag dan rerank membuat model menuruti tingkat yang diminta pada sebagian besar permintaan, dan learning "
         "curve menunjukkan kepatuhan pada tag sudah jenuh pada beberapa ratus kalimat. Yang tersisa adalah kesalahan makna "
         "di Krama Alus: model menulis kata yang *terdengar* alus tetapi salah isi. Fenomena ini dekat dengan hallucination "
         "dan penggantian isi pada NMT [@halluc; @needle], diperparah oleh kosakata Krama Alus yang jarang muncul di data."),
("para", "**Mengapa trik decoding buntu.** Rerank hanya bisa memilih di antara kandidat yang dihasilkan model. Audit "
         "kandidat menunjukkan top-1 kehilangan fakta pada 15 dari 22 baris yang diaudit, dan hanya satu yang punya kandidat "
         "benar di pool beam 16. Skor terjemahan balik lolos uji fakta (19 dari 20) tetapi tetap menurunkan chrF++ Krama Alus. "
         "Quality-aware decoding dan MBR [@qad; @mbr] mengandaikan pool kandidat yang cukup kaya; pada data sekecil ini "
         "pengandaian itu tidak terpenuhi. Perbaikan harus datang dari generator, bukan dari pemilih."),
("para", "**Data: jumlah dan jenisnya.** Learning curve dengan kontrol langkah mendukung bahwa data paralel tambahan masih "
         "akan menaikkan kualitas, tetapi jenis data lebih menentukan daripada jumlahnya. Krama Alus sintetis menurunkan "
         "ketepatan Krama Alus karena hanya mengganti kata tanpa mengubah struktur kalimat, sejalan dengan temuan bahwa "
         "augmentasi sintetis untuk formalitas bisa merugikan [@lee2023; @amazon]. Pseudo-label self-training lolos saringan "
         "tingkat tetapi banyak yang rusak maknanya, karena reranker tidak menilai makna. Data yang paling mungkin membantu "
         "adalah paralel Indonesia-Krama Alus yang ditulis atau diperiksa penutur, seperti yang dilakukan NusaWrites untuk "
         "bahasa daerah lain [@nusawrites]."),
("para", "**Keterbatasan.** Pertama, output belum dinilai penutur asli; audit makna dilakukan penulis. Kedua, juri tingkat "
         "hanya 0,89 akurat pada referensi Krama Alus, sehingga ketepatan Krama Alus cenderung under-estimate, dan juri "
         "yang dipakai memilih checkpoint ternyata menjadi sumber variasi terbesar antar run. Ketiga, sebagian besar "
         "perbandingan hanya satu seed, dan set uji berisi 27 kalimat Krama Alus, sehingga selisih di bawah sekitar 2,5 "
         "chrF++ berada di dalam interval kepercayaan. Keempat, dataset berisi kalimat pendek dari satu domain, sehingga "
         "kinerja pada kalimat panjang atau dialek belum diketahui."),

("h1", "Conclusion"),
("para", f"Kami membangun MT Indonesia ke Jawa yang tingkat tuturnya dapat dipilih lewat source tag. Resep fine-tuning "
         f"M2M100-418M dengan inisialisasi embedding tag, oversampling Krama Alus, multi-task Krama ke Krama Alus, dan rerank "
         f"classifier tingkat mencapai chrF++ {c('demo')}, BLEU {c('demo', m='BLEU')}, dan ketepatan tingkat {a('demo')} pada "
         "set uji. Dari tujuh keluarga metode yang dibandingkan, tidak ada yang secara meyakinkan mengunggulinya untuk "
         "Krama Alus. Kesalahan yang tersisa terutama kesalahan makna, bukan tingkat, dan learning curve dengan kontrol "
         "langkah menunjukkan kualitas masih naik bersama jumlah data paralel."),
("para", "Pekerjaan lanjutan yang kami sarankan: (1) evaluasi manusia oleh penutur Jawa dengan skala makna dan kesesuaian "
         "tingkat; (2) pengumpulan data paralel Indonesia-Krama Alus yang diperiksa penutur, termasuk kemungkinan memakai "
         "korpus NgokoKrama setelah lisensinya jelas [@yuliawati2025]; (3) metrik makna yang sadar arah hormat untuk "
         "menggantikan chrF++ sebagai penilai utama; dan (4) model yang lebih besar seperti NLLB atau LLM bahasa daerah "
         "[@nllb2024; @cendol] setelah data hormat yang cukup tersedia."),
("refs",),
]
