"""Semua gambar paper, digenerate dari data/results.json. Satu fungsi per gambar; output ke figures/*.png (dpi 220).
Teks tumpang-tindih dicek oleh checks.py lewat registry TEXT_BOXES yang diisi saat menggambar.
"""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Polygon
import numpy as np

HERE = os.path.dirname(__file__)
FIG = os.path.join(HERE, "figures")
R = json.load(open(os.path.join(HERE, "data", "results.json"), encoding="utf-8"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "savefig.dpi": 220})
C = {"ngoko": "#4C78A8", "krama": "#F58518", "krama_alus": "#B279A2", "ink": "#222222", "muted": "#8a8a8a",
     "accent": "#C0392B", "ok": "#2E8B57", "box": "#EEF2F7", "box2": "#FDF1E6", "dec": "#FFF7D6"}
LVN = {"ngoko": "Ngoko", "krama": "Krama", "krama_alus": "Krama Alus"}
FAM_C = {"Baseline": "#9E9E9E", "Zero-shot": "#757575", "Fine-tuning": "#4C78A8", "Resep v2": "#C0392B",
         "Data tambahan": "#2E8B57", "Model lain": "#F58518", "Pasca-proses": "#8E6C8A"}
SAVED = {}


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    SAVED[name] = fig
    return p


# ---------------------------------------------------------------- 1. flowchart metodologi
def _box(ax, x, y, w, h, text, fc, ec="#4a4a4a", bold=False, fs=7.4):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                fc=fc, ec=ec, lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)


def _diamond(ax, x, y, w, h, text, fs=7.5):
    ax.add_patch(Polygon([(x, y + h / 2), (x + w / 2, y), (x, y - h / 2), (x - w / 2, y)], closed=True,
                         fc=C["dec"], ec="#8a6d00", lw=0.9))
    ax.text(x, y, text, ha="center", va="center", fontsize=fs)


def _arrow(ax, x1, y1, x2, y2, label=None, lx=0, ly=0):
    ax.annotate("", (x2, y2), (x1, y1), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#333333"))
    if label:
        ax.text((x1 + x2) / 2 + lx, (y1 + y2) / 2 + ly, label, fontsize=7, color="#333333", ha="center", va="center")


def fig_flowchart():
    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 11); ax.axis("off")
    W, H = 4.9, 0.72
    _box(ax, 2.6, 10.3, W, H, "Unggah-Ungguh (2.009 baris)\nfilter 3 tingkat: 1.746 kalimat", C["box"], bold=True)
    _box(ax, 2.6, 9.15, W, H, "Split per grup (kalimat sumber)\ntrain 1.395 / valid 175 / test 176", C["box"])
    _box(ax, 2.6, 8.0, W, H, "Source tag <ngoko> <krama> <krama_alus>\n+ init embedding tag dari kata penanda", C["box"])
    _box(ax, 2.6, 6.85, W, H, "Fine-tuning M2M100-418M (fp16, 15 epoch)\noversampling KA x3 + multi-task Krama->KA", C["box"])
    _box(ax, 2.6, 5.7, W, H, "Pilih checkpoint di VALID\nchrF++ + 30 x ketepatan (juri kernel)", C["box"])
    _box(ax, 2.6, 4.55, W, H, "Beam 8 -> rerank\nlog p(y|x) + 2 log P(tingkat | y), RoBERTa", C["box2"])
    _diamond(ax, 2.6, 3.25, 3.4, 1.2, "Varian baru lolos\ngerbang VALID?")
    _box(ax, 2.6, 1.7, W, H, "Evaluasi TEST sekali (176 kalimat)\nchrF++, BLEU + juri tetap (225 permintaan)", C["box"], bold=True)
    _box(ax, 2.6, 0.55, W, H, "Model demo dibekukan\n+ analisis: multi-seed, learning curve", C["box"])
    for y1, y2 in [(10.3, 9.15), (9.15, 8.0), (8.0, 6.85), (6.85, 5.7), (5.7, 4.55)]:
        _arrow(ax, 2.6, y1 - H / 2, 2.6, y2 + H / 2)
    _arrow(ax, 2.6, 4.55 - H / 2, 2.6, 3.25 + 0.6)
    _arrow(ax, 2.6, 3.25 - 0.6, 2.6, 1.7 + H / 2, "ya", 0.25, 0)
    _arrow(ax, 2.6, 1.7 - H / 2, 2.6, 0.55 + H / 2)
    # cabang eksperimen
    _box(ax, 7.6, 7.4, 4.0, 2.9, "Varian yang diuji\n\n- model: NLLB-600M/1.3B, Cendol mT5\n- data: NusaX, KA sintetis,\n  Ngoko Alus, self-training\n- decoding: pool beam 16,\n  skor makna reverse\n- pasca-proses: post-edit leksikon,\n  konverter Krama->KA",
         "#F7F7F7", fs=7.5)
    _arrow(ax, 5.6, 7.4, 5.05, 6.85)
    _arrow(ax, 4.3, 3.25, 6.3, 3.25, "tidak", 0, 0.18)
    _box(ax, 7.6, 3.25 + 0.0, 2.6, 0.8, "hasil negatif dicatat,\ntest tidak dibaca", "#FBE9E7", fs=7.5)
    _arrow(ax, 7.6, 3.25 + 0.4, 7.6, 5.95)
    ax.text(7.75, 4.8, "iterasi\nberikutnya", fontsize=7, color="#555555")
    return save(fig, "fig01_flowchart_metodologi.png")


# ---------------------------------------------------------------- 2. fishbone penyebab KA meleset
def fig_fishbone():
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    ax.set_xlim(-2.6, 13.6); ax.set_ylim(0, 7); ax.axis("off")
    ax.annotate("", (11.2, 3.5), (-0.6, 3.5), arrowprops=dict(arrowstyle="-|>", lw=1.6, color=C["ink"]))
    ax.add_patch(FancyBboxPatch((11.25, 2.55), 2.15, 1.9, boxstyle="round,pad=0.03", fc="#FBE9E7", ec=C["accent"]))
    ax.text(12.32, 3.5, "Output\nKrama Alus\nsalah makna\natau tingkat", ha="center", va="center", fontsize=7.2, fontweight="bold")
    cats = [
        ("Data", 0.9, True, ["KA hanya 210 kalimat train", "23% token ref. valid tak ada di train", "kalimat pendek (~8 kata)"]),
        ("Model", 5.2, True, ["bias ke Krama biasa", "menebak kata alus (pundhut)", "tak ada sinyal makna"]),
        ("Decoding", 9.5, True, ["rerank hanya menilai tingkat", "pool beam miskin kandidat benar"]),
        ("Evaluasi", 0.9, False, ["juri KA 0,89 pada referensi", "chrF++ buta arah hormat", "juri pemilih checkpoint goyah"]),
        ("Bahasa", 5.2, False, ["krama inggil vs krama andhap", "arah hormat (siapa dihormati)"]),
        ("Augmentasi", 9.5, False, ["KA sintetis: struktur Krama", "pseudo-label rusak makna"]),
    ]
    for name, x, top, items in cats:
        y0 = 6.4 if top else 0.6
        ax.plot([x, x + 1.3], [y0, 3.5], color=C["muted"], lw=1.1)
        ax.text(x - 0.05, y0 + (0.22 if top else -0.22), name, ha="center", va="bottom" if top else "top",
                fontsize=8.5, fontweight="bold", color=C["accent"])
        for i, it in enumerate(items):
            t = (i + 1) / (len(items) + 1)
            yy = y0 + (3.5 - y0) * t
            xx = x + 1.3 * t
            ax.plot([xx - 0.25, xx], [yy, yy], color=C["muted"], lw=0.7)
            ax.text(xx - 0.3, yy, it, ha="right", va="center", fontsize=6.3)
    return save(fig, "fig02_fishbone_krama_alus.png")


# ---------------------------------------------------------------- 3. EDA
def fig_eda():
    e = R["eda"]
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.8), constrained_layout=True)
    sp = ["train", "valid", "test"]; x = np.arange(3); w = 0.26
    for i, lv in enumerate(["ngoko", "krama", "krama_alus"]):
        vals = [e[s].get(lv, 0) for s in sp]
        b = axs[0].bar(x + (i - 1) * w, vals, w, color=C[lv], label=LVN[lv])
        for rect, v in zip(b, vals):
            axs[0].text(rect.get_x() + rect.get_width() / 2, v + 8, str(v), ha="center", fontsize=6.5)
    axs[0].set_xticks(x, ["train", "valid", "test"]); axs[0].set_ylabel("jumlah kalimat")
    axs[0].set_title("(a) Distribusi tingkat per split"); axs[0].legend(frameon=False, fontsize=7)
    axs[0].set_ylim(0, 690)
    L = e["len_words"]
    lv = ["ngoko", "krama", "krama_alus"]
    axs[1].bar([LVN[l] for l in lv], [L[l]["mean"] for l in lv], color=[C[l] for l in lv])
    for i, l in enumerate(lv):
        axs[1].text(i, L[l]["mean"] - 0.8, f"{L[l]['mean']:.1f}".replace(".", ","), ha="center", fontsize=7.5, color="white",
                    fontweight="bold")
    axs[1].axhline(e["len_words_id"], color=C["ink"], ls="--", lw=0.9,
                   label=f"rata-rata sumber Indonesia ({e['len_words_id']:.1f})".replace(".", ","))
    axs[1].legend(frameon=False, fontsize=6.8, loc="upper center")
    axs[1].set_ylabel("rata-rata kata"); axs[1].set_title("(b) Panjang kalimat Jawa")
    axs[1].set_ylim(0, 10.5)
    return save(fig, "fig03_eda.png")


# ---------------------------------------------------------------- 4. trade-off chrF++ vs ketepatan KA (TEST)
LABEL_SHORT = {"ft_notag": "tanpa tag", "ft_tag_m2m": "M2M+tag", "ft_tag_nllb": "NLLB+tag", "r3_base": "M2M+tag 3t",
               "r3_initos": "+init+OS", "r3_nllb": "NLLB+init+OS+MT", "v2_beam": "v2 beam4", "demo": "DEMO (v2+rerank)",
               "v2_na_ls": "v2+NA+LS", "v2_nllb": "NLLB v2", "pe": "post-edit", "st": "self-train", "aug": "+KA sintetis",
               "nusax": "+NusaX", "aug_nusax": "+sintetis+NusaX", "cendol_A": "Cendol prompt", "cendol_B": "Cendol OS+LS"}
OFFS = {"demo": (6, -10), "pe": (6, 4), "v2_beam": (-4, -11), "nusax": (5, 3), "aug": (-40, 5), "aug_nusax": (5, -3),
        "st": (6, 2), "v2_na_ls": (5, 3), "cendol_A": (-54, 4), "cendol_B": (5, 3), "v2_nllb": (5, -6), "r3_nllb": (5, 2),
        "r3_initos": (-36, 5), "r3_base": (5, -8), "ft_tag_nllb": (5, 2), "ft_tag_m2m": (5, -8), "ft_notag": (5, 2)}


def fig_tradeoff():
    import content
    from matplotlib.patches import Rectangle
    fig, (ax, az) = plt.subplots(1, 2, figsize=(7.4, 3.6), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})
    ZX, ZY = (57.5, 61.0), (0.67, 0.95)
    zoff = {"r3_initos": (6, -3), "v2_beam": (6, -4), "demo": (-46, -3), "v2_na_ls": (7, 0), "v2_nllb": (7, -3),
            "nusax": (4, 6), "st": (-14, 6), "aug": (7, -4), "aug_nusax": (7, -3)}
    for i, rid in enumerate(content.ORDER, 1):
        r = R["test"][rid]
        if r["level_acc"] is None or rid == "pe":  # pe identik dengan model final
            continue
        x = r["scores"]["all"]["chrF++"]; y = r["level_acc"]["krama_alus"]
        mk = "D" if r["judge"] == "kernel" else "o"; col = FAM_C[r["family"]]
        inz = ZX[0] <= x <= ZX[1] and ZY[0] <= y <= ZY[1]
        for axx, big in ((ax, 40 if rid == "demo" else 22), (az, 80 if rid == "demo" else 40)):
            axx.scatter(x, y, s=big, marker=mk, color=col, edgecolor="white", lw=0.6, zorder=3)
        if inz:
            lab = f"{i} (final)" if rid == "demo" else str(i)
            az.annotate(lab, (x, y), xytext=zoff.get(rid, (6, 2)), textcoords="offset points", fontsize=7.5,
                        fontweight="bold" if rid == "demo" else "normal")
        else:
            off = {"ft_tag_nllb": (5, -9), "ft_tag_m2m": (-10, 3)}.get(rid, (5, 2))
            ax.annotate(str(i), (x, y), xytext=off, textcoords="offset points", fontsize=7)
    ax.add_patch(Rectangle((ZX[0], ZY[0]), ZX[1] - ZX[0], ZY[1] - ZY[0], fill=False, ec="#555555", lw=0.8, ls="--"))
    ax.set_xlim(40, 65); ax.set_ylim(0.15, 1.0)
    az.set_xlim(*ZX); az.set_ylim(*ZY)
    ax.set_xlabel("chrF++ uji (176 kalimat)"); az.set_xlabel("chrF++ uji (176 kalimat)")
    ax.set_ylabel("ketepatan Krama Alus (75 permintaan)")
    ax.set_title("(a) Semua varian"); az.set_title("(b) Perbesaran kotak putus-putus")
    for f, col in FAM_C.items():
        if f in ("Baseline", "Zero-shot", "Pasca-proses"): continue
        ax.scatter([], [], color=col, label=f, s=22)
    ax.scatter([], [], marker="D", color="#777777", label="juri kernel", s=18)
    ax.legend(frameon=False, fontsize=6.3, loc="upper left")
    return save(fig, "fig04_tradeoff_test.png")


# ---------------------------------------------------------------- 5. learning curve + kontrol langkah
def fig_learning_curve():
    lc = R["valid"]["learning_curve"]; kl = R["valid"]["kontrol_langkah"]
    runs = {k: v for k, v in lc.items() if not k.startswith("_")}
    fig, axs = plt.subplots(1, 2, figsize=(7.2, 2.9), constrained_layout=True, gridspec_kw={"width_ratios": [1.5, 1]})
    for lv in ["all", "ngoko", "krama", "krama_alus"]:
        pts = sorted((v["n_kalimat"], v["chrF++"][lv]) for v in runs.values())
        xs = sorted(set(p[0] for p in pts)); fr = {}
        for n, c in pts:
            key = min(xs, key=lambda q: abs(q - n)) if False else round(n, -1)
            fr.setdefault(key, []).append(c)
        xx = sorted(fr); yy = [np.mean(fr[k]) for k in xx]
        col = "#222222" if lv == "all" else C[lv]
        axs[0].plot(xx, yy, "-o", color=col, ms=3.5, lw=1.6 if lv == "all" else 1.1,
                    label="semua" if lv == "all" else LVN[lv])
        axs[0].scatter([p[0] for p in pts], [p[1] for p in pts], color=col, s=7, alpha=0.45)
    axs[0].set_xscale("log", base=2); axs[0].set_xticks([350, 700, 1050, 1395], ["350", "700", "1.050", "1.395"])
    axs[0].set_xlabel("kalimat train (subset per grup)"); axs[0].set_ylabel("chrF++ VALID (rerank)")
    axs[0].set_title("(a) Learning curve, 15 epoch"); axs[0].legend(frameon=False, fontsize=6.8, ncol=2)
    ak = kl["akhir"]
    names = ["f050_s0", "f050_s1", "f100"]; labs = ["50% s0", "50% s1", "100%"]
    vals = [ak[n]["all"]["chrF++"] for n in names]
    axs[1].bar(labs, vals, color=["#9DB4CC", "#9DB4CC", "#4C78A8"])
    for i, v in enumerate(vals):
        axs[1].text(i, v + 0.3, f"{v:.2f}".replace(".", ","), ha="center", fontsize=7)
    axs[1].set_ylim(50, 65); axs[1].set_ylabel("chrF++ VALID, titik akhir")
    axs[1].set_title("(b) Langkah disamakan: 1.905")
    return save(fig, "fig05_learning_curve.png")


# ---------------------------------------------------------------- 6. audit 27 output KA
def fig_audit():
    cats = ["salah makna", "salah makna + arah hormat", "salah tingkat saja", "lolos"]
    vals = [18, 5, 1, 3]
    fig, ax = plt.subplots(figsize=(7.2, 1.45), constrained_layout=True)
    left = 0; cols = [C["accent"], "#E67E22", C["krama_alus"], C["ok"]]
    for c, v, col in zip(cats, vals, cols):
        ax.barh([0], [v], left=left, color=col, height=0.6, label=f"{c} ({v})")
        ax.text(left + v / 2, 0, f"{v}", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
        left += v
    ax.set_xlim(0, 27); ax.set_ylim(-0.45, 0.45); ax.axis("off")
    ax.legend(frameon=False, fontsize=7, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.02))
    ax.set_title("Audit manual 27 output Krama Alus model final (set uji)", fontsize=9)
    return save(fig, "fig06_audit_ka.png")


ALL = [fig_flowchart, fig_fishbone, fig_eda, fig_tradeoff, fig_learning_curve, fig_audit]


def build():
    return [f() for f in ALL]


if __name__ == "__main__":
    for p in build():
        print(p)
