"""Cek otomatis sebelum DOCX dibangun. Gagal = AssertionError."""
import os, re


def _texts(content):
    out = [content.TITLE, content.KEYWORDS] + list(content.ABSTRACT.values())
    for blk in content.BODY:
        for a in blk[1:]:
            if isinstance(a, str): out.append(a)
            elif isinstance(a, (list, tuple)):
                for x in a:
                    if isinstance(x, str): out.append(x)
                    elif isinstance(x, (list, tuple)): out += [str(y) for y in x]
    return out


def check_text(content):
    allt = "\n".join(_texts(content))
    assert "—" not in allt, "em dash ditemukan"
    assert not re.search(r"<[A-Za-z][^>]*>", allt.replace("<ngoko>", "").replace("<krama>", "").replace("<krama_alus>", "")), "placeholder <...> tersisa"
    kw = [k.strip() for k in content.KEYWORDS.split(";")]
    assert 1 <= len(kw) <= 5, "keywords harus 1-5"
    # istilah teknis tidak boleh diterjemahkan
    BAD = ["penyetelan halus", "pemeringkatan ulang", "titik periksa", "pencarian berkas", "kurva pembelajaran",
           "sumber terbuka", "penghalusan label", "penyematan"]
    for b in BAD:
        assert b not in allt.lower(), f"istilah teknis diterjemahkan: {b}"
    return allt


def check_refs(content, references):
    allt = "\n".join(_texts(content))
    cited = set(re.findall(r"@\{([\w\-]+)\}", allt)) | {k.strip().lstrip("@") for g in re.findall(r"\[(@[^\]]+)\]", allt) for k in g.split(";")}
    unknown = cited - set(references.REFS)
    assert not unknown, f"sitasi tanpa entri: {unknown}"
    unused = set(references.REFS) - cited
    assert not unused, f"referensi tidak disitasi: {unused}"
    n = len(references.REFS); old = sum(int(r["year"]) < 2021 for r in references.REFS.values())
    assert n >= 30, f"referensi {n} < 30"
    assert old <= 5, f"referensi sebelum 2021: {old} > 5"


def check_figures(figures):
    names = os.listdir(figures.FIG)
    assert any("flowchart" in n for n in names), "flowchart tidak ada"
    assert any("fishbone" in n or "pohon" in n for n in names), "fishbone/pohon masalah tidak ada"
    for name, fig in figures.SAVED.items():
        fig.canvas.draw(); rend = fig.canvas.get_renderer()
        boxes = []
        for ax in fig.axes:
            for t in ax.texts + [ax.title, ax.xaxis.label, ax.yaxis.label]:
                if t.get_visible() and t.get_text().strip():
                    boxes.append((t.get_text()[:25], t.get_window_extent(rend)))
            leg = ax.get_legend()
            if leg: boxes.append(("legend", leg.get_window_extent(rend)))
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                a, b = boxes[i][1], boxes[j][1]
                ov = max(0, min(a.x1, b.x1) - max(a.x0, b.x0)) * max(0, min(a.y1, b.y1) - max(a.y0, b.y0))
                if ov > 0.15 * min(a.width * a.height, b.width * b.height):
                    raise AssertionError(f"{name}: teks tumpang-tindih '{boxes[i][0]}' x '{boxes[j][0]}'")


def check_body_refs(content):
    allt = "\n".join(_texts(content))
    keys = [("fig:" if b[0] == "figure" else "tab:") + b[1] for b in content.BODY if b[0] in ("figure", "table")]
    for k in keys:
        assert "[[" + k + "]]" in allt, f"{k} tidak dirujuk di teks"
    used = set(re.findall(r"\[\[((?:fig|tab):\w+)\]\]", allt))
    assert used <= set(keys), f"rujukan ke gambar/tabel yang tidak ada: {used - set(keys)}"
    assert not re.search(r"(Gambar|Tabel) \d", allt), "nomor gambar/tabel ditulis manual; pakai [[fig:..]]/[[tab:..]]"


def run(content, references, figures):
    check_text(content); check_refs(content, references); check_figures(figures); check_body_refs(content)
    print("checks OK")
