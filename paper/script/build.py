"""python build.py  ->  ../output/Paper_MT_Jawa_Tingkat_Tutur.docx dan .pdf

Urutan: (opsional) collect.py -> figures.py -> checks.py -> isi template DOCX -> PDF lewat Word (docx2pdf)
-> render tiap halaman PDF ke PNG (tmp_render/) untuk dicek.
Flag: --collect (hitung ulang results.json dari kaggle/*/output; lambat karena bootstrap), --no-pdf.
"""
import copy, os, re, sys, subprocess
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
from docx.shared import Cm, Pt, RGBColor

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "output")
NAME = "Paper_MT_Jawa_Tingkat_Tutur"
TEMPLATE = os.path.join(HERE, "template", "format_uas.docx")
sys.path.insert(0, HERE)

if "--collect" in sys.argv:
    subprocess.run([sys.executable, os.path.join(HERE, "collect.py")], check=True)
import figures
figures.build()
import content, references, checks
checks.run(content, references, figures)

TEXT_W = Cm(16.5)


# ---------------------------------------------------------------- inline markup
NUM = {}  # {"fig:eda": 3, "tab:cmp": 2}, diisi sebelum build dari urutan BODY
WIDTH = {"flow": 12.0, "trade": 14.5, "fish": 15.5, "audit": 15.0, "eda": 14.5, "lc": 15.5}


def number_blocks(body):
    f = t = 0
    for blk in body:
        if blk[0] == "figure": f += 1; NUM["fig:" + blk[1]] = f
        if blk[0] == "table": t += 1; NUM["tab:" + blk[1]] = t


def render_inline(text, used):
    """[@a; @b] -> (A, 2016; B, 2017);  @{a} -> A et al. (2016); *x* italic; **x** bold; [[fig:k]]/[[tab:k]] -> Gambar/Tabel n"""
    text = re.sub(r"\[\[(fig|tab):(\w+)\]\]", lambda m: ("Gambar " if m.group(1) == "fig" else "Tabel ") + str(NUM[f"{m.group(1)}:{m.group(2)}"]), text)
    def par(m):
        keys = [k.strip().lstrip("@") for k in m.group(1).split(";")]
        used.update(keys)
        return "(" + "; ".join(references.cite(k) for k in keys) + ")"

    def nar(m):
        k = m.group(1); used.add(k)
        c = references.cite(k); a, y = c.rsplit(", ", 1)
        return f"{a} ({y})"
    text = re.sub(r"\[(@[^\]]+)\]", par, text)
    text = re.sub(r"@\{([\w\-]+)\}", nar, text)
    segs = []
    for tok in re.split(r"(\*\*[^*]+\*\*|\*[^*]+\*)", text):
        if not tok: continue
        if tok.startswith("**"): segs.append((tok[2:-2], False, True))
        elif tok.startswith("*"): segs.append((tok[1:-1], True, False))
        else: segs.append((tok, False, False))
    return segs


def clear_runs(p):
    for r in list(p._p):
        if r.tag in (qn("w:r"), qn("w:hyperlink")):
            p._p.remove(r)


def add_runs(p, segs, size=None, base_bold=None, base_italic=None):
    for t, it, bd in segs:
        r = p.add_run(t)
        if it or base_italic: r.italic = True
        if bd or base_bold: r.bold = True
        if size: r.font.size = Pt(size)


# ---------------------------------------------------------------- builder
class Builder:
    def __init__(self):
        self.doc = Document(TEMPLATE)
        ps = self.doc.paragraphs
        self.proto = {"h1": copy.deepcopy(ps[20]._p), "p": copy.deepcopy(ps[21]._p),
                      "h2": copy.deepcopy(ps[28]._p), "cap": copy.deepcopy(ps[35]._p)}
        self.ps = ps
        self.used = set(); self.fig_n = 0; self.tab_n = 0
        self.body = self.doc.element.body
        self.sect = self.body[-1]
        self.cur_num = None

    def _append(self, el):
        self.sect.addprevious(el)
        from docx.text.paragraph import Paragraph
        return Paragraph(el, self.doc._body)

    def _new(self, kind):
        el = copy.deepcopy(self.proto[kind])
        p = self._append(el); clear_runs(p)
        return p

    # ---- bagian depan (isi placeholder template)
    def front(self, c):
        ps = self.ps
        clear_runs(ps[0]); r = ps[0].add_run(c.TITLE); r.bold = True; r.font.size = Pt(16)
        ps[2]._p.getparent().remove(ps[2]._p)
        clear_runs(ps[3]); ps[3].add_run(c.AUTHORS)
        for idx, key in [(7, "Background"), (9, "Objective"), (11, "Methods"), (13, "Results"), (15, "Conclusion")]:
            p = ps[idx]; clear_runs(p)
            r = p.add_run(f"{key}:"); r.bold = True
            add_runs(p, render_inline(" " + c.ABSTRACT[key], self.used))
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p = ps[17]; clear_runs(p)
        r = p.add_run("Keywords:"); r.bold = True; r.italic = True
        r = p.add_run(" " + c.KEYWORDS); r.italic = True
        # hapus seluruh isi dari "Introduction" sampai akhir (termasuk tabel contoh), sisakan sectPr
        start = ps[20]._p
        el = start
        while el is not None and el is not self.sect:
            nxt = el.getnext(); self.body.remove(el); el = nxt

    # ---- blok isi
    def h1(self, text):
        p = self._new("h1"); r = p.add_run(text); r.bold = True; r.font.size = Pt(14)
        p.paragraph_format.space_before = Pt(10); p.paragraph_format.space_after = Pt(4)
        p.paragraph_format.keep_with_next = True
        self.cur_num = None

    def h2(self, text):
        if self.cur_num is None:
            self.cur_num = self._restart_num()
        p = self._new("h2")
        p._p.pPr.numPr.numId.val = self.cur_num
        r = p.add_run(text); r.bold = True; r.font.size = Pt(12)
        p.paragraph_format.space_before = Pt(6); p.paragraph_format.space_after = Pt(2)
        p.paragraph_format.keep_with_next = True

    def _restart_num(self):
        numbering = self.doc.part.numbering_part.element
        ids = [int(n.get(qn("w:numId"))) for n in numbering.findall(qn("w:num"))]
        nid = max(ids) + 1
        num = OxmlElement("w:num"); num.set(qn("w:numId"), str(nid))
        a = OxmlElement("w:abstractNumId"); a.set(qn("w:val"), "1"); num.append(a)
        o = OxmlElement("w:lvlOverride"); o.set(qn("w:ilvl"), "0")
        s = OxmlElement("w:startOverride"); s.set(qn("w:val"), "1"); o.append(s); num.append(o)
        numbering.append(num)
        return nid

    def para(self, text):
        p = self._new("p"); add_runs(p, render_inline(text, self.used))
        p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        p.paragraph_format.space_after = Pt(6); p.paragraph_format.first_line_indent = Cm(0.75)

    def bullets(self, items):
        for it in items:
            p = self._new("p"); add_runs(p, render_inline("•  " + it, self.used))
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
            pf = p.paragraph_format; pf.left_indent = Cm(0.9); pf.first_line_indent = Cm(-0.45); pf.space_after = Pt(2)

    def caption(self, label, text):
        p = self._new("cap")
        r = p.add_run(label + " "); r.bold = True; r.font.size = Pt(10)
        add_runs(p, render_inline(text, self.used), size=10)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.space_after = Pt(8)
        return p

    def figure(self, key, fname, text):
        self.fig_n += 1; assert NUM["fig:" + key] == self.fig_n
        width_cm = WIDTH.get(key, 15.5)
        p = self._new("p"); p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.first_line_indent = Cm(0); p.paragraph_format.keep_with_next = True
        p.paragraph_format.space_before = Pt(4)
        p.add_run().add_picture(os.path.join(figures.FIG, fname), width=Cm(min(width_cm, 16.5)))
        self.caption(f"Gambar {self.fig_n}.", text)

    def table(self, key, title, header, rows, widths=None, notes=None, font=8.5, bold_rows=()):
        self.tab_n += 1; assert NUM["tab:" + key] == self.tab_n
        left = {i for i, h in enumerate(header) if h in ("Method", "Varian")}
        cap = self.caption(f"Tabel {self.tab_n}.", title)
        cap.paragraph_format.keep_with_next = True; cap.paragraph_format.space_after = Pt(3)
        t = self.doc.add_table(rows=1 + len(rows), cols=len(header))
        t.style = self.doc.styles["Table Grid"]
        self.sect.addprevious(t._tbl)
        for j, h in enumerate(header):
            self._cell(t.cell(0, j), h, font, bold=True, shade="D9E2EC", center=True)
        for i, row in enumerate(rows, 1):
            for j, v in enumerate(row):
                self._cell(t.cell(i, j), str(v), font, bold=(i - 1) in bold_rows, center=j not in left)
        for row in t.rows:  # header ikut berulang dan baris tidak terpotong halaman
            trPr = row._tr.get_or_add_trPr(); cs = OxmlElement("w:cantSplit"); trPr.append(cs)
        th = OxmlElement("w:tblHeader"); t.rows[0]._tr.get_or_add_trPr().append(th)
        if widths:
            for row in t.rows:
                for j, w in enumerate(widths):
                    row.cells[j].width = Cm(w)
        if notes:
            p = self._new("p"); add_runs(p, render_inline(notes, self.used), size=8.5)
            p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY; p.paragraph_format.space_after = Pt(8)
            p.paragraph_format.first_line_indent = Cm(0)
        else:
            self._new("p")

    def _cell(self, cell, text, font, bold=False, shade=None, center=True):
        p = cell.paragraphs[0]; clear_runs(p)
        add_runs(p, render_inline(text, self.used), size=font, base_bold=bold)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER if center else WD_ALIGN_PARAGRAPH.LEFT
        p.paragraph_format.space_after = Pt(0); p.paragraph_format.space_before = Pt(0)
        if shade:
            tcPr = cell._tc.get_or_add_tcPr(); sh = OxmlElement("w:shd")
            sh.set(qn("w:val"), "clear"); sh.set(qn("w:color"), "auto"); sh.set(qn("w:fill"), shade); tcPr.append(sh)

    def refs(self):
        for k in references.ordered_keys(self.used):
            p = self._new("p")
            for t, it in references.entry(k):
                r = p.add_run(t); r.italic = it; r.font.size = Pt(10)
            pf = p.paragraph_format; pf.left_indent = Cm(1.27); pf.first_line_indent = Cm(-1.27); pf.space_after = Pt(4)
            p.alignment = WD_ALIGN_PARAGRAPH.LEFT


def build_docx():
    number_blocks(content.BODY)
    b = Builder(); b.front(content)
    for blk in content.BODY:
        kind, *args = blk
        if kind == "refs":
            b.h1("References"); b.refs()
        else:
            getattr(b, kind)(*args)
    missing = set(references.REFS) - b.used
    assert not missing, f"referensi tidak disitasi: {missing}"
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, NAME + ".docx")
    b.doc.core_properties.title = content.TITLE
    b.doc.save(path)
    return path, b


def to_pdf(docx_path):
    from docx2pdf import convert
    pdf = docx_path[:-5] + ".pdf"
    convert(docx_path, pdf)
    return pdf


def render(pdf):
    import fitz
    d = os.path.join(HERE, "tmp_render"); os.makedirs(d, exist_ok=True)
    for f in os.listdir(d): os.remove(os.path.join(d, f))
    doc = fitz.open(pdf)
    for i, pg in enumerate(doc):
        pg.get_pixmap(matrix=fitz.Matrix(1.6, 1.6)).save(os.path.join(d, f"page{i + 1:02d}.png"))
    return len(doc)


if __name__ == "__main__":
    path, b = build_docx()
    print("DOCX", path, "| gambar", b.fig_n, "| tabel", b.tab_n, "| referensi", len(b.used))
    if "--no-pdf" not in sys.argv:
        pdf = to_pdf(path); n = render(pdf)
        print("PDF", pdf, "|", n, "halaman")
