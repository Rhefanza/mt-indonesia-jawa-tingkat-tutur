"""Daftar pustaka APA 7. Sumber: refs_raw/acl.bib (BibTeX resmi ACL Anthology) + entri manual yang metadatanya
diambil dari Crossref/JMLR/arXiv (lihat README bagian Sumber). Tidak ada entri yang ditulis dari ingatan.

cite(key) -> teks sitasi dalam kalimat, mis. "Sennrich et al., 2016". Kunci yang dipakai di content.py dicek di checks.py:
setiap referensi harus disitasi minimal sekali dan setiap sitasi harus ada di daftar.
"""
import os, re

HERE = os.path.dirname(__file__)
ACCENT = {r"{\v{r}}": "ř", r"{\"i}": "ï", r"{\'o}": "ó", r"{\'e}": "é", r"{\`a}": "à", r"{\'c}": "ć", "{'}": "'",
          r"\v{r}": "ř", r"\'e": "é", r"\'o": "ó", r"\`a": "à", r"\'c": "ć", r"\"i": "ï"}


import unicodedata
_COMB = {"`": "̀", "'": "́", "^": "̂", "~": "̃", '"': "̈", "v": "̌", "c": "̧"}


def _clean(s):
    s = " ".join(s.split()).replace("{'}", "'")
    s = re.sub(r"\{?\\([`'^~\"vc])\{?(\w)\}?\}?", lambda m: unicodedata.normalize("NFC", m.group(2) + _COMB[m.group(1)]), s)
    return s.replace("{", "").replace("}", "").replace("--", "–")


def _field(e, f):
    m = re.search(rf'\b{f}\s*=\s*"((?:[^"\\]|\\.)*)"', e) or re.search(rf"\b{f}\s*=\s*\{{(.*?)\}},?\s*\n", e, re.S)
    return _clean(m.group(1)) if m else None


def _authors(s):
    out = []
    for a in s.split(" and "):
        a = a.strip()
        if "," in a:
            last, first = [x.strip() for x in a.split(",", 1)]
        else:
            parts = a.split(); last, first = parts[-1], " ".join(parts[:-1])
        ini = " ".join(p[0] + "." for p in re.split(r"[\s]+", first) if p)
        ini = re.sub(r"(\w)\.-", r"\1.-", ini)
        out.append((last, ini))
    return out


def _parse_acl():
    s = open(os.path.join(HERE, "refs_raw", "acl.bib"), encoding="utf-8").read()
    refs = {}
    for e in re.findall(r"@\w+\{[^@]*", s):
        key = re.search(r"@\w+\{([^,]+)", e).group(1).strip()
        venue = _field(e, "booktitle") or _field(e, "journal")
        refs[key] = {"authors": _authors(_field(e, "author")), "year": _field(e, "year"), "title": _field(e, "title"),
                     "venue": venue, "pages": _field(e, "pages"), "doi": _field(e, "doi"), "url": _field(e, "url"),
                     "journal": bool(_field(e, "journal")), "volume": _field(e, "volume"), "number": _field(e, "number")}
    return refs


MANUAL = {
    "fan2021": {"authors": _authors("Fan, Angela and Bhosale, Shruti and Schwenk, Holger and Ma, Zhiyi and El-Kishky, Ahmed and Goyal, Siddharth and Baines, Mandeep and Celebi, Onur and Wenzek, Guillaume and Chaudhary, Vishrav and Goyal, Naman and Birch, Tom and Liptchinsky, Vitaliy and Edunov, Sergey and Auli, Michael and Joulin, Armand"),
                "year": "2021", "title": "Beyond English-centric multilingual machine translation", "venue": "Journal of Machine Learning Research",
                "volume": "22", "number": "107", "pages": "1–48", "url": "https://jmlr.org/papers/v22/20-1307.html", "journal": True},
    "nllb2024": {"authors": [("NLLB Team", "")] + _authors("Costa-jussà, Marta R. and Cross, James and Çelebi, Onur and Elbayad, Maha and Heafield, Kenneth and Heffernan, Kevin and Kalbassi, Elahe and Lam, Janice and Licht, Daniel and Maillard, Jean and Sun, Anna and Wang, Skyler and Wenzek, Guillaume and Youngblood, Al and Akula, Bapi and Barrault, Loic and Gonzalez, Gabriel Mejia and Hansanti, Prangthip and Hoffman, John and Jarrett, Semarley and Sadagopan, Kaushik Ram and Rowe, Dirk and Spruit, Shannon and Tran, Chau and Andrews, Pierre and Ayan, Necip Fazil and Bhosale, Shruti and Edunov, Sergey and Fan, Angela and Gao, Cynthia and Goswami, Vedanuj and Guzmán, Francisco and Koehn, Philipp and Mourachko, Alexandre and Ropers, Christophe and Saleem, Safiyyah and Schwenk, Holger and Wang, Jeff"),
                 "year": "2024", "title": "Scaling neural machine translation to 200 languages", "venue": "Nature", "volume": "630",
                 "number": "8018", "pages": "841–846", "doi": "10.1038/s41586-024-07335-x", "journal": True},
    "ranathunga2023": {"authors": _authors("Ranathunga, Surangika and Lee, En-Shiun Annie and Prifti Skenduli, Marjana and Shekhar, Ravi and Alam, Mehreen and Kaur, Rishemjit"),
                       "year": "2023", "title": "Neural machine translation for low-resource languages: A survey", "venue": "ACM Computing Surveys",
                       "volume": "55", "number": "11", "pages": "1–37", "doi": "10.1145/3567592", "journal": True},
    "wongso2021": {"authors": _authors("Wongso, Wilson and Setiawan, David Samuel and Suhartono, Derwin"), "year": "2021",
                   "title": "Causal and masked language modeling of Javanese language using transformer-based architectures",
                   "venue": "2021 International Conference on Advanced Computer Science and Information Systems (ICACSIS)",
                   "doi": "10.1109/ICACSIS53237.2021.9631331"},
    "prayoga2026": {"authors": _authors("Prayoga, Mahendra Bayu and Ermawan, Bagas Restya and Fadhillah, Akmal Rafi and Farizi, Mohammad Nizar and Utami, Ema"),
                    "year": "2026", "title": "Evaluating machine translation models and LLMs for Indonesian–Javanese translation across speech levels",
                    "venue": "Journal of Applied Informatics and Computing", "volume": "10", "number": "2", "pages": "1549–1560",
                    "url": "https://jurnal.polibatam.ac.id/index.php/JAIC/article/view/12326", "journal": True},
    "yuliawati2025": {"authors": _authors("Yuliawati, Arlisa and Alfina, Ika and Budi, Indra"), "year": "2025",
                      "title": "From corpus to benchmark: Evaluating pretrained language models for Indonesian-Javanese Krama translation",
                      "venue": "2025 International Conference on Asian Language Processing (IALP)", "pages": "199–204",
                      "doi": "10.1109/IALP68296.2024.11156402"},
    "wibawa2014": {"authors": _authors("Wibawa, Aji Prasetya and Nafalski, Andrew and Mahmudy, Wayan Firdaus"), "year": "2014",
                   "title": "Augmented Javanese speech levels machine translation", "venue": "ECTI Transactions on Computer and Information Technology",
                   "volume": "8", "number": "1", "pages": "45–55", "doi": "10.37936/ecti-cit.201481.54387", "journal": True},
    "lee2023": {"authors": _authors("Lee, Seugnjun and Moon, Hyeonseok and Park, Chanjun and Lim, Heuiseok"), "year": "2023",
                "title": "Data-driven approach for formality-sensitive machine translation: Language-specific handling and synthetic data generation",
                "venue": "arXiv", "url": "https://arxiv.org/abs/2306.14514", "arxiv": "2306.14514"},
    "susanto2023": {"authors": _authors("Susanto, Lucky and Diandaru, Ryandito and Krisnadhi, Adila and Purwarianti, Ayu and Wijaya, Derry"),
                    "year": "2023", "title": "Replicable benchmarking of neural machine translation (NMT) on low-resource local languages in Indonesia",
                    "venue": "arXiv", "url": "https://arxiv.org/abs/2311.00998", "arxiv": "2311.00998"},
}

# Nama pendek kunci ACL -> kunci yang dipakai di content.py
ALIAS = {"cendol": "cahyawijaya-etal-2024-cendol", "indonlg": "cahyawijaya-etal-2021-indonlg", "nusax": "winata-etal-2023-nusax",
         "hwtsc": "wu-etal-2023-improving", "amazon": "zhang-etal-2022-improving-machine", "cocoa": "nadejde-etal-2022-cocoa",
         "mt5": "xue-etal-2021-mt5", "haddow": "haddow-etal-2022-survey", "kocmi": "kocmi-etal-2021-ship", "aji": "aji-etal-2022-one",
         "rippeth": "rippeth-etal-2022-controlling", "iwslt23": "agrawal-etal-2023-findings", "iwslt22": "anastasopoulos-etal-2022-findings",
         "schioppa": "schioppa-etal-2021-controlling", "qad": "fernandes-etal-2022-quality", "discrr": "lee-etal-2021-discriminative",
         "mbr": "freitag-etal-2022-high", "needle": "guerreiro-etal-2023-looking", "selftrain": "jiao-etal-2021-self",
         "dasurvey": "feng-etal-2021-survey", "nusacrowd": "cahyawijaya-etal-2023-nusacrowd", "nusawrites": "cahyawijaya-etal-2023-nusawrites",
         "halluc": "raunak-etal-2021-curious", "sennrich": "sennrich-etal-2016-controlling", "feely": "feely-etal-2019-controlling",
         "bleu": "papineni-etal-2002-bleu", "chrf": "popovic-2017-chrf", "mbart50": "tang-etal-2021-multilingual",
         "farhansyah": "farhansyah-etal-2025-language"}

_ACL = _parse_acl()
REFS = {k: _ACL[v] for k, v in ALIAS.items()}
REFS["iwslt23"] = dict(REFS["iwslt23"], title="Findings of the IWSLT 2023 evaluation campaign")  # BibTeX asli huruf kapital semua
REFS.update(MANUAL)


def _surname(r):
    return r["authors"][0][0]


def _base_cite(r):
    a = r["authors"]
    if a[0][0] == "NLLB Team":
        return "NLLB Team et al."
    if len(a) == 1: return a[0][0]
    if len(a) == 2: return f"{a[0][0]} & {a[1][0]}"
    return f"{a[0][0]} et al."


# sufiks a/b untuk sitasi yang sama penulis-tahun
_groups = {}
for k, r in REFS.items():
    _groups.setdefault((_base_cite(r), r["year"]), []).append(k)
SUFFIX = {}
for (_, _), ks in _groups.items():
    if len(ks) > 1:
        for i, k in enumerate(sorted(ks, key=lambda k: REFS[k]["title"].lower())):
            SUFFIX[k] = "abcdefgh"[i]


def cite(key):
    r = REFS[key]
    return f"{_base_cite(r)}, {r['year']}{SUFFIX.get(key, '')}"


def _fmt_authors(a):
    names = [f"{l}, {i}".strip().rstrip(",") if i else l for l, i in a]
    if len(names) == 1: return names[0]
    if len(names) <= 20: return ", ".join(names[:-1]) + ", & " + names[-1]
    return ", ".join(names[:19]) + ", ... " + names[-1]


def _sentence_case(t):
    # pertahankan kata yang ada huruf kapital di tengah/akronim; kapital setelah titik dua
    words, out, cap = t.split(" "), [], True
    for w in words:
        parts = []
        for j, part in enumerate(w.split("-")):
            core = part.strip(".,:;?!()")
            keep = any(c.isupper() for c in core[1:]) or (core.isupper() and len(core) > 1) or core in KEEP
            if keep: parts.append(part)
            elif cap and j == 0: parts.append(part[:1].upper() + part[1:].lower())
            else: parts.append(part.lower())
        out.append("-".join(parts))
        cap = w.endswith(":") or w.endswith("?")
    s = " ".join(out)
    return s


KEEP = {"Indonesian", "Indonesia", "Javanese", "Japanese", "English", "English-to-Japanese", "Krama", "Bayes", "Indonesian-Javanese",
        "Indonesian–Javanese", "IWSLT", "NLP", "LLMs", "NMT"}


def entry(key):
    """-> list of (teks, italic)"""
    r = REFS[key]; seg = []
    seg.append((f"{_fmt_authors(r['authors'])} ({r['year']}{SUFFIX.get(key, '')}). ", False))
    title = _sentence_case(r["title"])
    if r.get("journal"):
        seg.append((f"{title}{'' if title.endswith('?') else '.'} ", False))
        seg.append((r["venue"], True))
        if r.get("volume"):
            seg.append((", ", False)); seg.append((r["volume"], True))
            if r.get("number"): seg.append((f"({r['number']})", False))
        if r.get("pages"): seg.append((f", {r['pages']}", False))
        seg.append((". ", False))
    elif r.get("arxiv"):
        seg.append((title, True)); seg.append((f" (arXiv:{r['arxiv']}). arXiv. ", False))
    else:
        seg.append((f"{title}{'' if title.endswith('?') else '.'} In ", False)); seg.append((r["venue"], True))
        seg.append((f" (pp. {r['pages']}). " if r.get("pages") else ". ", False))
    if r.get("doi"): seg.append((f"https://doi.org/{r['doi']}", False))
    elif r.get("url"): seg.append((r["url"], False))
    return seg


def ordered_keys(used):
    return sorted(used, key=lambda k: (_surname(REFS[k]).lower(), REFS[k]["year"], SUFFIX.get(k, "")))


if __name__ == "__main__":
    for k in ordered_keys(REFS):
        print(cite(k), "|", "".join(t for t, _ in entry(k))[:200])
    print(len(REFS), "referensi;", sum(int(r["year"]) < 2021 for r in REFS.values()), "sebelum 2021")
