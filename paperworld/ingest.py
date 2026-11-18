"""PDF -> structured chunks with section, page, and kind metadata.

Phase 0 goals: every chunk can be cited by id, section and page. Equations are
only *detected* (PDF math text is usually fragmented); proper equation parsing
is a later upgrade (see README, "Known limits").
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import dataclass, asdict, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

LIGATURES = {"\ufb01": "fi", "\ufb02": "fl", "\ufb00": "ff", "\ufb03": "ffi",
             "\ufb04": "ffl", "\u00ad": "", "\u2019": "'", "\u201c": '"', "\u201d": '"'}

# Numbered ("3.2 Predictor") or lettered appendix ("A. Details", "A.1. Pretraining") headings.
# A bare letter needs a following period so "A Simple Baseline" is not read as appendix "A".
NUM_HEAD = re.compile(r"^(\d+(?:\.\d+){0,2}|[A-Z]\.\d+(?:\.\d+)?|[A-Z](?=\.\s))\.?\s+([A-Z][^\n]{1,100})$")
NAMED_HEAD = {"abstract", "references", "bibliography", "acknowledgements",
              "acknowledgments", "appendix"}
CAPTION = re.compile(r"^(Figure|Fig\.|Table)\s+(\d+)\s*[:.]", re.I)
EQ_NUMBER = re.compile(r"\((\d{1,2})\)\s*$")
MATH_CHARS = set("=\u2211\u220f\u222b\u2225\u2016\u03b8\u0398\u03c6\u03c8\u03bb\u03bc\u03c3\u2208"
                 "\u2264\u2265\u2248\u2192\u22a4\u02c6\u00af~|^_{}")

TARGET_WORDS = 200   # chunk size target
MAX_WORDS = 280


def clean(text: str) -> str:
    for k, v in LIGATURES.items():
        text = text.replace(k, v)
    text = re.sub(r"(\w)-\s+(\w)", r"\1\2", text)  # join line-break hyphenation
    return re.sub(r"\s+", " ", text).strip()


@dataclass
class Line:
    text: str
    size: float
    bold: bool


@dataclass
class Chunk:
    id: str
    paper: str
    section_id: str
    section_title: str
    page_start: int
    page_end: int
    kind: str            # text | caption | equation | references
    label: str           # e.g. "Figure 2", "Eq. 4", ""
    text: str
    images: list = field(default_factory=list)
    page_image: str = ""


def _page_lines(page) -> list[list[Line]]:
    """Return one list of Line per text block, in PDF stream (reading) order."""
    blocks = []
    for b in page.get_text("dict")["blocks"]:
        if b["type"] != 0:
            continue
        lines = []
        for l in b["lines"]:
            spans = [s for s in l["spans"] if s["text"].strip()]
            if not spans:
                continue
            txt = "".join(s["text"] for s in l["spans"])
            size = statistics.median(s["size"] for s in spans)
            bold = all(bool(s["flags"] & 16) or "bold" in s["font"].lower()
                       or s["font"].lower().endswith(("-bd", "bx10", "bx12")) for s in spans)
            lines.append(Line(txt, size, bold))
        if lines:
            blocks.append(lines)
    return blocks


def _body_size(all_blocks) -> float:
    weights: Counter = Counter()
    for _, blocks in all_blocks:
        for lines in blocks:
            for ln in lines:
                weights[round(ln.size, 1)] += len(ln.text)
    return weights.most_common(1)[0][0] if weights else 10.0


def _repeated_keys(all_blocks, n_pages: int) -> set[str]:
    """Headers/footers: short blocks whose digit-normalized text repeats on many pages."""
    if n_pages < 4:
        return set()
    seen: dict[str, set[int]] = {}
    for pno, blocks in all_blocks:
        for lines in blocks:
            t = clean(" ".join(l.text for l in lines))
            if len(t) < 80:
                seen.setdefault(re.sub(r"\d+", "#", t), set()).add(pno)
    return {k for k, pages in seen.items() if len(pages) >= 0.4 * n_pages}


def _is_heading(line: Line, body: float) -> tuple[str, str] | None:
    t = clean(line.text)
    if len(t) > 110:
        return None
    styled = line.bold or line.size > body * 1.05
    m = NUM_HEAD.match(t)
    if m and styled:
        return m.group(1), m.group(2).strip()
    if t.lower().rstrip(":") in NAMED_HEAD and styled:
        return "", t.rstrip(":")
    return None


def _is_caption_line(line: Line, first_in_block: bool, body: float) -> bool:
    """A 'Figure 3.' at the start of a wrapped body line is a cross-reference, not a caption.
    Accept the match if it opens its block, or if it is set smaller than body text (captions usually are)."""
    if not CAPTION.match(clean(line.text)):
        return False
    return first_in_block or line.size < body * 0.97


def _looks_like_equation(t: str) -> bool:
    if len(t) > 200 or len(t) < 3:
        return False
    math = sum(c in MATH_CHARS for c in t)
    if EQ_NUMBER.search(t) and math >= 1:
        return True
    return math >= max(4, 0.15 * len(t))


def _split_long(text: str) -> list[str]:
    words = text.split()
    if len(words) <= MAX_WORDS:
        return [text]
    sentences = re.split(r"(?<=[.!?])\s+", text)
    out, buf = [], []
    for s in sentences:
        if sum(len(x.split()) for x in buf) + len(s.split()) > TARGET_WORDS and buf:
            out.append(" ".join(buf))
            buf = []
        buf.append(s)
    if buf:
        out.append(" ".join(buf))
    return out


def parse_pdf(pdf_path: Path, paper: str) -> list[Chunk]:
    doc = fitz.open(pdf_path)
    all_blocks = [(i + 1, _page_lines(p)) for i, p in enumerate(doc)]
    body = _body_size(all_blocks)
    drop = _repeated_keys(all_blocks, len(doc))

    chunks: list[Chunk] = []
    sec_id, sec_title = "0", "Front matter"
    buf: list[str] = []
    buf_pages: list[int] = []

    def flush():
        nonlocal buf, buf_pages
        if not buf:
            return
        kind = "references" if sec_title.lower().startswith(("references", "bibliography")) else "text"
        for piece in _split_long(clean(" ".join(buf))):
            chunks.append(Chunk(f"c{len(chunks):04d}", paper, sec_id, sec_title,
                                min(buf_pages), max(buf_pages), kind, "", piece))
        buf, buf_pages = [], []

    def handle_segment(whole: str, pno: int):
        whole = clean(whole)
        if not whole:
            return
        cap = CAPTION.match(whole)
        if cap:
            flush()
            label = f"{'Table' if cap.group(1).lower() == 'table' else 'Figure'} {cap.group(2)}"
            chunks.append(Chunk(f"c{len(chunks):04d}", paper, sec_id, sec_title,
                                pno, pno, "caption", label, whole))
            return
        if _looks_like_equation(whole):
            flush()
            n = EQ_NUMBER.search(whole)
            chunks.append(Chunk(f"c{len(chunks):04d}", paper, sec_id, sec_title, pno, pno,
                                "equation", f"Eq. {n.group(1)}" if n else "Eq. ?", whole))
            return
        words = sum(len(x.split()) for x in buf)
        if buf and words + len(whole.split()) > MAX_WORDS and words >= 80:
            flush()
        buf.append(whole)
        buf_pages.append(pno)

    for pno, blocks in all_blocks:
        for lines in blocks:
            whole_block = clean(" ".join(l.text for l in lines))
            if not whole_block or re.fullmatch(r"\d{1,3}", whole_block) \
                    or re.sub(r"\d+", "#", whole_block) in drop:
                continue
            # Headings and captions can sit mid-block, so split the block at those lines.
            seg: list[str] = []
            for li, ln in enumerate(lines):
                head = _is_heading(ln, body)
                if head:
                    handle_segment(" ".join(seg), pno)
                    seg = []
                    flush()
                    sec_id, sec_title = head
                elif _is_caption_line(ln, li == 0, body):
                    handle_segment(" ".join(seg), pno)
                    seg = [ln.text]
                else:
                    seg.append(ln.text)
            handle_segment(" ".join(seg), pno)
    flush()
    return chunks


def extract_figures(pdf_path: Path, chunks: list[Chunk], out_dir: Path) -> None:
    """Save embedded images and a page render for every page that has a caption."""
    figs = out_dir / "figs"
    pages = out_dir / "pages"
    figs.mkdir(parents=True, exist_ok=True)
    pages.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    by_page: dict[int, list[Chunk]] = {}
    for c in chunks:
        if c.kind == "caption":
            by_page.setdefault(c.page_start, []).append(c)
    for pno, caps in by_page.items():
        page = doc[pno - 1]
        saved = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                pix = fitz.Pixmap(doc, xref)
                if pix.width < 150 or pix.height < 150:
                    continue
                if pix.n - pix.alpha >= 4:
                    pix = fitz.Pixmap(fitz.csRGB, pix)
                fp = figs / f"p{pno}_x{xref}.png"
                pix.save(fp)
                saved.append(str(fp.relative_to(out_dir)))
            except Exception:
                continue
        pr = pages / f"p{pno}.png"
        page.get_pixmap(matrix=fitz.Matrix(1.5, 1.5)).save(pr)
        for c in caps:
            c.images = saved
            c.page_image = str(pr.relative_to(out_dir))


def ingest(pdf_path: str | Path, name: str, store_root: str | Path = "store") -> Path:
    pdf_path = Path(pdf_path)
    out = Path(store_root) / name
    out.mkdir(parents=True, exist_ok=True)
    chunks = parse_pdf(pdf_path, name)
    extract_figures(pdf_path, chunks, out)
    (out / "chunks.json").write_text(json.dumps([asdict(c) for c in chunks], indent=1), encoding="utf-8")
    kinds = Counter(c.kind for c in chunks)
    (out / "meta.json").write_text(json.dumps(
        {"name": name, "source_pdf": str(pdf_path), "n_chunks": len(chunks), "kinds": dict(kinds)}, indent=1),
        encoding="utf-8")
    (out / "embeddings.npy").unlink(missing_ok=True)
    return out


def load_chunks(name: str, store_root: str | Path = "store") -> list[dict]:
    return json.loads((Path(store_root) / name / "chunks.json").read_text(encoding="utf-8"))
