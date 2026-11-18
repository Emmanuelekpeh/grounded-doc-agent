"""Grounded question answering with machine-checked citations.

The model must return JSON: segments labeled SOURCE / INFERENCE / INTERPRETATION,
each SOURCE or INFERENCE segment carrying verbatim quotes tied to chunk ids.
We then check every quote against the chunk text. A SOURCE segment whose quotes
do not verify is downgraded to UNVERIFIED and shown that way.

Phase 0 is grounded-only: EXTERNAL and SPECULATION labels are reserved for the
exploratory-RAG phase and are not allowed here.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import llm

MODES = {
    "normal": "Clear, direct prose.",
    "ste": ("Controlled language, about 80% of ASD-STE100: short sentences (about 20 words max), "
            "one idea per sentence, simple present tense, active voice, one word per meaning, no idioms."),
    "intuitive": ("Give a concrete example or analogy, but put analogies in INTERPRETATION segments only. "
                  "Every factual claim still needs a SOURCE or INFERENCE segment."),
    "math": "Foreground equations and notation. Define each symbol only as the passages define it.",
    "implementation": ("Describe it as data flow: inputs, steps in order, outputs. Give tensor shapes "
                       "only if the passages state them."),
    "critical": "Emphasize assumptions, limitations, and what the passages do not establish.",
}

SYSTEM = """You answer questions about ONE research paper using only the numbered passages provided.

Rules:
1. Use only the passages. If they do not answer the question, set "supported": false, say what is missing in "gaps", and do not guess.
2. Return ONLY a JSON object, no prose outside it:
{
 "supported": true,
 "segments": [
  {"label": "SOURCE|INFERENCE|INTERPRETATION",
   "text": "one to three sentences",
   "evidence": [{"chunk": "c0012", "quote": "verbatim words copied from that passage, at most 25 words"}]}
 ],
 "gaps": "what the passages do not cover, or empty string"
}
3. SOURCE = the paper explicitly says it. Needs at least one evidence quote, copied exactly.
4. INFERENCE = follows from what the paper says. Quote the premises it follows from.
5. INTERPRETATION = a helpful framing of your own. No evidence needed; keep it short and clearly secondary.
6. Never use outside knowledge about the paper, its authors, or later work. Do not invent equation numbers, section numbers, or figure contents.
7. Quotes must be contiguous text from a single passage. Do not paraphrase inside quotes."""


@dataclass
class Evidence:
    chunk: str
    quote: str
    verified: bool = False
    where: str = ""


@dataclass
class Segment:
    label: str
    text: str
    evidence: list[Evidence] = field(default_factory=list)
    status: str = ""   # "verified" | "unverified" | "n/a"


@dataclass
class Answer:
    supported: bool
    segments: list[Segment]
    gaps: str
    raw: str = ""
    retrieved: list[str] = field(default_factory=list)

    def stats(self) -> dict:
        ev = [e for s in self.segments for e in s.evidence]
        ok = sum(e.verified for e in ev)
        return {"quotes": len(ev), "verified": ok,
                "rate": (ok / len(ev)) if ev else None,
                "unverified_segments": sum(s.status == "unverified" for s in self.segments)}


def _squash(s: str) -> str:
    """Case/punctuation/whitespace-insensitive form for quote matching (robust to PDF hyphenation)."""
    return re.sub(r"[^a-z0-9]+", "", s.lower())


def verify_quote(quote: str, chunk_text: str) -> bool:
    q = _squash(quote)
    return len(q) >= 12 and q in _squash(chunk_text)


def where(c: dict) -> str:
    sec = f"\u00a7{c['section_id']} " if c["section_id"] not in ("", "0") else ""
    lab = f"{c['label']}, " if c["label"] else ""
    return f"{sec}{c['section_title']} ({lab}p.{c['page_start']})"


def format_passages(chunks: list[dict]) -> str:
    return "\n\n".join(f"[{c['id']}] ({c['kind']}; {where(c)})\n{c['text']}" for c in chunks)


def build_user_message(question: str, passages: list[dict], mode: str, ctx: dict) -> str:
    pause = []
    if ctx.get("section"):
        pause.append(f"section {ctx['section']}")
    if ctx.get("concept"):
        pause.append(f"concept: {ctx['concept']}")
    return (f"Learner is paused at: {', '.join(pause) or 'unspecified'}.\n"
            f"Explanation style: {MODES.get(mode, MODES['normal'])}\n\n"
            f"PASSAGES\n{format_passages(passages)}\n\nQUESTION\n{question}")


def parse_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end == -1:
        raise ValueError("model did not return JSON")
    return json.loads(raw[start:end + 1])


def verify(data: dict, by_id: dict[str, dict], allowed: set[str], raw: str, retrieved: list[str]) -> Answer:
    segs = []
    for s in data.get("segments", []):
        label = str(s.get("label", "")).upper()
        if label not in ("SOURCE", "INFERENCE", "INTERPRETATION"):
            label = "INTERPRETATION"
        evs = []
        for e in s.get("evidence", []) or []:
            cid, quote = str(e.get("chunk", "")), str(e.get("quote", ""))
            c = by_id.get(cid)
            ok = bool(c) and cid in allowed and verify_quote(quote, c["text"])
            evs.append(Evidence(cid, quote, ok, where(c) if c else "unknown chunk"))
        if label == "INTERPRETATION":
            status = "n/a"
        elif evs and all(e.verified for e in evs):
            status = "verified"
        else:
            status = "unverified"
            label = "UNVERIFIED"
        segs.append(Segment(label, str(s.get("text", "")), evs, status))
    return Answer(bool(data.get("supported", True)), segs, str(data.get("gaps", "")), raw, retrieved)


def answer_question(question: str, index, ctx: dict | None = None, mode: str = "normal", k: int = 6) -> Answer:
    ctx = ctx or {}
    query = question
    if len(question.split()) < 5:  # "wait what" -> add the pause context to retrieval
        query = " ".join(x for x in (question, ctx.get("concept", ""), ctx.get("section_title", "")) if x)
    passages = index.search(query, k=k, section=ctx.get("section", ""))
    raw = llm.complete(SYSTEM, build_user_message(question, passages, mode, ctx))
    data = parse_json(raw)
    return verify(data, index.by_id, {p["id"] for p in passages}, raw, [p["id"] for p in passages])


def render(ans: Answer) -> str:
    out = []
    if not ans.supported:
        out.append("NOT SUPPORTED BY THE PAPER (as retrieved).")
    for s in ans.segments:
        out.append(f"[{s.label}] {s.text}")
        for e in s.evidence:
            mark = "\u2713" if e.verified else "\u2717"
            out.append(f"    {mark} {e.where} {e.chunk}: \u201c{e.quote}\u201d")
    if ans.gaps:
        out.append(f"GAPS: {ans.gaps}")
    st = ans.stats()
    if st["quotes"]:
        out.append(f"(quotes verified: {st['verified']}/{st['quotes']})")
    return "\n".join(out)
