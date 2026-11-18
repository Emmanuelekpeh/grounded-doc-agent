"""Offline tests (no network, no API key). Builds a tiny synthetic two-page paper."""
import json
import os

import fitz
import pytest

os.environ["PAPERWORLD_EMBED"] = "none"

from paperworld import ask as askmod
from paperworld.index import Index
from paperworld.ingest import ingest, load_chunks

BODY = 10
HEAD = 14


def _write(page, y, text, size=BODY, font="helv"):
    page.insert_text((72, y), text, fontsize=size, fontname=font)
    return y + size * 1.6 + (8 if size > BODY else 0)


@pytest.fixture()
def paper(tmp_path):
    doc = fitz.open()
    p1 = doc.new_page()
    y = 72
    y = _write(p1, y, "Abstract", HEAD, "hebo")
    y = _write(p1, y, "We study joint embedding methods that predict in representation space.")
    y += 10
    y = _write(p1, y, "1 Introduction", HEAD, "hebo")
    for t in ["Pixel reconstruction forces the model to spend capacity on unpredictable detail.",
              "Our predictor maps a context embedding to the embedding of a masked target block."]:
        y = _write(p1, y, t)
    y += 10
    y = _write(p1, y, "3.2 Predictor", HEAD, "hebo")
    for t in ["The predictor takes the context repre-", "sentation and positional tokens as input."]:
        y = _write(p1, y, t)
    y = _write(p1, y, "L = || s_pred - s_target ||^2   (4)")
    p2 = doc.new_page()
    y = 72
    y = _write(p2, y, "Figure 2: Architecture with a context encoder, target encoder and predictor.")
    y = _write(p2, y, "4 Experiments", HEAD, "hebo")
    y = _write(p2, y, "The target encoder is an exponential moving average of the context encoder.")
    y = _write(p2, y, "References", HEAD, "hebo")
    y = _write(p2, y, "[1] A. Author. A paper about nothing. 2020.")
    pdf = tmp_path / "toy.pdf"
    doc.save(pdf)
    ingest(pdf, "toy", tmp_path / "store")
    return tmp_path / "store", "toy"


def test_sections_and_kinds(paper):
    store, name = paper
    chunks = load_chunks(name, store)
    secs = {(c["section_id"], c["section_title"]) for c in chunks}
    assert ("3.2", "Predictor") in secs
    assert ("1", "Introduction") in secs
    kinds = {c["kind"] for c in chunks}
    assert {"text", "caption", "equation", "references"} <= kinds
    cap = next(c for c in chunks if c["kind"] == "caption")
    assert cap["label"] == "Figure 2" and cap["page_start"] == 2 and cap["page_image"]
    eq = next(c for c in chunks if c["kind"] == "equation")
    assert eq["label"] == "Eq. 4" and eq["section_id"] == "3.2"


def test_hyphenation_joined(paper):
    store, name = paper
    pred = next(c for c in load_chunks(name, store) if c["section_id"] == "3.2" and c["kind"] == "text")
    assert "representation" in pred["text"]


def test_references_excluded_from_search(paper):
    store, name = paper
    idx = Index(load_chunks(name, store), store / name)
    hits = idx.search("A paper about nothing", k=5)
    assert all(h["kind"] != "references" for h in hits)


def test_retrieval_finds_target_encoder(paper):
    store, name = paper
    idx = Index(load_chunks(name, store), store / name)
    hits = idx.search("how is the target encoder updated", k=3)
    assert any("moving average" in h["text"] for h in hits)


def test_section_boost_resolves_vague_question(paper):
    store, name = paper
    idx = Index(load_chunks(name, store), store / name)
    hits = idx.search("input", k=2, section="3.2")
    assert hits[0]["section_id"] == "3.2"


def test_quote_verification_tolerates_pdf_artifacts_but_rejects_fabrication():
    chunk = "The predictor takes the context repre-sentation and positional tokens as input."
    assert askmod.verify_quote("context representation and positional tokens", chunk)
    assert not askmod.verify_quote("the predictor outputs a pixel reconstruction", chunk)
    assert not askmod.verify_quote("input", chunk)  # too short to count


def test_answer_pipeline_with_fake_llm(paper, monkeypatch):
    store, name = paper
    idx = Index(load_chunks(name, store), store / name)
    pred = next(c for c in idx.chunks if c["section_id"] == "3.2" and c["kind"] == "text")
    target = next(c for c in idx.chunks if "moving average" in c["text"])
    fake = {
        "supported": True,
        "segments": [
            {"label": "SOURCE", "text": "The predictor takes context and positions.",
             "evidence": [{"chunk": pred["id"], "quote": "context representation and positional tokens as input"}]},
            {"label": "SOURCE", "text": "The target encoder is trained by backprop.",   # fabricated
             "evidence": [{"chunk": target["id"], "quote": "the target encoder is trained by backpropagation"}]},
            {"label": "INTERPRETATION", "text": "Think of it as a guess about meaning, not pixels."},
        ],
        "gaps": "",
    }
    monkeypatch.setattr(askmod.llm, "complete", lambda *a, **k: "```json\n" + json.dumps(fake) + "\n```")
    ans = askmod.answer_question("what does the predictor take as input", idx, {"section": "3.2"})
    labels = [s.label for s in ans.segments]
    assert labels == ["SOURCE", "UNVERIFIED", "INTERPRETATION"]
    assert ans.stats()["verified"] == 1 and ans.stats()["quotes"] == 2
    assert "\u2717" in askmod.render(ans)


def test_quote_from_unretrieved_chunk_is_rejected(paper, monkeypatch):
    store, name = paper
    idx = Index(load_chunks(name, store), store / name)
    fake = {"supported": True, "segments": [{"label": "SOURCE", "text": "x",
            "evidence": [{"chunk": "c9999", "quote": "something that sounds plausible here"}]}], "gaps": ""}
    monkeypatch.setattr(askmod.llm, "complete", lambda *a, **k: json.dumps(fake))
    ans = askmod.answer_question("anything", idx, {})
    assert ans.segments[0].label == "UNVERIFIED"


# ---- regressions found by running on the real I-JEPA PDF ----

def test_appendix_headings_after_references_are_searchable(tmp_path):
    doc = fitz.open()
    p = doc.new_page()
    y = _write(p, 72, "1 Introduction", HEAD, "hebo")
    y = _write(p, y, "We predict representations of masked image blocks.")
    y = _write(p, y, "References", HEAD, "hebo")
    y = _write(p, y, "[1] A. Author. A paper about nothing. 2020.")
    y = _write(p, y, "A. Implementation Details", HEAD, "hebo")
    y = _write(p, y, "Pretraining uses a batch size of 2048 on sixteen GPUs.")
    y = _write(p, y, "A.1. Pretraining", HEAD, "hebo")
    y = _write(p, y, "The learning rate is warmed up for fifteen epochs.")
    pdf = tmp_path / "app.pdf"
    doc.save(pdf)
    ingest(pdf, "app", tmp_path / "store")
    chunks = load_chunks("app", tmp_path / "store")
    secs = {(c["section_id"], c["section_title"]) for c in chunks}
    assert ("A", "Implementation Details") in secs and ("A.1", "Pretraining") in secs
    gpu = next(c for c in chunks if "sixteen GPUs" in c["text"])
    assert gpu["kind"] == "text"
    idx = Index(chunks, tmp_path / "store" / "app")
    assert any("sixteen GPUs" in h["text"] for h in idx.search("how many GPUs for pretraining", k=3))


def test_bare_letter_title_is_not_an_appendix_heading():
    from paperworld.ingest import Line, _is_heading
    assert _is_heading(Line("A Simple Baseline", 14, True), 10) is None
    assert _is_heading(Line("A. Implementation Details", 12, True), 10) == ("A", "Implementation Details")
    assert _is_heading(Line("A.1. Pretraining", 11, True), 10) == ("A.1", "Pretraining")


def test_crossref_at_line_start_is_not_a_caption():
    from paperworld.ingest import Line, _is_caption_line
    body = 10.0
    assert not _is_caption_line(Line("Figure 3. The overall objective is as follows", 9.96, False), False, body)
    assert _is_caption_line(Line("Figure 3. I-JEPA. The architecture uses", 8.97, False), False, body)
    assert _is_caption_line(Line("Figure 3. I-JEPA. The architecture uses", 10.0, False), True, body)
