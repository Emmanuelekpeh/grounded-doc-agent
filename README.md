# Paper World — a grounded document agent

Ask questions about a dense technical paper and get answers whose **quoted evidence is machine-checked against the
source text** — or an explicit *not supported* instead of a plausible guess.

The point of the project is the governance layer, not the chat: an LLM answering questions about a *controlled,
validated document set* must be checkable. So every answer segment is labelled by how it is grounded, and every quote
is verified back against the passage it claims to come from.

```text
SOURCE          the paper says it — quotes verified against the paper text
INFERENCE       follows from quoted premises — quotes verified
INTERPRETATION  the model's own framing — no evidence required, shown as secondary
UNVERIFIED      model claimed SOURCE/INFERENCE but a quote was not found in the cited passage
```

Each quote gets a `✓` / `✗`. If retrieval does not surface supporting passages, the answer is
**NOT SUPPORTED BY THE PAPER** rather than a fabricated one.

## Why this exists

Most "chat with your PDF" tools optimise for fluent answers. In a regulated or validated-content setting the failure
mode that matters is a confident claim that is not actually in the source. This project treats the corpus as a
governed knowledge source and makes grounding auditable: retrieval, quoting, verification and evaluation are separate,
inspectable steps.

## How it works

```
PDF  ->  ingest (sections, captions, equation candidates, figure images)
     ->  index  (BM25 + optional local embeddings, reciprocal-rank fusion, section boost)
     ->  ask    (grounded prompt -> structured JSON answer -> quote verification -> rendering)
     ->  eval   (question set -> verified rate + human review file)
```

- **Hybrid retrieval**: BM25 by default; if `sentence-transformers` is present it adds local embeddings
  (`BAAI/bge-small-en-v1.5`) and fuses the rankings. No vector database, no external service.
- **Context-aware question expansion**: a short or vague question is expanded with the current section and concept
  before retrieval, so "wait what?" resolves to something searchable.
- **Verified quoting, honestly scoped**: a verified quote proves the words exist in that passage — it does not prove
  they support the claim attached to them. `eval` writes `review.md` with each claim, quote and passage so a human
  judges support. That distinction is deliberate and documented.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=...          # only needed for LLM answers
```

```bash
python -m paperworld ingest ijepa.pdf --name ijepa   # you supply the PDF
python -m paperworld outline ijepa                   # check the section structure first
python -m paperworld search ijepa "target encoder update"   # retrieval only, no LLM
python -m paperworld ask ijepa "What is the predictor's input?" --section 3 --mode ste
python -m paperworld eval ijepa questions_ijepa.txt
```

`--no-llm` on `ask`/`pause` shows retrieved passages without calling a model.

## Evaluation

`eval` runs a question set and reports a **verified rate**, writing a human-review file. The question set
(`questions_ijepa.txt`) deliberately includes off-topic and not-in-the-paper questions — the correct behaviour there
is NOT SUPPORTED, and that is what the eval checks.

## Status — Phase 0

Working: ingest, outline, hybrid retrieval, grounded answers with quote verification, pause-and-ask REPL, eval harness,
and an offline test suite that uses a synthetic PDF and a fake LLM (so the tests do not require an API key).
**11 tests pass** (`python -m pytest -q`).

Not done yet, and stated rather than hidden:

- **The live eval has not been run against a real model in this repository.** The Anthropic path and embedding model
  were not exercised in the build environment; tests cover the offline pipeline only. Results table lands after the
  first real run.
- Equations are detected, not parsed (PDF math text is fragmented). Figures are captured as images; contents are not read.
- Heading detection is heuristic; appendix subsections like `A.1` are not recognised.
- Two-column reading order depends on PDF stream order — usually right for LaTeX, not guaranteed.

## Known limits

- Hyphenation joining can merge genuine hyphenated compounds split across lines.
- Vector figures are only visible via the page render.
- Phase 1 (concept model, generated diagrams, ladder questions) is designed but not built.

## Layout

```
paperworld/ingest.py   PDF -> chunks (sections, captions, equation candidates, figure images)
paperworld/index.py    BM25 + optional embeddings, reciprocal rank fusion, section boost
paperworld/ask.py      grounded prompt, JSON answer, quote verification, rendering
paperworld/llm.py      single place to change the model or provider
paperworld/cli.py      ingest | outline | search | ask | pause | eval
tests/                 python -m pytest -q
```

See `Paper World An Interactive Environment for Learning Dense Technical Papers.md` for the full design.

## Phase 0 acceptance checklist

- [ ] `outline` shows the real section structure of the target paper.
- [ ] `eval` on the question set: check `verified_rate` and read `review.md`.
- [ ] Off-topic and not-in-the-paper questions return NOT SUPPORTED rather than an answer.
- [ ] A vague question after `:at 3` / `:concept ...` retrieves the right passage.

## Next (Phase 1)

A hand-checked concept model for the ingested paper (components, flows, claims with source spans), one generated
diagram from it, and ladder levels 1–2 plus five retrieval questions. Chunk and section ids from Phase 0 are what the
concept model's source spans will point to.

## Licence

MIT for the code. Papers you ingest are your responsibility — none are included in this repository.
