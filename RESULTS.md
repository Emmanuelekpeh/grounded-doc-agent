# RESULTS — first live eval run

**Run date:** 18 November 2026
**Corpus:** *I-JEPA* paper (Assran et al.), 83 chunks ingested — 49 text, 23 captions, 2 equations, 9 references
**Model:** `qwen2.5:7b` (Q4_K_M, 7.6B), via local Ollama, **100% CPU**, `temperature 0`, `num_ctx 8192`
**Retrieval:** **BM25 only** — `sentence-transformers` was not installed, so the hybrid path (BM25 + `bge-small` embeddings + reciprocal-rank fusion) was **not** exercised. This matters: it is the weaker configuration.
**Questions:** 10, from `questions_ijepa.txt` (7 real questions, 2 out-of-scope, 1 nonsense control)
**Runtime:** ~27 minutes, ~160–300 s per question on CPU

Reproduce:

```bash
PAPERWORLD_PROVIDER=ollama PAPERWORLD_MODEL=qwen2.5:7b \
python -m paperworld ingest ijepa.pdf --name ijepa
python -m paperworld eval ijepa questions_ijepa.txt
```

---

## Headline

| Metric | Result |
|---|---|
| Questions answered | 10 / 10 |
| Quotes emitted | 10 |
| **Quotes verified against source text** | **10 / 10 (100%)** |
| Answered with a supported claim | 6 / 10 |
| Declared NOT SUPPORTED | 4 / 10 |
| **Answers that were behaviourally wrong** | **2 / 10** |

**The number that matters is the last row, not the third.** Every quote it produced was real — the verification layer never
let an invented quotation through. But two answers were still wrong, in opposite directions, and neither failure would
have been caught by quote-verification alone. That is the finding.

## Per question

| # | Question | Supported | Quotes verified | Verdict |
|---|---|---|---|---|
| 1 | What problem with existing self-supervised methods does it address? | ✅ | 1/1 | correct |
| 2 | What is the predictor and what does it take as input? | ✅ | 1/1 | correct |
| 3 | How is the target encoder updated? | ✅ | 1/1 | correct |
| 4 | **What is the training objective?** | ❌ | 1/1 | **wrong — answered from the wrong section** |
| 5 | How are context and target blocks chosen? | ✅ | 2/2 | correct |
| 6 | Predicting in representation space vs pixel space | ✅ | 2/2 | correct |
| 7 | What ablations are reported? | ✅ | 2/2 | correct |
| 8 | Does the paper say anything about video? | ❌ | 0/0 | correct refusal (0 chunks contain "video") |
| 9 | **What GPU does it train on, and for how many hours?** | ❌ | 0/0 | **wrong — false refusal; answer is in the paper** |
| 10 | Who is the senior author's favourite colour? | ❌ | 0/0 | correct refusal |

## The two failures, with evidence

### F1 — Right words, wrong section (question 4)

The system returned this, labelled `INFERENCE`, with a verified quote:

> The training objective is to minimize the loss function ‖h_ω(x̂, s_x) − ϵ‖ …
> ✓ verified — §E RCDM Visualizations (p.16): *"The decoder objective is to minimize the loss function ‖h_ω(x̂, s_x) − ϵ‖."*

That quote is genuinely in the paper and genuinely verified. It is also **the auxiliary decoder used only to visualise
representations**, not the I-JEPA training objective, which is the L2 distance between predicted and target
representations in §3 Method. BM25 matched "loss function" and "objective" and surfaced the appendix; the model then
answered confidently from it.

**Why this is the important failure:** quote verification proves the words exist. It cannot prove they answer the
question. The system correctly marked the answer `supported=False`, so the label did its job — but a reader skimming
the rendered output would see a verified citation and a confident sentence.

### F2 — False refusal (question 9)

The system returned NOT SUPPORTED. The paper does answer this, in two chunks:

- `c0001`: *"we train a ViT-Huge/14 on ImageNet using **16 A100 GPUs in under 72 hours**"*
- `c0007`: *"Pretraining a ViT-H/14 on ImageNet requires **less than 1200 GPU hours**"*

So the content was in the corpus and the question was fair. This is a **coverage/retrieval miss** — the chunks were not
surfaced to the model, or were surfaced and judged insufficient. It fails in the safe direction (it refused rather than
guessing), which is the right default, but it is still a miss and it would erode a user's trust in the opposite way: the
system looks like it does not know things it does.

## What the numbers do and do not say

- **Do** say: the verification mechanism is functioning — 10 of 10 quotes traced to the source, and the two
  unanswerable questions were correctly declined rather than hallucinated.
- **Do** say: BM25-only retrieval with a 7B local model reaches 6/10 fully correct on a fair question set.
- **Do not** say: hybrid retrieval gives the same number (untested — embeddings were not installed).
- **Do not** say: these are hand-verified for all ten. I hand-checked the four flagged cases and question 2; questions
  1, 3, 5, 6, 7 are confirmed by the harness's own mechanism but not manually re-read against the paper.

## What I would fix next, in order

1. **Section-aware retrieval for method questions.** F1 is caused by BM25 matching appendix prose to a method question.
   Cheap fix: boost §3 Method for "objective/loss/how" questions, or exclude appendix sections unless the question names
   one. This is a retrieval-ranking change, not a model change, and it is the single biggest win available.
2. **Surface a "matched section" warning.** If the retrieved chunks all come from an appendix while the question is
   about method, show that next to the citation. Cheap, and it turns F1 from a wrong answer into a visible caveat.
3. **Second-pass retrieval on refusal.** F2's fix: when the model returns NOT SUPPORTED, re-query with keywords lifted
   from the question ("GPU", "training time") before committing to a refusal. One extra call, not ten.
4. **Re-run with hybrid retrieval** once `sentence-transformers` is installed, and record whether F1/F2 change. This is
   the experiment the current numbers are missing.

## Comparison point for the next run

The number to beat is **6/10 fully correct, 10/10 quotes verified, 2/10 behaviourally wrong, BM25-only, qwen2.5:7b**.
Any change to retrieval or prompting should be judged against this table, and the run should be re-executed with the
same corpus and question set.
