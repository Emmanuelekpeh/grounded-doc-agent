# Paper World: An Interactive Environment for Learning Dense Technical Papers

Draft 1 · October 9, 2026 · Consolidated and tightened from a ChatGPT conversation (Oct 2–5, 2026). Evidence notes in the bibliography say which sources were re-opened in this pass and which were not.

## 1. Goal

Dense technical papers give too little perceptual and interactive feedback to keep attention coupled to the ideas. Paper World turns a paper into an environment. It keeps you engaged, ties every answer to the source text, and makes you generate the mental model instead of only consuming explanations.

The first course is a LeCun-line curriculum. Success means you can reconstruct how the ideas evolved (CNNs, similarity learning, energy-based models, JEPA, world models, planning) without the papers open, and apply the core mechanisms to a problem that was not taught. The platform is general. Changing the paper changes the content, not the learning loop.

## 2. Design principles

1. **Generate, don’t just consume.** Each explanation is followed by a task where you produce, predict, repair or explain something.
2. **Evidence first.** Every technical answer traces to a passage, figure, equation, or a clearly labeled external source. “Not supported by the available sources” is an acceptable answer.
3. **Concepts, not pages.** The system tracks concepts and their relations across papers.
4. **Visuals do explanatory work.** Visual intensity scales with conceptual complexity, and motion marks what changed.
5. **Assistance fades.** Help decreases as demonstrated competence rises.
6. **Disposable artifacts.** Interactive pages, toy simulations and animations are cheap to generate and can be thrown away. The concept model underneath is the persistent asset.
7. **Confusion is data.** A wrong answer triggers a diagnosis of where the mental model diverged, not just a correction.

## 3. Curriculum

| # | Paper | What it contributes to the story |
| --- | --- | --- |
| 1 | LeNet | Locality and weight sharing; learned spatial features |
| 2 | Siamese / contrastive learning | Learning similarity and invariance |
| 3 | Energy-based learning | Compatibility between configurations |
| 4 | “Deep Learning” (assumed: LeCun, Bengio & Hinton, Nature 2015) | Orientation and context |
| 5 | A Path Towards Autonomous Machine Intelligence | World model, JEPA proposal, planning |
| 6 | I-JEPA | Predicting representations of image regions |
| 7 | V-JEPA | Extending the idea to video |
| 8 | V-JEPA 2 | World-model use and planning |

Two suggestions to decide on: (a) read paper 4 first as orientation or treat it as optional; (b) consider adding VICReg, Barlow Twins or BYOL between 2 and 6, since they sit between contrastive methods and JEPA. Neither was in the original plan.

Between each pair of papers the system asks “What changed?” That question matters more than any single paper’s quiz.

## 4. Architecture

```
PAPER
  ↓
PARSER  (text, figures, equations, references)
  ↓
EVIDENCE LAYER
  ├─ Grounded RAG: paper, supplement, official code, project page, cited refs
  └─ Exploratory RAG: other papers, textbooks, docs, newer work
  ↓
STRUCTURED CONCEPT MODEL  (claims, concepts, equations, components, flows)
  ↓
RENDERERS  (controlled text, diagrams, HTML, simulations, audio, video)
  ↓
LEARNING LOOP  (explain, predict, retrieve, challenge, repair, transfer)
  ↓
LEARNER MODEL → SCHEDULER → back into the loop
```

The key rule is that every renderer draws from the same concept model. Text, diagram and video are views of one underlying structure, so they cannot contradict each other and each can cite the same evidence.

## 5. Evidence and trust layer

An LLM can be fluent and wrong at the same time, so this layer is not optional.

**Two retrieval scopes.** Grounded RAG answers “what does this paper say?” using only the paper and its official materials. Exploratory RAG answers “why does this work?” and may search other sources, but its answers are labeled.

**Five epistemic labels**, each styled differently on screen:

| Label | Meaning |
| --- | --- |
| SOURCE | The paper explicitly says this |
| INFERENCE | This follows from what the paper says |
| INTERPRETATION | A useful way to think about it |
| EXTERNAL | Other work says this |
| SPECULATION | A hypothesis, not established |

The goal is that you never forget which ideas are LeCun’s and which are the system’s reading of LeCun.

**Pause-and-answer format.** Every answer is followed by its evidence: section, figure, equation, or external source.

**Diagrams come from the model, not from free drawing.** The LLM produces a structured diagram specification from the verified concept model, and a renderer draws it. Then “what does this arrow mean?” has a real answer.

**Proposed verification of the concept model** (not specified in the original; a first proposal):

- every node and edge stores the source span it came from;
- extraction is run twice and disagreements are reviewed;
- the first paper is spot-checked by hand;
- nodes with no source span are flagged unverified and are not used for quizzing.

## 6. Structured concept model

**Mechanism layer.** Before reading deeply, each paper gets answers to eight questions: what goes in, what transformation happens, what comes out, what is learned, what the objective is, what changes during training, why it should work, and where it could fail.

**Five-level ladder.** Detail is added progressively instead of presenting the full notation at once:

1. What is the machine doing?
2. What exactly is represented?
3. How is it trained?
4. Why does the objective produce useful representations?
5. What assumptions are hiding in the method?

**Representation format.** For v1, use typed nodes (component, representation, objective, training step, claim) and typed edges (produces, consumes, predicts, updates, compared-with), stored as JSON or YAML, each with a source span. Render to Mermaid or SVG. A sketch:

```
TargetEncoder
  input:  target patches
  output: target_embedding
  source: §3.2, Fig. 2

Predictor
  input:  context_embedding + position
  output: predicted_embedding
  source: §3.1
```

SysML v2 and ISO 19450 (OPM, Object-Process Methodology) were raised as richer options. They are worth evaluating later as an export format or a stricter schema. Neither is needed to prove the learning loop, and their tooling cost is high for a first build.

## 7. Interaction

### The pause button

At any point you can pause and ask about “this moment.” The query automatically carries the paper, section, current concept, current visual and previous explanation, so a bare “wait, what?” is enough. Suggested actions: why does this exist, what does this term mean, show the math, show it visually, find the exact place in the paper, explain it as if I were implementing it.

### Three modes

- **Background.** Narration with simple visuals while you do something else. Every few minutes the system interrupts with a retrieval question you can answer by voice, gives feedback, and continues. The original suggested 3 to 7 minutes between interruptions; treat that as an untested starting value. Needs speech-to-text and text-to-speech.
- **Locked in.** Active tasks: place the arrow, fill the missing component, predict what happens if the predictor is removed. Prefer prediction tasks to multiple-choice.
- **Break the paper.** The system states a claim and you attack it, then it surfaces the authors’ actual argument and evidence. This separates author claim, author evidence, your interpretation and your objection.

Open question: the original says the system “notices” you are locked in but never says how. For v1, use a manual toggle. Automatic detection is a later problem.

### Controlled explanation

The learner can switch an explanation between styles: normal, plain controlled language (“80% of the way to ASD-STE100”), intuitive, mathematical, implementation, and critical (what the approach assumes). ASD-STE100 is a controlled language for aerospace maintenance documentation. Its comprehension benefit for ML explanations rests on anecdote, so it should be tested on yourself rather than assumed.

### Visual rules

- Segment content into coherent chunks and give the learner control over pace: pause, scrub, replay.
- Signal what matters with highlighting, zooming and color, and keep it purposeful.
- Let a diagram sit still and move one element at a time. Motion is an information channel, not wallpaper.
- Scale visual complexity to conceptual complexity. A one-line idea gets a tiny visual; a seven-part architecture gets a full diagram.
- Use time-based diagrams (state at t, action, state at t+1) for anything dynamic, such as video and world-model papers, so you can scrub and change an action.

## 8. Learning mechanics

| Mechanism | What it looks like | Evidence note |
| --- | --- | --- |
| Retrieval practice | Free recall, “draw it from memory,” “give the one equation that matters” | Strong |
| Spacing | Reviews scheduled by errors and weak links rather than the calendar | Spacing is strong; the error-driven scheduling is a design choice, untested here |
| Self-explanation | “Why does this component exist?” Answers classified as correct, missing, or misconception | Moderate to strong |
| Learner-built maps | Fill in or repair a diagram instead of only viewing it | Moderate |
| Worked examples with fading | Help slider: full explanation, partial, hint, question, you do it | Moderate to strong, mainly for novices |
| Expertise reversal | Interface becomes less helpful as you demonstrate mastery | Established in cognitive load research; recent results are mixed |
| Interleaving | “What changed?” across papers; contrast sets (why are B and C wrong?) | Moderate; not re-checked here |
| Prediction before explanation | Predict, run a toy simulation, compare | Plausible; no source cited in the original |
| Failure-first | “Where does it break?” (retained information, bottleneck size, compounding error) | Plausible; no source cited in the original |
| Multiple representations and translation | Move between text, diagram, equation, code; spot which link you lack | Plausible diagnostic design |
| Transfer tasks | Apply the mechanism in a setting that was not taught | The goal; not a validated method here |

A typical self-explanation check: you are asked why I-JEPA does not just reconstruct pixels. The system compares your answer with the evidence and tells you whether you stated the mechanism, missed a piece (pixel targets force unnecessary detail), or held a misconception (such as the target encoder generating the prediction). If a misconception appears, it asks one small question that separates the two ideas instead of dumping another explanation.

## 9. Learner model

Per concept, store:

- an evidence checklist: recognized it, explained it from memory, used it in a new problem, distinguished it from its nearest neighbor, connected it across text, diagram and equation, implemented a toy version;
- a misconception log with failure counts;
- last seen, current help level, and next review priority.

Confidence should be derived from that evidence. In the original conversation the percentages (92%, 61%, and so on) were illustrative and were not computed from anything. For v1, use a simple rule such as difficulty-weighted successes with decay, with failures raising review priority. Evaluating knowledge-tracing approaches and existing spaced-repetition schedulers is an open task.

## 10. MVP and build order

| Phase | Scope | Done when |
| --- | --- | --- |
| 0 | One paper (I-JEPA): parse text, figures and equations; local RAG; pause-and-ask with citations | Answers cite the right passage in spot checks |
| 1 | Hand-checked concept model for that paper; generated diagram; ladder levels 1–2; five retrieval questions | You can redraw the architecture from memory a day later |
| 2 | Background mode with voice interrupts | You can use it away from the screen and still answer the checks |
| 3 | Learner model and error-driven review; add V-JEPA to test cross-paper linking | The system correctly asks “what changed?” between papers |
| 4 | Interactive HTML and toy simulations; Exploratory RAG; generated video as an experiment | At least one concept is understood through a simulation |
| 5 | Rest of the curriculum | Reconstruct the idea trajectory without the papers |

## 11. How we will know it works

This is a single learner, so use within-person comparison. Study two comparable papers or sections: one with PDF plus notes, one with Paper World. Measure free recall and one transfer question at 1 day and 7 days, plus time on task, self-reported mind-wandering and perceived effort. Also track system quality: the share of answers whose cited passage actually supports the claim, and the hallucination rate found in spot checks. Expect novelty effects and treat results as suggestive.

## 12. Open decisions

- Tech stack, model choice, cost and latency (not specified in the original).
- How to detect “locked in” versus background.
- Whether to adopt a formal representation (SysML v2, OPM) or stay with the simple typed graph.
- Which “Deep Learning” paper is meant, and whether to add VICReg, Barlow Twins or BYOL.
- Treatment of generated video: an experiment, not a core feature, because the evidence for it is early.
- Assumptions with no source: that you learn better from seeing a mechanism operate, and the 3–7 minute interruption interval.

## 13. Bibliography and evidence notes

Status key: **Opened** means a search or fetch in this pass confirmed the source exists and I read the abstract or excerpt. **Not re-checked** means the original link is plausible but I did not reopen it. **Not located** means I could not find it.

| Source | Used for | Type | Evidence note | Status |
| --- | --- | --- | --- | --- |
| Nature Reviews Psychology, “The science of effective learning with spacing and retrieval practice” (https://doi.org/10.1038/s44159-022-00089-1) | Spacing and retrieval | Review | Strong support for both techniques | Not re-checked |
| Annual Reviews, “Practicing Retrieval Facilitates Learning” (https://doi.org/10.1146/annurev-psych-010419-051019) | Retrieval practice | Review | Strong | Not re-checked |
| “Five Strategies for Optimizing Instructional Materials” (https://pmc.ncbi.nlm.nih.gov/articles/PMC7940870/) | Signaling, segmenting, removing extraneous material | Review | Moderate to strong; general multimedia guidance | Not re-checked |
| “Cognitive load in multimedia learning environments: A systematic review” (https://www.sciencedirect.com/science/article/abs/pii/S036013151930171X) | Seductive details, redundancy, cognitive load | Systematic review | Moderate | Not re-checked |
| Segmenting-effect meta-analysis (https://doi.org/10.1007/S10648-018-9456-4) | Segmenting content | Meta-analysis | Moderate; benefits can differ by prior knowledge | Not re-checked |
| Bimba et al. 2017, “Adaptive feedback in computer-based learning environments” (https://journals.sagepub.com/doi/10.1177/1059712317727590) | Adaptive feedback | Review | Descriptive; weak evidence of effectiveness on its own | Not re-checked |
| Nesbit & Adesope 2006, concept and knowledge maps (https://journals.sagepub.com/doi/pdf/10.3102/00346543076003413) | Concept maps | Meta-analysis | Moderate; learner-constructed maps are the stronger case | Not re-checked |
| “Inducing Self-Explanation: a Meta-Analysis” (https://doi.org/10.1007/s10648-018-9434-x) | Self-explanation prompts (64 reports) | Meta-analysis | Moderate to strong. In the original, the “64 reports” claim was first attached to the concept-map link; it belongs to this one | Not re-checked |
| “The design and utilization of effective worked examples: A meta-analysis” (https://digitalcommons.unl.edu/dissertations/AAI3208114/) | Worked examples and fading | Doctoral dissertation | Not peer-reviewed. Prefer a journal source; Atkinson et al. 2000, “Learning from examples: instructional principles from the worked examples research,” appeared in reference lists I saw but I did not open it | Not re-checked |
| 2025 expertise-reversal meta-analysis (https://doi.org/10.1016/j.learninstruc.2025.102142) | “High assistance helps novices, reduced assistance helps experts” | Claimed meta-analysis | I could not find this DOI or paper | **Not located** |
| Kalyuga 2007, “Expertise Reversal Effect and Its Implications for Learner-Tailored Instruction” (https://link.springer.com/article/10.1007/s10648-007-9054-3) | Expertise reversal (replacement anchor) | Review | Established framework within cognitive load theory | **Opened** |
| Gorbunova et al. 2025, “Rethinking pre-training” (https://doi.org/10.3389/fpsyg.2025.1628047) | Counterpoint to expertise reversal | Experiment | Reported pre-training reduced extraneous load even for higher-prior-knowledge learners, so reversal is not universal | **Opened** |
| Interleaving (https://doi.org/10.1002/rev3.3266) | Interleaving, transfer | Review/meta-analysis (unverified) | Moderate; strongest for similar, confusable concepts | Not re-checked |
| ASD-STE100 FAQ (https://asd-ste100.org/STE\_faq.html) | What the controlled language is | Specification site | Describes the language; not evidence of a learning benefit | Not re-checked |
| Generative Lecture, arXiv 2512.21796 (https://arxiv.org/abs/2512.21796) | Interactive AI-generated lectures | Preprint | Early; system description | Not re-checked |
| AI-generated instructional videos (https://www.sciencedirect.com/science/article/pii/S2155709826000150) | Generated video | Study | Narrow domain (grammar, research methods); early, so generated video is an interface idea, not proven pedagogy | Not re-checked |
| Heldreth et al., “Experimentally Testing AI-Powered Content Transformations on Student Learning” (https://arxiv.org/abs/2509.18664; also published in Frontiers in Artificial Intelligence) | Interactive AI platform vs conventional reader | Between-subjects experiment | 60 students aged 15–18, textbook chapter, Google-affiliated authors. Positive but small and not about research papers or adult self-learners. The original cited a PMC link (PMC13008931) that I did not open | **Opened** (arXiv version) |
| OwlMentor RAG study (https://pmc.ncbi.nlm.nih.gov/articles/PMC11502405/) | RAG tool for scientific text in a course | Longitudinal study | Not in the original; found during checking and closest to this project’s use case | **Opened** (abstract) |

**Claims in the source conversation with no citation:** that a 2025 study of AI support at video pause points gave “nuanced” results; that you personally learn better from seeing mechanisms; that 3–7 minutes is the right interruption interval; and that STE-style writing is more readable for ML explanations.
