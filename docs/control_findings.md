# Control findings: random, closed-book, full-text oracle

All four answering models completed the two control conditions (2026-06-13), 640 questions
each, decoding per `docs/methods_decoding.md`. Numbers from `scripts/analyze_results.py`
(bootstrap 95% CIs, questions resampled, seeded).

## The control table

| System | Model | Accuracy | 95% CI |
|---|---|---|---|
| Random baseline | (none) | 0.208 | [0.177, 0.241] |
| Closed-book | deepseek-v4-pro | 0.581 | [0.544, 0.620] |
| Closed-book | Claude Sonnet 4.6 | 0.595 | [0.558, 0.634] |
| Closed-book | gpt-5.5 | 0.675 | [0.639, 0.711] |
| Closed-book | Claude Opus 4.8 | 0.686 | [0.650, 0.722] |
| Full-text oracle | deepseek-v4-pro | 0.805 | [0.773, 0.834] |
| Full-text oracle | Claude Sonnet 4.6 | 0.830 | [0.800, 0.859] |
| Full-text oracle | gpt-5.5 | 0.833 | [0.803, 0.861] |
| Full-text oracle | Claude Opus 4.8 | 0.853 | [0.825, 0.880] |

## 1. The decision point: oracle accuracy is high, so we proceed to RAG

Oracle accuracy is **0.81 to 0.85** across all four models, comfortably high. LLMs *can* read
Luxembourgish when given the full correct passage, so the paper proceeds as planned (RAG
grid), not the alternative "LLMs cannot yet read Luxembourgish even with context" reframe.
The oracle scores cluster tightly across very different models (a 4.8-point spread, and
several CIs overlap), suggesting **about 0.83 is roughly the comprehension ceiling** for these
LLMs on LuxDiagRC, and that cross-model differences at the oracle are small. Opus 4.8 is
nominally highest (0.853) but only Opus vs. deepseek (0.853 vs. 0.805) is a clean separation;
the middle three overlap.

## 2. Context is required, and the closed-book level warrants a contamination caveat

Every model beats the 0.25 random baseline closed-book by a wide margin (0.58 to 0.69), and
oracle exceeds closed-book by **+16 to +24 points** for every model. The large gap is the
key control result: **the questions genuinely require the passage** (they are not solvable
from the options alone), validating the context-grounded framing.

But the *absolute* closed-book level is high: Opus 4.8 reaches 0.686 with no text at all.
This is exactly what the closed-book control exists to surface. Two non-exclusive causes:
(a) reading-comprehension MCQs have a guessable subset (world knowledge, option plausibility,
elimination); (b) the corpus is **published Luxembourgish literature**, so partial
memorization/contamination is plausible. **Recommended follow-up:** a per-text closed-book
breakdown. If specific texts score far above the rest closed-book, that flags memorization
and should be discussed openly (it does not invalidate the oracle/RAG comparisons, which are
within-model, but it qualifies the closed-book interpretation).

deepseek-v4-pro additionally **refuses** some closed-book items (e.g., "the reading text is
missing, please provide it"), a qualitatively different, honest non-answer rather than a
guess, captured as unparseable.

## 3. Cognitive-demand gradient (diagnostic)

At the oracle, accuracy tracks cognitive demand cleanly (deepseek-v4-pro, representative):

| Cognitive type | Oracle accuracy |
|---|---|
| Retrieve (fact lookup) | 0.930 |
| Evaluative | 0.789 |
| Inferential | 0.776 |
| Interpret | 0.760 |

**Retrieve** (locate a stated fact) is easiest by a wide margin; the deeper-comprehension
types (Interpret/Infer/Evaluate) are about 15 to 17 points harder. This is the expected gradient and
is the backbone of the cognitive-breakdown analysis once RAG runs are in.

## 4. Parse/robustness notes

- Reasoning models (gpt-5.5, deepseek-v4-pro) occasionally exhaust even an 8192-token budget
  on internal reasoning and emit no answer (deepseek: about 4% closed-book, about 2% oracle, logged
  unparseable). The Claude models, given a 2048-token budget, had near 0 unparseable after the
  budget fix.
- These controls were collected with the same harness/parser used for RAG; deepseek's
  unparseables are genuine (empty/refusal), not parser-recoverable.

## RAG results

See `docs/rag_findings.md` for the answering-with-retrieval results. As of 2026-06-15 the RAG
grid is **complete for all four models** and confirms the predicted shape: accuracy rises
monotonically with k toward each model's oracle (e.g. deepseek text-restricted k=10 reaches
0.795 raw / 0.811 parseable-only, at the 0.805 oracle), every RAG config beats closed-book by a
wide margin, and text-restricted beats open-corpus at every k. The retrieval-trap diagnostic
separates retrieval failure from comprehension failure cleanly (see `docs/rag_findings.md`).
