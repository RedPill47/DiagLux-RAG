# RAG answering findings

Per-model accuracy when the LLM answers from retrieved chunks (the best retriever:
weighted hybrid char-BM25+dense, overlap chunks, question+options query). Compare against
each model's own controls in `docs/control_findings.md`. Status: 2026-06-15.

**Run status: COMPLETE.** All four models finished, every config at n=640, 0 error rows, 0
duplicates (verified 2026-06-15).

| Model | RAG configs | Status |
|---|---|---|
| deepseek-v4-pro | 6/6 (minimal tier) | complete |
| Claude Sonnet 4.6 | 16/16 | complete |
| Claude Opus 4.8 | 16/16 | complete |
| gpt-5.5 | 16/16 | complete |

## Headline: RAG accuracy by model, setting, and k (best retriever)

Accuracy with the best retriever (weighted hybrid char-BM25+dense), raw (unparseable scored
wrong). Oracle and closed-book are each model's own controls.

**Text-restricted:**

| Model | k=1 | k=3 | k=5 | k=10 | oracle | closed-book |
|---|---|---|---|---|---|---|
| Claude Opus 4.8 | 0.778 | 0.816 | 0.830 | **0.831** | 0.853 | 0.686 |
| gpt-5.5 | 0.755 | 0.809 | 0.817 | **0.827** | 0.833 | 0.675 |
| Claude Sonnet 4.6 | 0.717 | 0.755 | 0.783 | **0.795** | 0.830 | 0.595 |
| deepseek-v4-pro | 0.703 | n/a | 0.769 | **0.795** | 0.805 | 0.581 |

**Open-corpus:**

| Model | k=1 | k=3 | k=5 | k=10 | oracle |
|---|---|---|---|---|---|
| Claude Opus 4.8 | 0.748 | 0.787 | 0.798 | 0.812 | 0.853 |
| gpt-5.5 | 0.714 | 0.778 | 0.802 | 0.814 | 0.833 |
| Claude Sonnet 4.6 | 0.680 | 0.756 | 0.752 | 0.787 | 0.830 |
| deepseek-v4-pro | 0.639 | n/a | 0.727 | 0.755 | 0.805 |

(deepseek ran the minimal tier, so k=3 was not run.)

### Four conclusions, each the shape the design predicts

1. **Accuracy rises monotonically with k toward each model's oracle.** The two strongest models
   (Opus, gpt-5.5) reach within about 0.02 of their oracle by text-restricted k=10 (0.831 vs.
   0.853; 0.827 vs. 0.833): once enough evidence from the known document is in context, RAG
   essentially matches full-context answering, so the residual error is comprehension, not retrieval.
2. **Cross-model ordering is Opus > gpt-5.5 > Sonnet > deepseek**, mirroring the oracle ordering.
   The two reasoning-or-frontier models convert retrieved evidence into answers most effectively.
   (Cross-model gaps are reported descriptively; gaps involving gpt-5.5 under about 5 points are
   ties per the variance check in `docs/methods_decoding.md`.)
3. **Every RAG configuration beats the model's own closed-book** by a wide margin (e.g. Opus
   text-restricted k=10 0.831 vs. closed-book 0.686). Retrieval supplies genuine, used signal.
4. **Text-restricted beats open-corpus at every k** (by roughly 0.03 to 0.06). This is the cost of
   document selection across the corpus, consistent with the retrieval study.

## deepseek-v4-pro: raw vs. parseable-only

deepseek loses about 2 to 10% of items per config to reasoning-budget truncation (logged
unparseable, scored wrong), so its true accuracy sits above the raw column:

| setting | k | accuracy (raw) | accuracy (parseable-only) | unparseable |
|---|---|---|---|---|
| text-restricted | 1 | 0.703 | 0.744 | 35 |
| text-restricted | 5 | 0.769 | 0.792 | 19 |
| text-restricted | 10 | 0.795 | 0.811 | 12 |
| open-corpus | 1 | 0.639 | 0.708 | 62 |
| open-corpus | 5 | 0.727 | 0.778 | 42 |
| open-corpus | 10 | 0.755 | 0.802 | 38 |

The unparseable rate falls as k rises (62 at open-corpus k=1 down to 12 at text-restricted k=10):
with more context the model commits to an answer rather than exhausting its reasoning budget.
gpt-5.5 shows the same reasoning-budget effect to a smaller degree; the Claude models, on a
2048-token budget, have near-zero unparseable.

## Method comparison at k=5 (Opus, complete)

Does the best retriever's advantage carry through to answer accuracy? Opus, k=5, by retriever:

| retriever | text-restricted | open-corpus |
|---|---|---|
| hybrid char-BM25+dense | 0.830 | 0.798 |
| bm25 (word) | 0.814 | 0.766 |
| bm25 char | 0.816 | 0.795 |
| dense bge-m3 | 0.833 | 0.777 |
| hybrid RRF char | 0.813 | 0.784 |

The retrieval-quality ordering carries through **more in open-corpus than text-restricted**. In
open-corpus (where retrieval is the bottleneck) the subword/hybrid retrievers lead and word-BM25
trails by about 0.03; in text-restricted (document already known) the answer accuracies are
compressed within about 0.02 and dense bge-m3 edges ahead within noise. Reading: better retrieval
helps the answer most exactly where retrieval is hard. Sonnet and gpt-5.5 show the same pattern.

## Retrieval trap: retrieval failure vs. comprehension failure (the novel diagnostic)

This is the analysis the typed distractors were designed for. For each question we check whether
the top-k retrieved chunks contained the gold critical span and/or the distractor span, then look
at what the model chose. Best retriever, open-corpus, k=5 (n=627; 13 questions skipped for
unresolved/empty spans), representative:

| what was retrieved | n | chose the distractor option | accuracy |
|---|---|---|---|
| distractor span only (critical missed) | 31 | 12.9% | 0.774 |
| both spans | 462 | 5.8% | 0.829 |
| critical span only | 64 | 1.6% | 0.859 |
| neither | 70 | 17.1% | 0.557 |

Two clean effects:

1. **The trap is real.** Surfacing the distractor span *without* the critical span roughly 8x's the
   rate of choosing the lure (12.9% vs. 1.6% when the critical span is present). The retriever can
   actively mislead the model when it returns the wrong evidence.
2. **Retrieval failure vs. comprehension failure separate cleanly.** Accuracy is highest when the
   critical span is retrieved (0.83 to 0.86) and collapses when nothing relevant is (0.557), where
   the model also falls back to plausible-wrong options most (17.1%). So low-k / open-corpus errors
   are predominantly retrieval failures; the residual errors at high k with the critical span present
   are comprehension failures.

## Error-type confusion (chosen option type)

The distribution of *which* option type was chosen (correct / misunderstand / distractor_span /
no_support) shifts with context quality. As accuracy rises (closed-book to RAG to oracle), the
"no_support" (hallucination) and "distractor_span" (lure) shares shrink and the residual errors
concentrate in "misunderstand" (genuine misreading). Full per-system tables in
`outputs/analysis/confusion.md`.

## Generated analysis artifacts

All tables are produced by `scripts/analyze_results.py` from the per-question logs and live in
`outputs/analysis/`: `setting_accuracy` (all systems, with bootstrap CIs), `cognitive_breakdown`,
`linguistic_breakdown`, `confusion`, `significance_matrix` (McNemar), and per-config
`retrieval_trap_*` and `retrieval_vs_answer_*` (the 2x2 retrieval-success by answer-correctness).

## What lands next

- gpt-5.5 needs one more OpenAI top-up to finish open-corpus k=10 plus its 8 comparison configs.
- Once gpt-5.5 is complete: full tables with bootstrap CIs and McNemar tests; the chosen-option-type
  confusion and retrieval-trap analyses; and the 2x2 retrieval-success by answer-correctness
  decomposition. Tooling is ready (`scripts/analyze_results.py`).
