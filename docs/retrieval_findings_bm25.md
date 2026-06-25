# BM25 Retrieval Grid: Findings (Phase 3, BM25 only)

> **Superseded for cross-method comparison by `retrieval_findings_dense.md`**, which adds
> dense (e5, bge-m3) and hybrid fusion. This doc stands as the BM25-only analysis (chunking,
> query mode, saturation). Headline update from the full study: subword (character-n-gram)
> BM25 is the best single retriever, dense alone *underperforms* BM25 for open-corpus, and
> nothing saturates, so the distractor corpus is unnecessary.

**Run date:** 2026-06-13. **Scope:** BM25 over the LuxDiagRC corpus, no LLM, no dense
(torch-free). Grid: {text_restricted, open_corpus}, {question_only, question_options},
{overlap, paragraph, sentence} chunks. Source data: `outputs/retrieval/bm25_grid_metrics.csv`
(420 rows). Evidence metrics computed against the aligned `critical_span` offsets
(6 questions skipped for null/empty spans). Dense and hybrid results are in
`retrieval_findings_dense.md`.

## Headline: open-corpus retrieval does NOT saturate for BM25

The plan (Sec. 2.3) worried that source-text Recall@10 over only 16 texts would sit near 100%,
flattening the retrieval comparison. **For BM25 this does not happen:**

| chunks | query | SrcRecall@1 | @3 | @5 | @10 | MRR |
|---|---|---|---|---|---|---|
| overlap | question_options | 0.773 | 0.844 | 0.880 | **0.912** | 0.819 |
| overlap | question_only | 0.628 | 0.727 | 0.761 | 0.805 | 0.689 |
| paragraph | question_options | 0.706 | 0.820 | 0.864 | 0.916 | 0.778 |
| sentence | question_options | 0.548 | 0.689 | 0.769 | 0.838 | 0.646 |

Even at k=10, BM25 fails to surface the correct source text for **8 to 20%** of questions. The
most likely cause is the **orthography gap** (Sec. 2.9): questions use informal spelling ("Waat",
"Wei") while the texts use literary orthography. This is a *positive* result for the paper: the
open-corpus comparison has real headroom, and the distractor-augmented corpus (Sec. 2.3 "Better")
is unnecessary. (Dense retrieval, run later, did not bridge the gap or saturate either; see
`retrieval_findings_dense.md`.)

## Best configuration: overlap chunks + question+options query

**Evidence Recall@k vs critical span (criterion = any overlap):**

| setting | chunks | query | @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|---|
| text_restricted | overlap | question_options | 0.579 | 0.787 | 0.863 | **0.956** |
| text_restricted | overlap | question_only | 0.517 | 0.724 | 0.819 | 0.954 |
| text_restricted | paragraph | question_options | 0.472 | 0.744 | 0.839 | 0.945 |
| text_restricted | sentence | question_options | 0.315 | 0.505 | 0.580 | 0.705 |
| open_corpus | overlap | question_options | 0.539 | 0.675 | 0.749 | 0.809 |
| open_corpus | overlap | question_only | 0.382 | 0.519 | 0.582 | 0.670 |
| open_corpus | sentence | question_options | 0.262 | 0.366 | 0.420 | 0.481 |

Evidence recall is the **non-saturating** retrieval metric the plan (Sec. 2.4) wanted as primary, and
it cleanly separates configurations:

- **Chunking:** overlap beats paragraph, and both clearly beat sentence. Sentence chunks are too
  granular for BM25 (Evidence Recall@10 only 0.57 to 0.70 text-restricted), confirming the
  proposal's caution.
- **Query mode:** question+options beats question_only everywhere (Sec. 2.7 ablation).
- **Setting drop:** text-restricted to open-corpus costs about 0.10 to 0.15 evidence recall (real
  document-selection difficulty, consistent with the non-saturating source-text recall above).

## Sets up the retrieval-trap analysis (Sec. 2.5)

Distractor-span retrieval rate, open_corpus:

| chunks | query | @1 | @3 | @5 | @10 |
|---|---|---|---|---|---|
| overlap | question_options | 0.450 | 0.603 | 0.690 | 0.774 |
| overlap | question_only | 0.251 | 0.412 | 0.487 | 0.596 |

Adding the options to the query nearly **doubles** the distractor-span retrieval rate (0.25 to 0.45
at k=1). So question+options retrieves more *evidence* and more *distractor*, exactly the tension the
retrieval-trap table is designed to expose. The Sec. 2.7 query ablation and the Sec. 2.5 trap analysis
are therefore coupled: report both query modes through to the answering stage.

## Recommendations carried forward

1. **Main config:** overlap chunks, question+options query. Keep question_only and paragraph as
   reported ablations; drop sentence chunks to an appendix (clearly inferior for retrieval).
2. **Distractor corpus (Sec. 2.3): dropped.** Open-corpus already discriminates for BM25 and for
   dense; no augmentation needed.
3. **Carry both query modes into answering** so the retrieval-trap analysis has the contrast.

## Notes on scope

- This file covers **BM25 only**. Dense (multilingual-E5, BGE-M3) and hybrid (RRF + weighted grid)
  were run subsequently in a Python 3.12 side-environment and are reported in
  `retrieval_findings_dense.md`.
- The **subword character-n-gram BM25 ablation (Sec. 2.9) was run** and is the best single retriever;
  see `retrieval_findings_dense.md`. (Its rankings use the method label `bm25_char_ngram` so they do
  not overwrite the word-analyzer rankings.)
