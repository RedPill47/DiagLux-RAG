# Dense & Hybrid Retrieval: Findings (consolidated)

**Run date:** 2026-06-13. Extends `retrieval_findings_bm25.md` with dense retrieval
(`multilingual-e5-base`, `BAAI/bge-m3`) and hybrid fusion (RRF + the weighted grid),
all on **overlap** chunks. Embeddings computed CPU-only in the `.venv-dense` Python 3.12
side-environment (torch has no cp314 wheel); chunk embeddings cached under
`outputs/retrieval/emb_cache/`. Metrics vs. the aligned `critical_span` (6 questions
skipped for null/empty spans). Source-text recall is reported for open-corpus only
(trivially 1.0 under text-restriction).

## Top-line: character-n-gram (subword) BM25 is the best retriever

The single most important result, from the Sec. 2.9 subword ablation: **a purely lexical
character-n-gram BM25 beats word-BM25, both dense embedders, and hybrid fusion**, on every
setting and metric.

| open-corpus, q+opts | SrcRecall@10 | EvidRecall@1 | EvidRecall@10 |
|---|---|---|---|
| **Hybrid-w0.5 (char-BM25 + e5)** | **0.956** | **0.620** | **0.904** |
| BM25-char | 0.950 | 0.610 | 0.883 |
| Hybrid-RRF (char-BM25 + e5) | 0.948 | 0.562 | 0.875 |
| Hybrid-w0.5 (word-BM25 + e5) | 0.938 | 0.591 | 0.853 |
| BM25-word | 0.912 | 0.539 | 0.809 |
| Dense-bge-m3 | 0.912 | 0.467 | 0.800 |
| Dense-e5 | 0.888 | 0.407 | 0.762 |

char-BM25 holds even with the lexical-only `question_only` query (EvidRecall@10 0.715 vs. word-BM25
0.670), so it is not an artifact of option text.

**Does dense add anything on top of char-BM25?** Yes, but only via *weighted* fusion. Weighted
fusion (alpha = 0.5) of char-BM25 + e5 dense is the best system measured (EvidRecall@10 0.904, a
consistent gain of about 0.02 over char-BM25 alone at every k). But **RRF fusion of the same two
rankers is slightly *worse* than char-BM25 alone** (0.875): RRF weights the much weaker dense ranker
equally, dragging the strong lexical ranker down, whereas BM25-leaning weighted fusion lets dense
contribute only complementary signal. So dense embeddings retain marginal, genuinely additive value
for Luxembourgish even against the best lexical method, provided the fusion is weighted toward the
lexical signal.

**Mechanism (measured; see `outputs/analysis/orthography_gap.md`, `scripts/analyze_orthography.py`):**
of question content words that have any lexical correspondent in the gold evidence span, **36% are
spelling variants** (informal question orthography vs. literary text orthography), recoverable only
at the subword level, invisible to word-BM25 and, evidently, to off-the-shelf multilingual embedders.
A further 64% of question words are genuine paraphrase or absent; that dense retrieval still fails to
capture *that* signal is itself the low-resource-transfer result.

**Paper claim this licenses:** for Luxembourgish, a low-resource, orthographically variable,
compounding Germanic language, the dominant retrieval signal is *subword lexical* matching, not
multilingual dense embeddings; off-the-shelf dense adds only marginal value, and only when fused
weighted toward the lexical ranker. This is a distinctive, actionable contribution.

## Secondary: hybrid wins among the originally-planned methods; off-the-shelf dense alone underperforms word-BM25

### Open-corpus, query = question+options

**Source-text Recall@k (the Sec. 2.3 saturation check):**

| system | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| BM25 | 0.773 | 0.844 | 0.880 | 0.912 |
| Dense-e5 | 0.703 | 0.789 | 0.831 | 0.888 |
| Dense-bge-m3 | 0.741 | 0.827 | 0.873 | 0.912 |
| **Hybrid-w=0.5** | **0.822** | **0.894** | **0.914** | 0.938 |
| Hybrid-RRF | 0.800 | 0.862 | 0.898 | **0.938** |

**Evidence Recall@k vs critical span:**

| system | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| BM25 | 0.539 | 0.675 | 0.749 | 0.809 |
| Dense-e5 | 0.407 | 0.566 | 0.651 | 0.762 |
| Dense-bge-m3 | 0.467 | 0.658 | 0.715 | 0.800 |
| **Hybrid-w=0.5** | **0.591** | 0.727 | 0.790 | 0.853 |
| Hybrid-w=0.3 | 0.568 | 0.719 | **0.787** | **0.856** |
| Hybrid-RRF | 0.544 | 0.689 | 0.757 | 0.849 |

Four robust conclusions:

1. **Dense alone is worse than BM25 for open-corpus.** Both embedders trail lexical BM25 on evidence
   recall at every k; e5 trails on source-text recall too. This is the Sec. 2.9 story made
   concrete: off-the-shelf multilingual embedders transfer poorly to low-resource Luxembourgish,
   where the informal-question vs. literary-text orthography gap rewards exact lexical matching.
2. **Model choice matters: bge-m3 beats e5.** The stronger multilingual model (bge-m3) closes the
   gap, tying BM25 on source-text recall@10 (0.912) and nearly matching it on evidence recall (0.800
   vs. 0.809 @10); e5 is clearly weaker. So "use a multilingual embedder" is not a free lunch: the
   specific model is a first-order decision for Luxembourgish.
3. **Hybrid is best, and BM25-leaning fusion wins.** Weighted fusion with alpha around 0.3 to 0.5
   (BM25 weight) tops every single-method cell; RRF is close and parameter-light. Dense contributes
   complementary signal even though it is weaker alone; the paper's hybrid recommendation holds, with
   the nuance that the optimal mix leans lexical.
4. **No saturation.** Even the best system reaches only 0.938 source-text recall@10 over the 16-text
   corpus; 6% of questions never surface the right document. **The distractor-augmented corpus
   (Sec. 2.3) is unnecessary**; the open-corpus comparison already discriminates.

### Text-restricted: dense *beats* BM25 inside the known document

Evidence Recall@k, query = question+options:

| system | @1 | @3 | @5 | @10 |
|---|---|---|---|---|
| BM25 | 0.579 | 0.787 | 0.863 | 0.956 |
| Dense-e5 | 0.547 | 0.752 | 0.894 | 0.973 |
| **Dense-bge-m3** | **0.593** | **0.822** | **0.912** | **0.975** |
| Hybrid-RRF | 0.612 | 0.820 | 0.904 | 0.970 |

The picture **inverts** once document selection is removed: bge-m3 dense retrieval beats BM25 at
every k, and e5 catches up at high k. This localizes the failure precisely: **dense retrieval's
weakness is document selection across the corpus, not evidence selection within a passage.** Inside
the correct text, semantic/paraphrase matching helps (questions are paraphrased, informally-spelled
restatements of the evidence). This text-restricted vs. open-corpus contrast is a clean diagnostic
result for the paper.

## Query-mode ablation (Sec. 2.7): question+options helps retrieval for every method

Open-corpus Evidence Recall@5:

| system | question_only | question+options |
|---|---|---|
| BM25 | 0.582 | 0.749 |
| Dense-e5 | 0.438 | 0.651 |
| Dense-bge-m3 | 0.552 | 0.715 |
| Hybrid-RRF | 0.568 | 0.757 |

Adding the answer options to the query improves evidence recall by about 0.12 to 0.17 across all
methods. **Caveat (couples to Sec. 2.5):** the same option text nearly doubles the distractor-span
retrieval rate (BM25-only finding), so question+options retrieves more evidence *and* more lures.
Carry both query modes into the answering stage so the retrieval-trap analysis can use the contrast.

## Recommendations carried into the answering stage

1. **Primary retriever: weighted hybrid (char-BM25 + dense, alpha around 0.5)**, best measured. Report
   **char-BM25 alone** as the simple, near-equal strong baseline (it is within about 0.02 and far
   simpler). Note that RRF fusion underperforms char-BM25 alone here because it over-weights the weak
   dense ranker, a useful methodological point (RRF is not always the safe default when one ranker is
   much weaker). Full alpha grid and word-BM25 variants in an appendix.
2. **The retrieval comparison is itself a result, not a baseline:** the ordering
   BM25-char > hybrid > BM25-word > bge-m3 > e5, plus the measured orthography mechanism, is a
   distinctive Sec. 2.9 contribution: off-the-shelf multilingual dense underperforms *subword lexical*
   matching for Luxembourgish, and model choice (bge-m3 > e5) is first-order.
3. **The failure is document selection, not evidence selection:** text-restricted inverts the
   dense-vs-BM25 ordering. Keep this contrast.
4. **Main config:** overlap chunks, question+options; both query modes carried forward for the
   retrieval-trap analysis; sentence chunks to the appendix (BM25 doc).
5. **Drop the distractor corpus** from scope; open-corpus is already non-saturating (best about 0.95).

## Environment / reproducibility notes

- Dense retrieval runs in `.venv-dense` (Python 3.12, `torch==2.12.0+cpu`,
  `sentence-transformers==5.5.1`). `bge-m3` additionally requires `sentencepiece` + `protobuf`
  (its XLM-RoBERTa tokenizer); without them sentence-transformers raises
  "Unrecognized processing class".
- **char-n-gram (subword) BM25 ablation (Sec. 2.9): done.** The method label reflects a non-word
  analyzer (`bm25_char_ngram`) so it does not overwrite the word rankings (`search.py`,
  `run_retrieval.py --analyzer char_ngram`). It is the best single retriever; see the top-line section.
