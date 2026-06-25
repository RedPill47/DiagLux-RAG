# DiagLux-RAG

**Evaluating Retrieval-Augmented Generation for Luxembourgish Reading Comprehension** (LaTeLL).

This repository implements and evaluates RAG baselines for Luxembourgish multiple-choice reading
comprehension on the LuxDiagRC dataset. It decomposes the task into three settings (full-text
oracle, text-restricted retrieval, open-corpus retrieval) and four answering models, and uses the
dataset's evidence-span and typed-distractor annotations to separate *retrieval* failure from
*comprehension* failure. A full reproducible pipeline (data, retrieval, answering, analysis) with
197 automated tests.

This README documents the whole codebase so it can be shared as a single reference. For results,
see the findings docs in `docs/`.

---

## 1. Headline results (status: 2026-06-15)

**Retrieval (complete).** A character-n-gram (subword) BM25, and a weighted hybrid of char-BM25
plus dense, is the best retriever, beating off-the-shelf multilingual dense embeddings
(multilingual-E5, BGE-M3) for Luxembourgish. The cause is a measured orthography gap: of question
content words with any lexical match in the gold evidence, about 36% are spelling variants
(informal question spelling vs. literary text spelling) recoverable only at the subword level.
Open-corpus retrieval does not saturate over the 16 texts, so no distractor corpus is needed.

**Comprehension controls (complete).** Full-text oracle accuracy is 0.81 to 0.85 across all four
models, so LLMs can read Luxembourgish given the passage. Context is required: the oracle beats
closed-book by 16 to 24 points. Closed-book is nonetheless high (Opus 0.69), which warrants a
contamination caveat (the texts are published literature). Cognitive gradient: Retrieve
(fact-lookup) questions are easiest (about 0.93), Interpret/Infer/Evaluate are harder (0.76 to 0.79).

**RAG (complete, all four models).** Accuracy rises monotonically with k toward each model's oracle;
the strongest models reach within about 0.02 of full-context by k=10 (text-restricted), so the
residual error there is comprehension, not retrieval. Cross-model ordering Opus > gpt-5.5 > Sonnet
> deepseek. Text-restricted beats open-corpus at every k. The retrieval-trap diagnostic separates
retrieval failure from comprehension failure: surfacing the distractor span without the critical
span roughly 8x's the rate of choosing the lure.

Detailed numbers and tables: `docs/retrieval_findings_dense.md`, `docs/control_findings.md`,
`docs/rag_findings.md`, `docs/methods_decoding.md`.

---

## 2. Repository structure (every file)

### Top level

| Path | Purpose |
|---|---|
| `README.md` | This file. |
| `pyproject.toml` | Package definition (`diaglux`), dependencies, optional extras, pytest config. |
| `.gitignore` | Ignores caches, the dense venv, `.env`, and regenerable run artifacts (`outputs/runs`, `outputs/retrieval`, `outputs/analysis`). |
| `.env` | API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY`). Not committed. Auto-loaded by the answering scripts. |

### `src/diaglux/` (the package)

**`data/` (dataset parsing and processing)**

| File | Purpose |
|---|---|
| `kb.py` | Parse `KnowledgeBaseAnnot.txt` (16 lines, each `'Title': [40 question dicts]`) into question records. |
| `texts.py` | `load_clean_text(text_id)` and `find_data_root()`: the canonical clean-text loader (title line, author line, NFC-normalized body) that defines the character-offset coordinate system for spans and chunks. |
| `tags.py` | Strip inline annotation tags (`<LEX-FALSE-FRIEND>` etc.) and extract linguistic tags/prefixes (LEX/SYN/MORPH/ORTHO/DISC) from spans. |
| `align.py` | Locate `criticalSpan`/`distractorSpan` in the clean text and return character offsets + status. Cascade: exact, de-hyphenated, title/author-prefix-stripped, length-aware fuzzy, partial sentence-piece. Resolves all spans (only genuinely empty ones excluded). |
| `shuffle.py` | Seeded per-question option shuffling with semantic round-trip (the answer key is positionally A, so shuffling is mandatory). |
| `chunking.py` | Three chunking strategies (paragraph, overlapping, sentence) with clean-text offsets. |

**`retrieval/` (retrieval systems and metrics)**

| File | Purpose |
|---|---|
| `tokenize.py` | Luxembourgish-aware tokenization (NFC, clitic-apostrophe splitting, diacritics kept, pluggable stopwords) plus a character-n-gram analyzer. |
| `bm25.py` | Pure-numpy Okapi BM25 (no torch dependency). |
| `dense.py` | `DenseRetriever` with a lazily imported sentence-transformers backend; per-model query/passage prefixing; on-disk embedding cache. Tests inject a fake embedder. |
| `fuse.py` | Reciprocal Rank Fusion (RRF) and weighted min-max fusion. |
| `search.py` | Runs retrieval for every question across settings (text-restricted / open-corpus) and query modes (question-only / question+options); writes full rankings. |
| `metrics.py` | Evidence Recall@k / MRR vs. critical span (primary), source-text Recall@k / MRR, distractor-span retrieval rate, context-length stats. |

**`answering/` (the LLM answering harness)**

| File | Purpose |
|---|---|
| `prompts.py` | The fixed strict prompt; closed-book variant. |
| `clients.py` | `LLMClient` interface; `OpenAICompatClient` (OpenAI + DeepSeek), `AnthropicClient`, `MockClient`. Clients adapt to per-model API constraints (e.g. `max_completion_tokens`, temperature deprecation) and record effective settings. |
| `parsing.py` | Robust answer-letter parser (exact / extracted / unparseable), resilient to chain-of-thought and Luxembourgish negation ("net C"). |
| `context.py` | Builds context per system: oracle (full clean text), RAG (top-k retrieved chunks), closed-book (none). |
| `runner.py` | Runs one configuration over the questions; per-question JSONL logging; resumable; retries transient API errors, logs persistent failures as `error`, aborts cleanly on quota/auth errors. |

**`analysis/` (statistics and diagnostic tables)**

| File | Purpose |
|---|---|
| `loading.py` | Schema-validating loaders for questions, predictions, rankings, chunks (fail loudly on contract violations). |
| `accuracy.py` | Accuracy by system/setting/k/cognitive type/linguistic feature; bootstrap 95% CIs. |
| `significance.py` | McNemar's exact paired test; pairwise significance matrix. |
| `confusion.py` | Distribution of chosen option *type* (correct/misunderstand/distractor_span/no_support/unparseable). |
| `diagnostics.py` | The 2x2 retrieval-success by answer-correctness table and the retrieval-trap table. |
| `tables.py` | Render all tables to CSV + Markdown. |

### `scripts/` (CLI entry points)

| Script | Purpose |
|---|---|
| `build_dataset.py` | Phase 1: parse all sources, align spans, shuffle options, chunk; write `questions.jsonl`, chunk files, `alignment_report.md`. |
| `run_retrieval.py` | Run a retrieval method/setting/query-mode and/or compute retrieval metrics. |
| `analyze_orthography.py` | Quantify the question-vs-evidence orthography gap (the 36% subword-only figure). |
| `run_answering.py` | Run one answering configuration (random / closed_book / oracle / rag) for one model. |
| `run_rag_grid.py` | Driver for the focused RAG answering grid per model (tiers: full / reduced / minimal). |
| `make_variance_sample.py` | Build the seeded 64-question subset for the decoding variance check. |

### `tests/` (197 tests across 21 files)

`pytest` suite covering every module: `test_data_*` (parsing, alignment, shuffling, chunking),
`test_retrieval_*` (tokenization, BM25, fusion, metrics, dense via a fake embedder),
`test_answering_*` (prompts, parsing, semantic mapping, context, runner, the adaptive clients, and
API-failure resilience), and `test_analysis_*` (loading, accuracy, significance, confusion,
diagnostics). Helpers: `answering_testutils.py`, `_analysis_fixtures.py`, `_retrieval_fixtures.py`.
No network and no real LLM calls in tests.

### `docs/`

| File | Purpose |
|---|---|
| `CONTRACTS.md` | Single source of truth for IDs, the clean-text coordinate system, and all JSONL schemas (questions, chunks, rankings, predictions). Read this first to understand the data flow. |
| `retrieval_findings_bm25.md` | BM25-only grid (chunking, query mode, saturation). |
| `retrieval_findings_dense.md` | Consolidated retrieval comparison (BM25 word/char, dense, hybrid) and the orthography-gap result. |
| `control_findings.md` | Random / closed-book / full-text oracle results, with the contamination caveat and cognitive gradient. |
| `rag_findings.md` | Per-model RAG accuracy by setting and k, and the retrieval-method comparison. |
| `methods_decoding.md` | Model list, per-model decoding decisions, and the run-to-run variance check (paper methods text). |

### `dataset/` (read-only input; auto-discovered by `find_data_root()`)

| Path | Contents |
|---|---|
| `dataset/Texts/text{1..16}.txt` | Clean source texts (line 1 title, line 2 author, then body). |
| `dataset/Annotations/text{1..16}.txt` | Tagged versions (one line each; a tag annotates the preceding word). |
| `dataset/KnowledgeBaseAnnot.txt` | 640 questions (16 texts x 40), with critical/distractor spans and typed options. |
| `dataset/info.md` | Dataset notes from the annotator. |

### `outputs/` (generated artifacts)

| Path | Contents | Tracked? |
|---|---|---|
| `outputs/processed/` | `questions.jsonl`, `corpus_chunks_{paragraph,overlap,sentence}.jsonl`, `alignment_report.md`, `questions_variance_sample.jsonl`. The Phase 1 deliverables. | yes |
| `outputs/retrieval/` | Full ranking files per method/setting/query-mode, per-config metric CSVs, and the embedding cache. | gitignored (regenerable) |
| `outputs/runs/` | Per-question prediction logs (`preds_*.jsonl`) plus config sidecars, one per answering configuration. | gitignored (regenerable) |
| `outputs/analysis/` | Generated tables (CSV + Markdown): accuracy, confusion, significance, retrieval-trap, 2x2, orthography. | gitignored (regenerable) |

### Tooling directories (not part of the project)

`.venv-dense/`, `.pytest_cache/`, and `.env` are environment and credential artifacts. They are not
part of the research code and can be ignored when reviewing the codebase.

---

## 3. Installation

Two Python environments are used. The main environment runs everything except dense retrieval; a
separate Python 3.12 environment runs dense retrieval because PyTorch has no wheel for the main
interpreter's version (Python 3.14).

**Main environment (data, BM25 retrieval, answering, analysis):**

```
python -m pip install -e .[dev]                        # core + pytest
python -m pip install openai anthropic python-dotenv   # for live LLM runs
python -m pip install pandas scipy statsmodels         # for analysis
```

**Dense-retrieval environment (only needed to (re)run dense/hybrid retrieval):**

```
py -V:3.12 -m venv .venv-dense
.venv-dense/Scripts/python -m pip install -e . sentence-transformers torch sentencepiece protobuf
```

(`sentencepiece` + `protobuf` are required by BGE-M3's tokenizer.) Dense embeddings are cached to
disk, so this environment is only needed once to produce the rankings.

**API keys.** Put `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `DEEPSEEK_API_KEY` in a `.env` file at the
repo root. The answering scripts load it automatically. DeepSeek uses the OpenAI-compatible client
with `--base-url https://api.deepseek.com`.

---

## 4. Reproducing the pipeline

**Phase 1, build the dataset:**

```
python scripts/build_dataset.py
```

Produces `outputs/processed/` and the alignment report. No experiments run until the report is clean.

**Phase 2, retrieval (BM25 in the main env; dense/hybrid in `.venv-dense`):**

```
# BM25 (word and subword), both settings, both query modes, all chunk strategies
python scripts/run_retrieval.py --method bm25 --setting open_corpus --query-mode question_options --chunks outputs/processed/corpus_chunks_overlap.jsonl
python scripts/run_retrieval.py --method bm25 --analyzer char_ngram --setting open_corpus --query-mode question_options --chunks outputs/processed/corpus_chunks_overlap.jsonl

# Dense and hybrid (in the dense venv)
.venv-dense/Scripts/python scripts/run_retrieval.py --method dense --model intfloat/multilingual-e5-base --setting open_corpus --chunks outputs/processed/corpus_chunks_overlap.jsonl
.venv-dense/Scripts/python scripts/run_retrieval.py --method hybrid_w --alpha 0.5 --analyzer char_ngram --setting open_corpus --chunks outputs/processed/corpus_chunks_overlap.jsonl

python scripts/analyze_orthography.py    # the orthography-gap analysis
```

**Phase 2/3, answering (controls and RAG):**

```
# Controls per model
python scripts/run_answering.py --system random
python scripts/run_answering.py --system closed_book --provider anthropic --model claude-opus-4-8
python scripts/run_answering.py --system oracle      --provider anthropic --model claude-opus-4-8

# RAG grid per model (focused)
python scripts/run_rag_grid.py --model sonnet
python scripts/run_rag_grid.py --model opus
python scripts/run_rag_grid.py --model deepseek --tier minimal
python scripts/run_rag_grid.py --model gpt5.5
```

All answering runs are resumable: re-running the identical configuration skips already-answered
questions, so an interrupted or quota-aborted run continues with no rework.

**Phase 4, analysis:**

```
python scripts/analyze_results.py --tables all \
    --rankings outputs/retrieval/rankings_open_corpus_hybrid_w0.5_char_ngram_overlap_question_options.jsonl \
    --chunks outputs/processed/corpus_chunks_overlap.jsonl
```

Generates all tables into `outputs/analysis/`. Everything is computed from the per-question logs;
no table is hand-assembled.

---

## 5. Models and decoding

| Model | Provider | Reasoning | Effective decoding | Token budget |
|---|---|---|---|---|
| GPT-5.5 | OpenAI | yes | temperature = default (rejects 0); `max_completion_tokens` | 8192 |
| deepseek-v4-pro | DeepSeek | yes | temperature 0 | 8192 |
| Claude Opus 4.8 | Anthropic | no | temperature deprecated (model default) | 2048 |
| Claude Sonnet 4.6 | Anthropic | no | temperature 0 | 2048 |

Decoding is necessarily heterogeneous (not every model supports temperature 0; the reasoning models
emit hidden reasoning tokens and need a large budget). The harness adapts automatically and records
the effective settings in each run's config sidecar. Load-bearing comparisons are within-model;
cross-model ranking is descriptive. A variance check (`docs/methods_decoding.md`) shows Opus is
effectively deterministic and gpt-5.5 carries about a 5-point run-to-run spread, so cross-model gaps
involving gpt-5.5 under about 5 points are reported as ties.

---

## 6. Testing

```
python -m pytest -q          # 197 tests, no network, no real API calls
```

---

## 7. Current status

- Data pipeline: complete (640 questions, all spans aligned, three chunk sets).
- Retrieval study: complete (BM25 word/char, dense E5/BGE-M3, hybrid RRF/weighted).
- Answering controls: complete for all four models (random, closed-book, oracle).
- RAG grid: complete for all four models (every config at n=640, 0 error rows, 0 duplicates).
- Analysis: full cross-model tables generated (accuracy with CIs, McNemar, confusion,
  retrieval-trap, 2x2) in `outputs/analysis/`.
- Paper: complete.
