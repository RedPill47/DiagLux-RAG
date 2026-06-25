# DiagLux-RAG Data Contracts and Module Ownership

This file is the single source of truth for the schemas that connect the pipeline stages.
**Do not change a schema without updating this file and every consumer.**

## Directory ownership

| Path | Owner module | Contents |
|---|---|---|
| `src/diaglux/data/` | data pipeline | parsers, span alignment, shuffling, chunking |
| `src/diaglux/answering/` | answering harness | prompts, LLM clients, letter parsing, runners |
| `src/diaglux/retrieval/` | retrieval | BM25, dense, hybrid (RRF + weighted), retrieval metrics |
| `src/diaglux/analysis/` | analysis | accuracy tables, CIs, McNemar, confusion/diagnostic tables |
| `scripts/` | all (one script per stage, prefixed: `build_`, `run_`, `analyze_`) | CLI entry points |
| `outputs/processed/` | data pipeline writes; others read | questions.jsonl, chunk files, alignment_report.md |
| `outputs/retrieval/` | retrieval writes; answering + analysis read | ranking files |
| `outputs/runs/` | answering writes; analysis reads | per-question prediction logs |
| `tests/` | each module adds `test_<module>_*.py` files | pytest |

Raw data (read-only, never modified): `dataset/dataset/Texts/text{1..16}.txt`,
`dataset/dataset/Annotations/text{1..16}.txt`, `dataset/dataset/KnowledgeBaseAnnot.txt`.

## Identifiers

- `text_id`: file stem, `"text1"` … `"text16"`.
- `question_id`: `"{text_id}_q{idx:02d}"`, idx = 0-based position within the text's question list (e.g. `"text3_q07"`).
- `chunk_id`: `"{text_id}_{strategy}_c{idx:03d}"` (e.g. `"text5_overlap_c012"`).
- Semantic option types: `"correct" | "misunderstand" | "distractor_span" | "no_support"`.

## `outputs/processed/questions.jsonl` (one JSON object per line, one per question)

```json
{
  "question_id": "text1_q00",
  "text_id": "text1",
  "text_title": "Catherine, ech sinn esou glécklech",
  "question": "...",
  "cognitive_type": "Retrieve",            // Retrieve | Interpret | Inferential | Evaluative
  "options": {                              // stored semantic order, original texts
    "correct": "...", "misunderstand": "...", "distractor_span": "...", "no_support": "..."
  },
  "presented": {"A": "...", "B": "...", "C": "...", "D": "..."},  // after seeded shuffle
  "permutation": ["distractor_span", "correct", "no_support", "misunderstand"],
      // permutation[i] = semantic type shown as letter chr(65+i); gold_letter follows from it
  "gold_letter": "B",
  "shuffle_seed": 13,
  "critical_span": {"start": 1042, "end": 1311, "status": "exact"},
  "distractor_span": {"start": 2200, "end": 2350, "status": "fuzzy"},
      // status: exact | dehyphen | fuzzy | multiple | unresolved | empty
      // start/end are character offsets into the CLEAN text body (see below); null when unresolved/empty
  "linguistic_tags": ["LEX-FALSE-FRIEND", "SYN-VERB-SEP"],     // tags inside criticalSpan, deduped
  "linguistic_categories": ["LEX", "SYN"]                       // unique tag prefixes
}
```

Clean text body = file content with the title line and author line removed, then
`unicodedata.normalize("NFC")`, preserving original whitespace otherwise. Span offsets index into
this exact string. The data module exposes `load_clean_text(text_id) -> (title, author, body)` so
all consumers share one definition.

## `outputs/processed/corpus_chunks_{strategy}.jsonl`

Strategies: `paragraph` (natural units), `overlap` (~150-token windows, 50% overlap), `sentence` (optional).

```json
{"chunk_id": "text1_overlap_c003", "text_id": "text1", "chunk_text": "...",
 "start_char": 1800, "end_char": 2750, "n_tokens": 152}
```

`start_char`/`end_char` index into the clean text body (same coordinate system as spans).
For the `overlap` strategy the union of chunk spans must cover each text completely.

## `outputs/retrieval/rankings_{setting}_{method}_{strategy}.jsonl` (one per question)

```json
{"question_id": "text1_q00", "setting": "text_restricted",   // text_restricted | open_corpus
 "method": "bm25",                                            // bm25 | dense_<model_slug> | hybrid_rrf | hybrid_w<alpha>
 "query_mode": "question_options",                            // question_only | question_options
 "chunk_strategy": "overlap",
 "ranking": [{"chunk_id": "...", "score": 12.3, "rank": 1}, ...]}   // FULL ranking, not top-k
```

Retrieval metrics (evidence recall/MRR vs. critical span, source-text recall, distractor-span
retrieval rate) are computed post hoc from these files plus questions.jsonl + chunk files;
top-k views are slices of the full ranking.

## `outputs/runs/preds_{config_id}.jsonl` (one per question) + `preds_{config_id}.config.json`

```json
{"question_id": "text1_q00",
 "system": "bm25",            // random | closed_book | oracle | bm25 | dense_<slug> | hybrid_rrf | ...
 "setting": "text_restricted",// text_restricted | open_corpus | none (closed_book/oracle/random)
 "k": 5,                      // null for non-retrieval systems
 "model": "model-id-string",
 "context_chunk_ids": ["..."],
 "raw_output": "...",
 "parsed_letter": "B",        // "A"-"D" or null when unparseable/error
 "parse_status": "exact",     // exact | extracted | unparseable | error (API call failed after retries)
 "semantic_choice": "correct",// via permutation; null when unparseable/error
 "is_correct": true,          // false when unparseable (but track parse_status separately)
 "timestamp": "2026-06-12T14:00:00Z"}
```

The sidecar `.config.json` stores the full run configuration (model params, prompt template hash,
k, retrieval file used, seed). `config_id` = short hash of that configuration.

## Prompt (fixed, from the concept doc Section 7)

```
You are answering a Luxembourgish reading-comprehension question.
Use only the provided context.
Choose exactly one answer: A, B, C, or D.
Return only the letter of the correct answer.

Context: {context}

Question: {question}

Options:
A. {presented[A]}
B. {presented[B]}
C. {presented[C]}
D. {presented[D]}

Answer:
```

Closed-book variant omits the Context block (and the "Use only the provided context." line).
Decoding: temperature 0, max_tokens small but enough for a stray sentence (e.g. 64).

## Environment notes

- Python 3.14: torch/sentence-transformers wheels may be unavailable. Dense retrieval must be
  an optional import (`diaglux.retrieval.dense` imports sentence-transformers lazily); nothing
  else may import torch. Tests must not require torch or network access.
- Install for development: `python -m pip install -e .[dev]` (plus extras as needed).
- No real LLM API calls in tests; use a mock client.
