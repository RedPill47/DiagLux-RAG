"""End-to-end retrieval runs producing rankings files per docs/CONTRACTS.md.

For every question in ``questions.jsonl`` this module builds a query, scores
EVERY candidate chunk, and writes one full ranking per question to::

    outputs/retrieval/rankings_{setting}_{method}_{strategy}_{query_mode}.jsonl

FILENAME CONTRACT EXTENSION: the contract specifies
``rankings_{setting}_{method}_{strategy}.jsonl``; because both query modes
are run (review_and_plan Section 2.7 ablation), the filename is extended with
``_{query_mode}`` as the final component. The ``query_mode`` field inside
each record (already part of the contract) is authoritative.

Query modes (Section 2.7 ablation):
- ``question_only``     : the question text alone.
- ``question_options``  : question + the four PRESENTED option texts (A-D,
  post-shuffle), whitespace-joined. Note this deliberately injects the
  distractor-span option into the query — that is the point of the ablation.

Settings:
- ``text_restricted`` : candidate chunks are only those whose ``text_id``
  matches the question's. BM25 corpus statistics (idf, avgdl) are computed
  over that text's chunks only (one index per text) so that the restricted
  setting is a self-contained corpus rather than a masked global index.
- ``open_corpus``     : all chunks from all texts are candidates; one global
  BM25 index.

Dense embeddings are computed once over all chunks regardless of setting
(embeddings are query-independent); restriction happens at ranking time.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

from diaglux.retrieval.bm25 import BM25Index
from diaglux.retrieval.fuse import reciprocal_rank_fusion, weighted_minmax_fusion
from diaglux.retrieval.tokenize import TokenizerFn, word_tokenize

QUERY_MODES = ("question_only", "question_options")
SETTINGS = ("text_restricted", "open_corpus")
DEFAULT_OUT_DIR = Path("outputs/retrieval")
LETTERS = ("A", "B", "C", "D")


# ---------------------------------------------------------------------- I/O

def load_jsonl(path) -> List[dict]:
    """Load a .jsonl file into a list of dicts."""
    records = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(records: Iterable[dict], path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


def rankings_filename(setting: str, method: str, strategy: str, query_mode: str) -> str:
    """Contract filename extended with ``_{query_mode}`` (see module docstring)."""
    return f"rankings_{setting}_{method}_{strategy}_{query_mode}.jsonl"


def infer_strategy(chunks: Sequence[dict]) -> str:
    """Chunking strategy from ``chunk_id`` = ``{text_id}_{strategy}_c{idx:03d}``."""
    chunk_id = chunks[0]["chunk_id"]
    parts = chunk_id.split("_")
    if len(parts) < 3:
        raise ValueError(f"cannot infer strategy from chunk_id {chunk_id!r}")
    return "_".join(parts[1:-1])


# -------------------------------------------------------------------- query

def build_query(question: dict, query_mode: str) -> str:
    """Build the retrieval query for ``question`` under ``query_mode``."""
    if query_mode == "question_only":
        return question["question"]
    if query_mode == "question_options":
        presented = question["presented"]
        return " ".join([question["question"]] + [presented[letter] for letter in LETTERS])
    raise ValueError(f"unknown query_mode {query_mode!r} (expected one of {QUERY_MODES})")


# ----------------------------------------------------------------- ranking

def _candidate_ids(chunks: Sequence[dict], setting: str, text_id: str) -> List[str]:
    if setting == "open_corpus":
        return [c["chunk_id"] for c in chunks]
    if setting == "text_restricted":
        return [c["chunk_id"] for c in chunks if c["text_id"] == text_id]
    raise ValueError(f"unknown setting {setting!r} (expected one of {SETTINGS})")


def _bm25_indices(
    chunks: Sequence[dict], setting: str, tokenizer: TokenizerFn
) -> Dict[Optional[str], BM25Index]:
    """One global index for open_corpus; one per text_id for text_restricted."""
    if setting == "open_corpus":
        return {None: BM25Index.from_chunks(chunks, tokenizer=tokenizer)}
    by_text: Dict[str, List[dict]] = {}
    for c in chunks:
        by_text.setdefault(c["text_id"], []).append(c)
    return {
        text_id: BM25Index.from_chunks(text_chunks, tokenizer=tokenizer)
        for text_id, text_chunks in by_text.items()
    }


def _ranking_payload(ranked) -> List[dict]:
    return [
        {"chunk_id": cid, "score": float(score), "rank": rank}
        for rank, (cid, score) in enumerate(ranked, start=1)
    ]


def run_search(
    questions: Sequence[dict],
    chunks: Sequence[dict],
    method: str,
    setting: str,
    query_mode: str,
    dense_retriever=None,
    alpha: Optional[float] = None,
    tokenizer: Optional[TokenizerFn] = None,
    rrf_k: int = 60,
    analyzer: str = "word",
) -> List[dict]:
    """Run retrieval for every question; return rankings records (contract schema).

    ``method`` in {"bm25", "dense", "hybrid_rrf", "hybrid_w"}. Dense and the
    hybrids require ``dense_retriever`` (a ``DenseRetriever`` or any object
    with ``index_chunks(chunks)``, ``rank(query, candidate_ids)``, ``score``,
    ``chunk_ids`` and ``slug``). ``hybrid_w`` additionally requires ``alpha``.

    The method string written into the records (and used in filenames) is
    ``bm25`` | ``dense_<model_slug>`` | ``hybrid_rrf`` | ``hybrid_w<alpha>``.
    A non-default BM25 ``analyzer`` (e.g. ``char_ngram``) is reflected in the
    label as ``bm25_<analyzer>`` so it does not overwrite the word-analyzer
    rankings (the §2.9 subword ablation).
    """
    if setting not in SETTINGS:
        raise ValueError(f"unknown setting {setting!r}")
    if query_mode not in QUERY_MODES:
        raise ValueError(f"unknown query_mode {query_mode!r}")
    needs_bm25 = method in ("bm25", "hybrid_rrf", "hybrid_w")
    needs_dense = method in ("dense", "hybrid_rrf", "hybrid_w")
    if needs_dense and dense_retriever is None:
        raise ValueError(f"method {method!r} requires a dense_retriever")
    # A non-word BM25 analyzer is reflected in the label for every method whose
    # BM25 component uses it (standalone bm25 and both hybrids), so the subword
    # variant never overwrites the word-analyzer rankings.
    bm25_suffix = "" if analyzer == "word" else f"_{analyzer}"
    if method == "hybrid_w":
        if alpha is None:
            raise ValueError("method 'hybrid_w' requires alpha")
        method_str = f"hybrid_w{alpha:g}{bm25_suffix}"
    elif method == "dense":
        method_str = f"dense_{dense_retriever.slug}"
    elif method == "bm25":
        method_str = f"bm25{bm25_suffix}"
    elif method == "hybrid_rrf":
        method_str = f"hybrid_rrf{bm25_suffix}"
    else:
        raise ValueError(f"unknown method {method!r}")

    tok = tokenizer or word_tokenize
    strategy = infer_strategy(chunks)
    bm25_by_text = _bm25_indices(chunks, setting, tok) if needs_bm25 else {}
    if needs_dense:
        dense_retriever.index_chunks(chunks)

    records = []
    for q in questions:
        query = build_query(q, query_mode)
        cand_ids = _candidate_ids(chunks, setting, q["text_id"])
        if needs_bm25:
            index = bm25_by_text[None if setting == "open_corpus" else q["text_id"]]
            bm25_ranked = index.rank(tok(query))  # full ranking over candidates
        if needs_dense:
            dense_ranked = dense_retriever.rank(query, candidate_ids=cand_ids)

        if method == "bm25":
            ranked = bm25_ranked
        elif method == "dense":
            ranked = dense_ranked
        elif method == "hybrid_rrf":
            ranked = reciprocal_rank_fusion([bm25_ranked, dense_ranked], k=rrf_k)
        else:  # hybrid_w
            ranked = weighted_minmax_fusion(
                dict(bm25_ranked), dict(dense_ranked), alpha=alpha
            )

        records.append(
            {
                "question_id": q["question_id"],
                "setting": setting,
                "method": method_str,
                "query_mode": query_mode,
                "chunk_strategy": strategy,
                "ranking": _ranking_payload(ranked),
            }
        )
    return records


def run_and_write(
    questions: Sequence[dict],
    chunks: Sequence[dict],
    method: str,
    setting: str,
    query_mode: str,
    out_dir=DEFAULT_OUT_DIR,
    **kwargs,
) -> Path:
    """``run_search`` + write the contract rankings file; returns its path."""
    records = run_search(questions, chunks, method, setting, query_mode, **kwargs)
    strategy = infer_strategy(chunks)
    method_str = records[0]["method"] if records else method
    path = Path(out_dir) / rankings_filename(setting, method_str, strategy, query_mode)
    return write_jsonl(records, path)
