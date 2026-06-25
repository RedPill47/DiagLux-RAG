"""Retrieval systems for DiagLux-RAG (Phase 3).

Modules
-------
tokenize : Luxembourgish-aware text preprocessing (word + char n-gram analyzers).
bm25     : pure Python/numpy Okapi BM25 index.
dense    : DenseRetriever (sentence-transformers imported lazily; injectable embedder).
fuse     : reciprocal rank fusion (primary hybrid) + weighted min-max fusion.
search   : end-to-end retrieval runs producing rankings_*.jsonl per docs/CONTRACTS.md.
metrics  : evidence-span / source-text / distractor retrieval metrics.

NOTE: nothing in this package imports torch or sentence-transformers at module
import time. ``dense.DenseRetriever`` performs the import lazily inside
``__init__`` and only when no ``embed_fn`` is injected (Python 3.14 may have no
torch wheels; see docs/CONTRACTS.md "Environment notes").
"""

from diaglux.retrieval import bm25, fuse, metrics, search, tokenize  # noqa: F401

__all__ = ["tokenize", "bm25", "dense", "fuse", "search", "metrics"]
