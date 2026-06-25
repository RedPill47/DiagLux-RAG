"""Pure Python/numpy Okapi BM25.

Implemented in-repo (no ``rank_bm25``/``bm25s`` dependency) so that the exact
scoring formula is documented and reproducible:

    score(q, d) = sum_{t in q} qtf(t) * idf(t) *
                  tf(t, d) * (k1 + 1) / (tf(t, d) + k1 * (1 - b + b * |d| / avgdl))

with the standard Okapi defaults ``k1 = 1.5`` and ``b = 0.75``.

Decisions:
- **IDF** uses the Lucene/"+1 inside the log" variant
  ``idf(t) = ln(1 + (N - df + 0.5) / (df + 0.5))``, which is always
  non-negative (the classic Robertson IDF goes negative for terms in more
  than half the documents — undesirable on a small corpus of short-story
  chunks where common Luxembourgish function words easily exceed df > N/2).
- **Repeated query terms** contribute once per occurrence (``qtf``
  multiplier), the usual Okapi treatment.
- Query terms absent from the index contribute 0.
- ``score`` returns a score for EVERY document (the contract requires full
  rankings, not top-k), as a float64 numpy array aligned with ``doc_ids``.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from diaglux.retrieval.tokenize import TokenizerFn, word_tokenize


class BM25Index:
    """Okapi BM25 index over a tokenized corpus.

    Parameters
    ----------
    tokenized_docs : one token list per document.
    doc_ids : optional external ids aligned with ``tokenized_docs``
        (defaults to integer indices). For DiagLux these are ``chunk_id``s.
    k1, b : Okapi parameters (defaults 1.5 / 0.75, documented above).
    """

    def __init__(
        self,
        tokenized_docs: Sequence[Sequence[str]],
        doc_ids: Optional[Sequence] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if doc_ids is not None and len(doc_ids) != len(tokenized_docs):
            raise ValueError("doc_ids length must match tokenized_docs length")
        self.k1 = float(k1)
        self.b = float(b)
        self.doc_ids = list(doc_ids) if doc_ids is not None else list(range(len(tokenized_docs)))
        self.n_docs = len(tokenized_docs)

        self.doc_lengths = np.array([len(d) for d in tokenized_docs], dtype=np.float64)
        self.avgdl = float(self.doc_lengths.mean()) if self.n_docs else 0.0

        # Postings: term -> (doc index array, term frequency array).
        postings_raw: Dict[str, Dict[int, int]] = {}
        for i, doc in enumerate(tokenized_docs):
            for term, tf in Counter(doc).items():
                postings_raw.setdefault(term, {})[i] = tf
        self._postings: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._idf: Dict[str, float] = {}
        for term, hits in postings_raw.items():
            idx = np.fromiter(hits.keys(), dtype=np.int64, count=len(hits))
            tfs = np.fromiter(hits.values(), dtype=np.float64, count=len(hits))
            self._postings[term] = (idx, tfs)
            df = len(hits)
            self._idf[term] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    @classmethod
    def from_chunks(
        cls,
        chunks: Sequence[dict],
        tokenizer: Optional[TokenizerFn] = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> "BM25Index":
        """Build an index from chunk records (contract: corpus_chunks_*.jsonl)."""
        tok = tokenizer or word_tokenize
        return cls(
            [tok(c["chunk_text"]) for c in chunks],
            doc_ids=[c["chunk_id"] for c in chunks],
            k1=k1,
            b=b,
        )

    def idf(self, term: str) -> float:
        """IDF of ``term`` (0.0 if unseen)."""
        return self._idf.get(term, 0.0)

    def score(self, query_tokens: Sequence[str]) -> np.ndarray:
        """BM25 scores for ALL documents (float64 array aligned with doc_ids)."""
        scores = np.zeros(self.n_docs, dtype=np.float64)
        if self.n_docs == 0:
            return scores
        denom_norm = self.k1 * (1.0 - self.b + self.b * self.doc_lengths / self.avgdl) \
            if self.avgdl > 0 else np.full(self.n_docs, self.k1)
        for term, qtf in Counter(query_tokens).items():
            posting = self._postings.get(term)
            if posting is None:
                continue
            idx, tfs = posting
            contrib = self._idf[term] * tfs * (self.k1 + 1.0) / (tfs + denom_norm[idx])
            scores[idx] += qtf * contrib
        return scores

    def rank(self, query_tokens: Sequence[str]) -> List[Tuple[object, float]]:
        """Full ranking ``[(doc_id, score), ...]`` best-first.

        Ties are broken by corpus order (stable sort) for reproducibility.
        """
        scores = self.score(query_tokens)
        order = np.argsort(-scores, kind="stable")
        return [(self.doc_ids[i], float(scores[i])) for i in order]
