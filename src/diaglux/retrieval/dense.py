"""Dense retrieval with a lazily imported sentence-transformers backend.

ENVIRONMENT CONSTRAINT (docs/CONTRACTS.md): Python 3.14 may have no
torch/sentence-transformers wheels. Therefore:

- ``sentence_transformers`` is imported INSIDE ``DenseRetriever.__init__`` and
  only when no ``embed_fn`` is injected. Importing this module never touches
  torch.
- Tests inject a fake ``embed_fn(list[str]) -> np.ndarray`` and never load a
  real model or the network.

Intended model candidates (review_and_plan Section 2.9):

- ``intfloat/multilingual-e5-base`` — strong general multilingual model;
  E5 REQUIRES asymmetric prefixes: ``"query: "`` for queries and
  ``"passage: "`` for documents (applied automatically when the model name
  contains ``"e5"``).
- ``BAAI/bge-m3`` — multilingual long-context model; no prefixes required.

Prefix handling is per-model via ``_PREFIX_RULES`` and can be overridden with
the ``query_prefix`` / ``passage_prefix`` constructor arguments.

Similarity: embeddings are L2-normalized (idempotent if the backend already
normalizes) and scored by dot product == cosine similarity.

Caching: chunk embeddings are cached to
``outputs/retrieval/emb_cache/{model_slug}_{chunks_hash}.npy`` where
``chunks_hash`` is a sha256 over (passage prefix + chunk ids + chunk texts),
so a cache entry is invalidated whenever the chunk file or the prefixing
changes. Set ``cache_dir=None`` to disable caching.
"""

from __future__ import annotations

import hashlib
import re
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_CACHE_DIR = Path("outputs/retrieval/emb_cache")

#: embed_fn signature: list of strings -> (n, d) array of embeddings.
EmbedFn = Callable[[List[str]], np.ndarray]

# (substring matched against the lowercased model name) -> (query_prefix, passage_prefix)
_PREFIX_RULES: Tuple[Tuple[str, Tuple[str, str]], ...] = (
    ("e5", ("query: ", "passage: ")),  # all E5 variants need asymmetric prefixes
    ("bge-m3", ("", "")),              # BGE-M3: no prefixes
)


def model_slug(model_name: str) -> str:
    """Filesystem-safe slug for a model name (``intfloat/multilingual-e5-base``
    -> ``intfloat_multilingual-e5-base``)."""
    return re.sub(r"[^A-Za-z0-9.\-]+", "_", model_name).strip("_")


def default_prefixes(model_name: str) -> Tuple[str, str]:
    """(query_prefix, passage_prefix) for ``model_name`` per ``_PREFIX_RULES``."""
    lowered = model_name.lower()
    for needle, prefixes in _PREFIX_RULES:
        if needle in lowered:
            return prefixes
    return ("", "")


def _normalize_rows(mat: np.ndarray) -> np.ndarray:
    mat = np.asarray(mat, dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0  # guard zero vectors
    return mat / norms


class DenseRetriever:
    """Dense retriever over chunk records.

    Parameters
    ----------
    model_name : sentence-transformers model id; also used for the cache slug
        and the ``dense_<model_slug>`` method string in rankings files.
    embed_fn : optional injected embedding function (tests / alternative
        backends). When provided, sentence-transformers is NEVER imported.
    cache_dir : directory for chunk-embedding ``.npy`` caches
        (default ``outputs/retrieval/emb_cache``); ``None`` disables caching.
    query_prefix / passage_prefix : override the per-model prefix rules.
    batch_size : encode batch size for the real backend.
    """

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-base",
        embed_fn: Optional[EmbedFn] = None,
        cache_dir: Optional[Path] = DEFAULT_CACHE_DIR,
        query_prefix: Optional[str] = None,
        passage_prefix: Optional[str] = None,
        batch_size: int = 32,
    ) -> None:
        self.model_name = model_name
        self.slug = model_slug(model_name)
        self.cache_dir = Path(cache_dir) if cache_dir is not None else None
        dq, dp = default_prefixes(model_name)
        self.query_prefix = dq if query_prefix is None else query_prefix
        self.passage_prefix = dp if passage_prefix is None else passage_prefix

        if embed_fn is not None:
            self._embed = embed_fn
        else:
            # LAZY import: only here, only when no embed_fn is injected.
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(model_name)
            self._embed = lambda texts: model.encode(
                texts, batch_size=batch_size, convert_to_numpy=True,
                normalize_embeddings=True, show_progress_bar=False,
            )

        self.chunk_ids: List[str] = []
        self._chunk_emb: Optional[np.ndarray] = None
        self._id_to_idx: dict = {}

    # ------------------------------------------------------------------ index

    def _chunks_hash(self, chunks: Sequence[dict]) -> str:
        h = hashlib.sha256()
        h.update(self.passage_prefix.encode("utf-8"))
        for c in chunks:
            h.update(b"\x00")
            h.update(str(c["chunk_id"]).encode("utf-8"))
            h.update(b"\x01")
            h.update(c["chunk_text"].encode("utf-8"))
        return h.hexdigest()[:16]

    def cache_path(self, chunks: Sequence[dict]) -> Optional[Path]:
        if self.cache_dir is None:
            return None
        return self.cache_dir / f"{self.slug}_{self._chunks_hash(chunks)}.npy"

    def index_chunks(self, chunks: Sequence[dict]) -> None:
        """Embed (or load cached embeddings for) all chunk records."""
        self.chunk_ids = [c["chunk_id"] for c in chunks]
        self._id_to_idx = {cid: i for i, cid in enumerate(self.chunk_ids)}
        path = self.cache_path(chunks)
        if path is not None and path.exists():
            emb = np.load(path)
        else:
            texts = [self.passage_prefix + c["chunk_text"] for c in chunks]
            emb = _normalize_rows(self._embed(texts))
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.save(path, emb)
        if emb.shape[0] != len(chunks):
            raise ValueError(
                f"embedding cache {path} has {emb.shape[0]} rows for {len(chunks)} chunks"
            )
        self._chunk_emb = _normalize_rows(emb)

    # ------------------------------------------------------------------ score

    def embed_query(self, query: str) -> np.ndarray:
        vec = _normalize_rows(self._embed([self.query_prefix + query]))
        return vec[0]

    def score(self, query: str) -> np.ndarray:
        """Cosine similarity of ``query`` against every indexed chunk."""
        if self._chunk_emb is None:
            raise RuntimeError("call index_chunks() before score()")
        return self._chunk_emb @ self.embed_query(query)

    def rank(
        self, query: str, candidate_ids: Optional[Sequence[str]] = None
    ) -> List[Tuple[str, float]]:
        """Full ranking ``[(chunk_id, cosine), ...]`` best-first.

        ``candidate_ids`` restricts the ranking to a subset (text_restricted
        setting); the FULL candidate set is still returned, fully scored.
        Ties break by index order (stable sort).
        """
        scores = self.score(query)
        if candidate_ids is not None:
            idx = np.array([self._id_to_idx[cid] for cid in candidate_ids], dtype=np.int64)
        else:
            idx = np.arange(len(self.chunk_ids), dtype=np.int64)
        order = idx[np.argsort(-scores[idx], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in order]
