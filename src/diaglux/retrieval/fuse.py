"""Hybrid fusion of retrieval rankings.

Two methods (review_and_plan Section 2.8):

- ``reciprocal_rank_fusion`` (PRIMARY hybrid). Rank-based, parameter-light,
  robust to incomparable score scales — the default hybrid for the paper.
- ``weighted_minmax_fusion`` (secondary). The supervisor's specified grid:
  per-query min-max normalization of each system's scores, then
  ``alpha * bm25 + (1 - alpha) * dense`` with alpha in {0.5, 0.7, 0.3}.
"""

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence, Tuple, Union

# A ranking is either an ordered sequence of doc ids, or an ordered sequence
# of (doc_id, score) pairs (the score is ignored by RRF — only ranks matter).
RankingLike = Sequence[Union[str, Tuple[str, float]]]


def _ids_only(ranking: RankingLike) -> List[str]:
    out: List[str] = []
    for item in ranking:
        if isinstance(item, tuple):
            out.append(item[0])
        else:
            out.append(item)
    return out


def reciprocal_rank_fusion(
    rankings: Sequence[RankingLike], k: int = 60
) -> List[Tuple[str, float]]:
    """Reciprocal Rank Fusion (Cormack et al., 2009).

    ``RRF(d) = sum_over_rankings 1 / (k + rank(d))`` with 1-based ranks and
    the standard ``k = 60``. Documents absent from a ranking simply receive
    no contribution from it.

    Returns the fused FULL ranking ``[(doc_id, rrf_score), ...]`` best-first.
    Ties break by first-appearance order across the input rankings (stable).
    """
    scores: Dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(_ids_only(ranking), start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # dicts preserve insertion order -> stable tie-break by first appearance
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _minmax(scores: Mapping[str, float], keys: Sequence[str]) -> Dict[str, float]:
    """Min-max normalize over the union ``keys``; ids missing from ``scores``
    are treated as scoring the observed minimum. A constant (or empty) score
    distribution normalizes to all-zeros (documented degenerate case)."""
    if not scores:
        return {key: 0.0 for key in keys}
    lo = min(scores.values())
    hi = max(scores.values())
    span = hi - lo
    if span == 0.0:
        return {key: 0.0 for key in keys}
    return {key: (scores.get(key, lo) - lo) / span for key in keys}


def weighted_minmax_fusion(
    bm25_scores: Mapping[str, float],
    dense_scores: Mapping[str, float],
    alpha: float,
) -> List[Tuple[str, float]]:
    """Weighted linear fusion of per-query min-max-normalized scores.

    ``fused(d) = alpha * norm(bm25)(d) + (1 - alpha) * norm(dense)(d)``

    over the UNION of documents present in either score map (a document
    missing from one system is assigned that system's minimum, i.e. 0 after
    normalization). Supervisor's grid: alpha in {0.5, 0.7, 0.3}.

    Returns the fused full ranking ``[(doc_id, fused_score), ...]``
    best-first; ties break by bm25-map-then-dense-map insertion order.
    """
    if not 0.0 <= alpha <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    keys = list(bm25_scores)
    seen = set(keys)
    keys.extend(key for key in dense_scores if key not in seen)
    b_norm = _minmax(bm25_scores, keys)
    d_norm = _minmax(dense_scores, keys)
    fused = {key: alpha * b_norm[key] + (1.0 - alpha) * d_norm[key] for key in keys}
    return sorted(fused.items(), key=lambda kv: -kv[1])
