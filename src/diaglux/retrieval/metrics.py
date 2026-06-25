"""Retrieval metrics computed post hoc from rankings files (docs/CONTRACTS.md).

Primary metrics (review_and_plan Section 2.4 — evidence-span metrics are the
paper's primary retrieval metrics):

- **Evidence Recall@k**: fraction of eligible questions for which at least one
  of the top-k chunks overlaps the question's ``critical_span``. Two overlap
  criteria are reported: ``any`` (>= 1 character of overlap; the default
  threshold) and ``cov50`` (the chunk covers >= 50% of the span's characters).
- **Evidence MRR**: mean reciprocal rank (over the FULL ranking) of the first
  chunk satisfying the overlap criterion; 0 when no chunk does.
- Questions whose ``critical_span`` is unresolved/empty/null (status in
  {unresolved, empty}, null offsets, or non-positive length) are SKIPPED for
  evidence metrics and the skip count is reported.

Secondary metrics:

- **Source-text Recall@k / MRR**: top-k contains a chunk whose ``text_id``
  equals the question's. Informative only in the ``open_corpus`` setting (it
  is 1.0 by construction in ``text_restricted``); computed regardless, with
  the setting recorded so the analysis layer can ignore the trivial case.
- **Distractor-span retrieval rate@k** (retrieval-trap analysis, Section
  2.5/2.7): among questions with a resolved ``distractor_span``, the fraction
  for which any top-k chunk overlaps it (any-overlap criterion).
- **Retrieved-context token length** (Section 2.10 — does top-k reconstruct
  the whole text?): per k, the sum of ``n_tokens`` over the top-k chunks;
  mean / median / min / max across questions.

All chunk/span offsets are character offsets into the clean text body (shared
coordinate system per the contract). Overlap compares chunk
``[start_char, end_char)`` with span ``[start, end)``. In ``open_corpus``,
a chunk from a different text NEVER counts as span overlap.

k in {1, 3, 5, 10} by default; top-k views are slices of the full ranking.
"""

from __future__ import annotations

import csv
import statistics
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

DEFAULT_KS = (1, 3, 5, 10)
_UNRESOLVED_STATUSES = {"unresolved", "empty"}


# ------------------------------------------------------------------ overlap

def span_overlap_chars(
    chunk_start: int, chunk_end: int, span_start: int, span_end: int
) -> int:
    """Characters shared by ``[chunk_start, chunk_end)`` and ``[span_start, span_end)``."""
    return max(0, min(chunk_end, span_end) - max(chunk_start, span_start))


def span_resolved(span: Optional[Mapping]) -> bool:
    """True if a span record has usable offsets (see module docstring)."""
    if not span:
        return False
    if span.get("status") in _UNRESOLVED_STATUSES:
        return False
    start, end = span.get("start"), span.get("end")
    return start is not None and end is not None and end > start


def _first_hit_rank(
    ranking: Sequence[Mapping],
    chunk_lookup: Mapping[str, dict],
    span: Mapping,
    text_id: str,
    min_coverage: float,
) -> Optional[int]:
    """1-based rank of the first chunk overlapping ``span``; None if none does.

    ``min_coverage`` = 0.0 means any (> 0 chars) overlap; 0.5 means the chunk
    must cover >= 50% of the span's characters. Chunks from other texts never
    match (open-corpus safety).
    """
    span_len = span["end"] - span["start"]
    for entry in ranking:
        chunk = chunk_lookup.get(entry["chunk_id"])
        if chunk is None or chunk["text_id"] != text_id:
            continue
        overlap = span_overlap_chars(
            chunk["start_char"], chunk["end_char"], span["start"], span["end"]
        )
        if overlap <= 0:
            continue
        if overlap / span_len >= min_coverage:
            return entry["rank"]
    return None


def _first_source_text_rank(
    ranking: Sequence[Mapping], chunk_lookup: Mapping[str, dict], text_id: str
) -> Optional[int]:
    for entry in ranking:
        chunk = chunk_lookup.get(entry["chunk_id"])
        if chunk is not None and chunk["text_id"] == text_id:
            return entry["rank"]
    return None


# ------------------------------------------------------------------ compute

def compute_metrics(
    rankings: Sequence[dict],
    questions: Sequence[dict],
    chunks: Sequence[dict],
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Compute all retrieval metrics for one rankings file.

    Returns ``{"meta": {...}, "rows": [tidy rows], "skipped": {...}}`` where
    each tidy row is ``{"metric", "criterion", "k", "value", "n"}``
    (``k`` is None for MRR rows, which use the full ranking).
    """
    q_lookup = {q["question_id"]: q for q in questions}
    chunk_lookup = {c["chunk_id"]: c for c in chunks}
    ks = sorted(ks)

    evidence_hits: Dict[str, List[Optional[int]]] = {"any": [], "cov50": []}
    distractor_hits: List[Optional[int]] = []
    source_hits: List[Optional[int]] = []
    context_tokens: Dict[int, List[int]] = {k: [] for k in ks}
    n_skipped_evidence = 0
    n_skipped_distractor = 0
    n_missing_questions = 0

    for rec in rankings:
        question = q_lookup.get(rec["question_id"])
        if question is None:
            n_missing_questions += 1
            continue
        ranking = rec["ranking"]
        text_id = question["text_id"]

        # Evidence (critical span)
        crit = question.get("critical_span")
        if span_resolved(crit):
            evidence_hits["any"].append(
                _first_hit_rank(ranking, chunk_lookup, crit, text_id, 0.0)
            )
            evidence_hits["cov50"].append(
                _first_hit_rank(ranking, chunk_lookup, crit, text_id, 0.5)
            )
        else:
            n_skipped_evidence += 1

        # Distractor span (any overlap)
        dist = question.get("distractor_span")
        if span_resolved(dist):
            distractor_hits.append(
                _first_hit_rank(ranking, chunk_lookup, dist, text_id, 0.0)
            )
        else:
            n_skipped_distractor += 1

        # Source text
        source_hits.append(_first_source_text_rank(ranking, chunk_lookup, text_id))

        # Context token length of the top-k retrieved chunks
        for k in ks:
            total = 0
            for entry in ranking[:k]:
                chunk = chunk_lookup.get(entry["chunk_id"])
                if chunk is not None:
                    total += chunk.get("n_tokens") or 0
            context_tokens[k].append(total)

    def recall_at(hits: List[Optional[int]], k: int) -> Optional[float]:
        if not hits:
            return None
        return sum(1 for r in hits if r is not None and r <= k) / len(hits)

    def mrr(hits: List[Optional[int]]) -> Optional[float]:
        if not hits:
            return None
        return sum(1.0 / r for r in hits if r is not None) / len(hits)

    rows: List[dict] = []

    def add(metric: str, criterion: str, k: Optional[int], value: Optional[float], n: int):
        if value is not None:
            rows.append(
                {"metric": metric, "criterion": criterion, "k": k,
                 "value": round(value, 6), "n": n}
            )

    for criterion in ("any", "cov50"):
        hits = evidence_hits[criterion]
        for k in ks:
            add("evidence_recall", criterion, k, recall_at(hits, k), len(hits))
        add("evidence_mrr", criterion, None, mrr(hits), len(hits))
    for k in ks:
        add("source_text_recall", "text_id", k, recall_at(source_hits, k), len(source_hits))
    add("source_text_mrr", "text_id", None, mrr(source_hits), len(source_hits))
    for k in ks:
        add("distractor_retrieval_rate", "any", k,
            recall_at(distractor_hits, k), len(distractor_hits))
    for k in ks:
        lengths = context_tokens[k]
        if lengths:
            add("context_tokens_mean", "topk_sum", k, statistics.fmean(lengths), len(lengths))
            add("context_tokens_median", "topk_sum", k, statistics.median(lengths), len(lengths))
            add("context_tokens_min", "topk_sum", k, min(lengths), len(lengths))
            add("context_tokens_max", "topk_sum", k, max(lengths), len(lengths))

    meta = {}
    if rankings:
        first = rankings[0]
        meta = {key: first.get(key) for key in ("setting", "method", "query_mode", "chunk_strategy")}
    meta["n_questions_scored"] = len(rankings) - n_missing_questions

    return {
        "meta": meta,
        "rows": rows,
        "skipped": {
            "evidence_span_skipped": n_skipped_evidence,
            "distractor_span_skipped": n_skipped_distractor,
            "rankings_without_question": n_missing_questions,
        },
    }


# ------------------------------------------------------------------- output

def metrics_to_csv(result: dict, path) -> Path:
    """Write the tidy rows (plus meta columns) as CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    meta = result["meta"]
    meta_cols = ["setting", "method", "query_mode", "chunk_strategy"]
    fieldnames = meta_cols + ["metric", "criterion", "k", "value", "n"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in result["rows"]:
            writer.writerow({**{c: meta.get(c, "") for c in meta_cols}, **row})
    return path


def metrics_to_markdown(result: dict) -> str:
    """Render a markdown summary table (plus skip counts)."""
    meta = result["meta"]
    lines = [
        f"### Retrieval metrics — setting=`{meta.get('setting')}` "
        f"method=`{meta.get('method')}` query_mode=`{meta.get('query_mode')}` "
        f"strategy=`{meta.get('chunk_strategy')}`",
        "",
        "| metric | criterion | k | value | n |",
        "|---|---|---|---|---|",
    ]
    for row in result["rows"]:
        k = "" if row["k"] is None else row["k"]
        lines.append(
            f"| {row['metric']} | {row['criterion']} | {k} | {row['value']:.4f} | {row['n']} |"
        )
    skipped = result["skipped"]
    lines += [
        "",
        f"Skipped for evidence metrics (unresolved/empty/null critical_span): "
        f"{skipped['evidence_span_skipped']}; "
        f"skipped for distractor rate: {skipped['distractor_span_skipped']}; "
        f"rankings without matching question: {skipped['rankings_without_question']}.",
    ]
    return "\n".join(lines)
