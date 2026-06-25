"""Retrieval-vs-comprehension diagnostics (plan Sections 2.4, 2.5, 4.2).

(a) The 2x2 retrieval-success x answer-correctness table. Retrieval success at k
    is defined span-level (plan Section 2.4): at least one top-k chunk overlaps the
    question's critical span by at least one character, in the same text's clean-text
    coordinate system (a chunk only counts if its ``text_id`` matches the question's).
    Questions whose critical span is unresolved/empty are skipped and counted.

(b) The retrieval-trap table (plan Section 2.5): partition questions by whether the
    distractor span and/or the critical span were retrieved in the top-k, and report
    per group how often the model chose the distractor_span option (and its accuracy).
    Requires both spans resolved; skipped questions are counted.
"""

from __future__ import annotations

import pandas as pd

from .loading import RESOLVED_SPAN_STATUSES, SchemaError

__all__ = [
    "span_is_resolved",
    "topk_span_hits",
    "retrieval_vs_answer",
    "retrieval_trap",
    "TRAP_GROUPS",
]

#: Group labels for the retrieval-trap partition, in report order.
TRAP_GROUPS = ["distractor_only", "both_retrieved", "critical_only", "neither_retrieved"]


def span_is_resolved(status, start, end) -> bool:
    # start/end arrive as None from raw JSON but as NaN from pandas DataFrames
    return status in RESOLVED_SPAN_STATUSES and pd.notna(start) and pd.notna(end)


def _topk_chunk_ids(rankings: pd.DataFrame, question_id: str, k: int) -> list[str]:
    try:
        full = rankings.loc[question_id, "ranking"]
    except KeyError:
        raise SchemaError(
            f"rankings file has no entry for question {question_id!r}; cannot compute "
            "retrieval diagnostics for this prediction set"
        ) from None
    return list(full[:k])


def _chunk_overlaps(chunks: pd.DataFrame, chunk_id: str, text_id: str,
                    span_start: int, span_end: int) -> bool:
    try:
        row = chunks.loc[chunk_id]
    except KeyError:
        raise SchemaError(f"ranking references unknown chunk_id {chunk_id!r}") from None
    if row["text_id"] != text_id:
        return False
    return int(row["start_char"]) < span_end and int(row["end_char"]) > span_start


def topk_span_hits(
    rankings: pd.DataFrame,
    chunks: pd.DataFrame,
    questions: pd.DataFrame,
    k: int,
    span: str = "critical",
    question_ids=None,
) -> tuple[pd.Series, list[str]]:
    """Per-question flag: does any top-k chunk overlap the chosen span?

    ``span`` is ``"critical"`` or ``"distractor"``. Returns (hits, skipped) where
    ``hits`` is a boolean Series indexed by question_id covering questions with a
    resolved span, and ``skipped`` lists question_ids with unresolved/empty spans.
    """
    if span not in ("critical", "distractor"):
        raise ValueError("span must be 'critical' or 'distractor'")
    qdf = questions
    if question_ids is not None:
        qdf = qdf[qdf["question_id"].isin(set(question_ids))]
    hits: dict[str, bool] = {}
    skipped: list[str] = []
    for q in qdf.itertuples(index=False):
        status = getattr(q, f"{span}_status")
        start = getattr(q, f"{span}_start")
        end = getattr(q, f"{span}_end")
        if not span_is_resolved(status, start, end):
            skipped.append(q.question_id)
            continue
        top = _topk_chunk_ids(rankings, q.question_id, k)
        hits[q.question_id] = any(
            _chunk_overlaps(chunks, cid, q.text_id, int(start), int(end)) for cid in top
        )
    return pd.Series(hits, dtype=bool, name=f"{span}_retrieved"), skipped


def retrieval_vs_answer(
    preds: pd.DataFrame,
    rankings: pd.DataFrame,
    chunks: pd.DataFrame,
    questions: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, dict]:
    """The 2x2 retrieval-success x answer-correctness table for one prediction set.

    ``preds`` must contain a single configuration (one row per question with
    ``question_id`` and ``is_correct``). Returns (table, meta): the table has one
    row per retrieval outcome (retrieved / missed) with n, n_correct, n_incorrect
    and accuracy; meta reports k and the skipped (unresolved/empty critical span)
    question count.
    """
    if preds["question_id"].duplicated().any():
        raise SchemaError("retrieval_vs_answer expects a single configuration "
                          "(duplicate question_ids found)")
    hits, skipped = topk_span_hits(
        rankings, chunks, questions, k, span="critical",
        question_ids=preds["question_id"],
    )
    correct = preds.set_index("question_id")["is_correct"].astype(bool)
    rows = []
    for flag, label in ((True, "critical_span_retrieved"), (False, "critical_span_missed")):
        qids = hits.index[hits == flag]
        c = correct.loc[qids]
        n = int(len(c))
        rows.append({
            "retrieval": label,
            "n": n,
            "n_correct": int(c.sum()),
            "n_incorrect": int(n - c.sum()),
            "accuracy": float(c.mean()) if n else float("nan"),
        })
    meta = {
        "k": k,
        "n_evaluated": int(len(hits)),
        "n_skipped_unresolved_span": len(skipped),
        "skipped_question_ids": sorted(skipped),
    }
    return pd.DataFrame(rows), meta


def retrieval_trap(
    preds: pd.DataFrame,
    rankings: pd.DataFrame,
    chunks: pd.DataFrame,
    questions: pd.DataFrame,
    k: int,
) -> tuple[pd.DataFrame, dict]:
    """The retrieval-trap table for one prediction set (plan Section 2.5).

    Partitions questions (both spans resolved) into:

    - ``distractor_only``: distractor span retrieved in top-k AND critical span not
      (the trap condition);
    - ``both_retrieved``, ``critical_only``, ``neither_retrieved``: the complements.

    For each group reports n, how many/what fraction chose the distractor_span
    option, and accuracy. Returns (table, meta) with skip counts in meta.
    """
    if preds["question_id"].duplicated().any():
        raise SchemaError("retrieval_trap expects a single configuration "
                          "(duplicate question_ids found)")
    qids = preds["question_id"]
    crit_hits, crit_skipped = topk_span_hits(
        rankings, chunks, questions, k, span="critical", question_ids=qids)
    dist_hits, dist_skipped = topk_span_hits(
        rankings, chunks, questions, k, span="distractor", question_ids=qids)
    common = crit_hits.index.intersection(dist_hits.index)
    skipped = sorted(set(crit_skipped) | set(dist_skipped))

    p = preds.set_index("question_id")
    chose_distractor = (p["semantic_choice"] == "distractor_span")
    correct = p["is_correct"].astype(bool)

    def group_of(qid: str) -> str:
        d, c = bool(dist_hits.loc[qid]), bool(crit_hits.loc[qid])
        if d and not c:
            return "distractor_only"
        if d and c:
            return "both_retrieved"
        if c:
            return "critical_only"
        return "neither_retrieved"

    membership = {g: [] for g in TRAP_GROUPS}
    for qid in common:
        membership[group_of(qid)].append(qid)

    rows = []
    for g in TRAP_GROUPS:
        ids = membership[g]
        n = len(ids)
        n_trap = int(chose_distractor.loc[ids].sum()) if n else 0
        n_corr = int(correct.loc[ids].sum()) if n else 0
        rows.append({
            "group": g,
            "n": n,
            "n_chose_distractor": n_trap,
            "pct_chose_distractor": n_trap / n if n else float("nan"),
            "n_correct": n_corr,
            "accuracy": n_corr / n if n else float("nan"),
        })
    meta = {
        "k": k,
        "n_evaluated": int(len(common)),
        "n_skipped_unresolved_span": len(skipped),
        "skipped_question_ids": skipped,
    }
    return pd.DataFrame(rows), meta
