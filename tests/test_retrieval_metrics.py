"""Evidence/source/distractor metrics on hand-computed overlap examples."""

import pytest

from _retrieval_fixtures import make_chunk, make_question
from diaglux.retrieval.metrics import (
    compute_metrics,
    metrics_to_csv,
    metrics_to_markdown,
    span_overlap_chars,
    span_resolved,
)

# Hand-designed geometry (all offsets in text1's clean-text coordinates):
#   cA = [0, 10)   no overlap with critical span [10, 20) (end-exclusive!)
#   cB = [15, 30)  overlap 5 chars = 50% of the span      -> cov50 hit
#   cC = [18, 40)  overlap 2 chars = 20% of the span      -> any-overlap hit only
# distractor span [35, 50): only cC overlaps (5 chars).
CHUNKS = [
    make_chunk("text1_overlap_c000", "text1", "aaaa", 0, 10, 4),
    make_chunk("text1_overlap_c001", "text1", "bbbb", 15, 30, 5),
    make_chunk("text1_overlap_c002", "text1", "cccc", 18, 40, 6),
    # same offsets as cB but in ANOTHER text: must never count as evidence
    make_chunk("text2_overlap_c000", "text2", "zzzz", 15, 30, 3),
]

QX = make_question(
    "text1_q00", "text1", "Q?",
    critical_span={"start": 10, "end": 20, "status": "exact"},
    distractor_span={"start": 35, "end": 50, "status": "fuzzy"},
)
QY = make_question(  # unresolved spans -> skipped for evidence + distractor
    "text1_q01", "text1", "Q?",
    critical_span={"start": None, "end": None, "status": "unresolved"},
    distractor_span={"start": None, "end": None, "status": "empty"},
)


def ranking_record(question_id, chunk_ids):
    return {
        "question_id": question_id,
        "setting": "text_restricted",
        "method": "bm25",
        "query_mode": "question_only",
        "chunk_strategy": "overlap",
        "ranking": [
            {"chunk_id": cid, "score": float(10 - i), "rank": i + 1}
            for i, cid in enumerate(chunk_ids)
        ],
    }


RANKINGS = [
    # qx: cA(r1, miss), cC(r2, any-hit), cB(r3, cov50-hit)
    ranking_record("text1_q00", ["text1_overlap_c000", "text1_overlap_c002", "text1_overlap_c001"]),
    ranking_record("text1_q01", ["text1_overlap_c000", "text1_overlap_c001", "text1_overlap_c002"]),
]


def get(rows, metric, criterion, k):
    for row in rows:
        if (row["metric"], row["criterion"], row["k"]) == (metric, criterion, k):
            return row
    raise KeyError((metric, criterion, k))


def test_span_overlap_chars():
    assert span_overlap_chars(0, 10, 10, 20) == 0   # touching, end-exclusive
    assert span_overlap_chars(15, 30, 10, 20) == 5
    assert span_overlap_chars(18, 40, 10, 20) == 2
    assert span_overlap_chars(0, 100, 10, 20) == 10  # containment


def test_span_resolved():
    assert span_resolved({"start": 1, "end": 5, "status": "exact"})
    assert not span_resolved({"start": None, "end": None, "status": "unresolved"})
    assert not span_resolved({"start": 3, "end": 9, "status": "empty"})
    assert not span_resolved({"start": 5, "end": 5, "status": "exact"})  # zero length
    assert not span_resolved(None)


def test_evidence_recall_and_mrr_any_overlap():
    result = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))
    rows = result["rows"]
    # only qx eligible: first any-overlap hit at rank 2
    assert get(rows, "evidence_recall", "any", 1)["value"] == 0.0
    assert get(rows, "evidence_recall", "any", 3)["value"] == 1.0
    assert get(rows, "evidence_mrr", "any", None)["value"] == pytest.approx(0.5)
    assert get(rows, "evidence_recall", "any", 1)["n"] == 1


def test_evidence_cov50_variant():
    rows = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))["rows"]
    # cC covers only 20% -> first cov50 hit is cB at rank 3
    assert get(rows, "evidence_recall", "cov50", 1)["value"] == 0.0
    assert get(rows, "evidence_recall", "cov50", 3)["value"] == 1.0
    assert get(rows, "evidence_mrr", "cov50", None)["value"] == pytest.approx(1 / 3)


def test_unresolved_spans_are_skipped_and_reported():
    result = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))
    assert result["skipped"]["evidence_span_skipped"] == 1
    assert result["skipped"]["distractor_span_skipped"] == 1


def test_distractor_retrieval_rate():
    rows = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))["rows"]
    # qx only; cC (the only distractor-overlapping chunk) sits at rank 2
    assert get(rows, "distractor_retrieval_rate", "any", 1)["value"] == 0.0
    assert get(rows, "distractor_retrieval_rate", "any", 3)["value"] == 1.0


def test_source_text_recall_and_mrr():
    rows = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))["rows"]
    assert get(rows, "source_text_recall", "text_id", 1)["value"] == 1.0
    assert get(rows, "source_text_mrr", "text_id", None)["value"] == 1.0


def test_other_text_chunk_never_counts_as_evidence():
    # text2 chunk shares cB's offsets and is ranked FIRST -> must be ignored
    rankings = [ranking_record("text1_q00", ["text2_overlap_c000", "text1_overlap_c001"])]
    rows = compute_metrics(rankings, [QX], CHUNKS, ks=(1, 3))["rows"]
    assert get(rows, "evidence_recall", "any", 1)["value"] == 0.0
    assert get(rows, "evidence_mrr", "any", None)["value"] == pytest.approx(0.5)


def test_context_token_length_stats():
    rows = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))["rows"]
    # qx top-1 = cA (4 tokens); qy top-1 = cA (4) -> mean 4
    assert get(rows, "context_tokens_mean", "topk_sum", 1)["value"] == 4.0
    # top-3 for both = 4 + 6 + 5 = 15
    assert get(rows, "context_tokens_mean", "topk_sum", 3)["value"] == 15.0
    assert get(rows, "context_tokens_min", "topk_sum", 3)["value"] == 15.0
    assert get(rows, "context_tokens_max", "topk_sum", 3)["value"] == 15.0


def test_outputs_csv_and_markdown(tmp_path):
    result = compute_metrics(RANKINGS, [QX, QY], CHUNKS, ks=(1, 3))
    csv_path = metrics_to_csv(result, tmp_path / "m.csv")
    text = csv_path.read_text(encoding="utf-8")
    assert "evidence_recall" in text and "bm25" in text
    md = metrics_to_markdown(result)
    assert "| evidence_recall | any | 1 |" in md
    assert "Skipped for evidence metrics" in md
