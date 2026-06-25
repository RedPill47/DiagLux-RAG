"""Retrieval-vs-answer and retrieval-trap diagnostics on hand-built geometry.

Chunk layout (text1 clean-text coordinates, end-exclusive):
    c000 = [0, 10)    c001 = [15, 30)    c002 = [35, 50)
plus text2_overlap_c000 = [15, 30) in ANOTHER text (must never count).

Questions (k=2 everywhere):
    q00 critical [10,20) dist [35,45)  top2={c001,c000} -> critical_only
    q01 critical [40,45) dist [16,18)  top2={c000,c001} -> distractor_only (trap)
    q02 critical [0,5)   dist [20,25)  top2={c000,c001} -> both_retrieved
    q03 critical [16,18) dist [30,34)  top2={text2_c000,c000} -> neither
        (text2_c000 overlaps the offsets but belongs to the wrong text)
    q04 critical unresolved (null offsets)            -> skipped
    q05 critical in_title (resolved status, null offsets) -> skipped (REGRESSION)
"""

import math

import pandas as pd
import pytest

from _analysis_fixtures import write_jsonl
from answering_testutils import make_question
from diaglux.analysis.diagnostics import (
    TRAP_GROUPS,
    retrieval_trap,
    retrieval_vs_answer,
    span_is_resolved,
    topk_span_hits,
)
from diaglux.analysis.loading import (
    SchemaError,
    load_chunks,
    load_questions,
    load_rankings,
)

K = 2

CHUNK_DEFS = [
    ("text1_overlap_c000", "text1", 0, 10),
    ("text1_overlap_c001", "text1", 15, 30),
    ("text1_overlap_c002", "text1", 35, 50),
    ("text2_overlap_c000", "text2", 15, 30),
]

SPANS = {  # qid -> (critical_span, distractor_span)
    "text1_q00": ({"start": 10, "end": 20, "status": "exact"},
                  {"start": 35, "end": 45, "status": "fuzzy"}),
    "text1_q01": ({"start": 40, "end": 45, "status": "exact"},
                  {"start": 16, "end": 18, "status": "exact"}),
    "text1_q02": ({"start": 0, "end": 5, "status": "dehyphen"},
                  {"start": 20, "end": 25, "status": "exact"}),
    "text1_q03": ({"start": 16, "end": 18, "status": "exact"},
                  {"start": 30, "end": 34, "status": "exact"}),
    "text1_q04": ({"start": None, "end": None, "status": "unresolved"},
                  {"start": 16, "end": 18, "status": "fuzzy"}),
    "text1_q05": ({"start": None, "end": None, "status": "exact", "in_title": True},
                  {"start": None, "end": None, "status": "empty"}),
}

RANK_ORDERS = {
    "text1_q00": ["text1_overlap_c001", "text1_overlap_c000",
                  "text1_overlap_c002", "text2_overlap_c000"],
    "text1_q01": ["text1_overlap_c000", "text1_overlap_c001",
                  "text1_overlap_c002", "text2_overlap_c000"],
    "text1_q02": ["text1_overlap_c000", "text1_overlap_c001",
                  "text1_overlap_c002", "text2_overlap_c000"],
    "text1_q03": ["text2_overlap_c000", "text1_overlap_c000",
                  "text1_overlap_c001", "text1_overlap_c002"],
    "text1_q04": ["text1_overlap_c001", "text1_overlap_c000",
                  "text1_overlap_c002", "text2_overlap_c000"],
    "text1_q05": ["text1_overlap_c000", "text1_overlap_c001",
                  "text1_overlap_c002", "text2_overlap_c000"],
}

PERM = ["distractor_span", "correct", "no_support", "misunderstand"]

# qid -> (semantic_choice, is_correct) for the prediction set under test.
PRED_CHOICES = {
    "text1_q00": ("correct", True),
    "text1_q01": ("distractor_span", False),  # fell for the trap
    "text1_q02": ("correct", True),
    "text1_q03": ("no_support", False),
    "text1_q04": ("correct", True),   # skipped (unresolved critical span)
    "text1_q05": ("misunderstand", False),  # skipped (in_title critical span)
}


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("diag")
    question_records = []
    for qid, (crit, dist) in SPANS.items():
        rec = make_question(qid, "text1", f"Fro {qid}?", PERM)
        rec["critical_span"] = crit
        rec["distractor_span"] = dist
        question_records.append(rec)
    questions = load_questions(
        write_jsonl(tmp_path / "questions.jsonl", question_records))

    chunk_records = [
        {"chunk_id": cid, "text_id": tid, "chunk_text": f"Chunk {cid}.",
         "start_char": s, "end_char": e, "n_tokens": 5}
        for cid, tid, s, e in CHUNK_DEFS
    ]
    chunks = load_chunks(write_jsonl(tmp_path / "chunks.jsonl", chunk_records))

    ranking_records = [
        {"question_id": qid, "setting": "text_restricted", "method": "bm25",
         "query_mode": "question_options", "chunk_strategy": "overlap",
         "ranking": [{"chunk_id": cid, "score": float(10 - i), "rank": i + 1}
                     for i, cid in enumerate(order)]}
        for qid, order in RANK_ORDERS.items()
    ]
    rankings = load_rankings(write_jsonl(tmp_path / "rankings.jsonl", ranking_records))

    preds = pd.DataFrame([
        {"question_id": qid, "semantic_choice": choice, "is_correct": correct}
        for qid, (choice, correct) in PRED_CHOICES.items()
    ])
    return questions, chunks, rankings, preds


def test_span_is_resolved_handles_none_and_nan():
    assert span_is_resolved("exact", 10, 20)
    assert span_is_resolved("fuzzy", 0, 1)
    assert not span_is_resolved("unresolved", None, None)
    assert not span_is_resolved("empty", None, None)
    # REGRESSION: resolved status with null offsets (in_title spans) must be
    # treated as unresolved, both as raw None and as pandas NaN.
    assert not span_is_resolved("exact", None, None)
    assert not span_is_resolved("exact", math.nan, math.nan)


def test_topk_critical_hits_known_geometry(fixtures):
    questions, chunks, rankings, _preds = fixtures
    hits, skipped = topk_span_hits(rankings, chunks, questions, K, span="critical")
    assert hits.to_dict() == {
        "text1_q00": True,   # c001 overlaps [10,20)
        "text1_q01": False,  # c002 (rank 3) holds the span
        "text1_q02": True,   # c000 overlaps [0,5)
        "text1_q03": False,  # text2 chunk overlaps offsets but wrong text
    }
    assert skipped == ["text1_q04", "text1_q05"]


def test_topk_distractor_hits_known_geometry(fixtures):
    questions, chunks, rankings, _preds = fixtures
    hits, skipped = topk_span_hits(rankings, chunks, questions, K, span="distractor")
    assert hits.to_dict() == {
        "text1_q00": False,  # distractor [35,45) only in c002 (rank 3)
        "text1_q01": True,   # c001 overlaps [16,18)
        "text1_q02": True,   # c001 overlaps [20,25)
        "text1_q03": False,  # gap [30,35): touches c001's end, end-exclusive
        "text1_q04": True,   # c001 (rank 1) overlaps [16,18)
    }
    assert skipped == ["text1_q05"]


def test_retrieval_vs_answer_2x2(fixtures):
    questions, chunks, rankings, preds = fixtures
    table, meta = retrieval_vs_answer(preds, rankings, chunks, questions, K)
    rows = table.set_index("retrieval")
    retrieved = rows.loc["critical_span_retrieved"]  # q00, q02 (both correct)
    assert retrieved["n"] == 2
    assert retrieved["n_correct"] == 2
    assert retrieved["n_incorrect"] == 0
    assert retrieved["accuracy"] == pytest.approx(1.0)
    missed = rows.loc["critical_span_missed"]  # q01, q03 (both wrong)
    assert missed["n"] == 2
    assert missed["n_correct"] == 0
    assert missed["accuracy"] == pytest.approx(0.0)
    assert meta["k"] == K
    assert meta["n_evaluated"] == 4
    assert meta["n_skipped_unresolved_span"] == 2
    assert meta["skipped_question_ids"] == ["text1_q04", "text1_q05"]


def test_retrieval_trap_partition(fixtures):
    questions, chunks, rankings, preds = fixtures
    table, meta = retrieval_trap(preds, rankings, chunks, questions, K)
    assert table["group"].tolist() == TRAP_GROUPS
    rows = table.set_index("group")

    trap = rows.loc["distractor_only"]  # q01: chose the distractor option
    assert trap["n"] == 1
    assert trap["n_chose_distractor"] == 1
    assert trap["pct_chose_distractor"] == pytest.approx(1.0)
    assert trap["accuracy"] == pytest.approx(0.0)

    assert rows.loc["both_retrieved", "n"] == 1       # q02
    assert rows.loc["both_retrieved", "accuracy"] == pytest.approx(1.0)
    assert rows.loc["critical_only", "n"] == 1        # q00
    assert rows.loc["critical_only", "n_chose_distractor"] == 0
    assert rows.loc["neither_retrieved", "n"] == 1    # q03
    assert rows.loc["neither_retrieved", "n_chose_distractor"] == 0

    assert meta["n_evaluated"] == 4
    assert meta["n_skipped_unresolved_span"] == 2
    assert meta["skipped_question_ids"] == ["text1_q04", "text1_q05"]


def test_regression_null_span_questions_are_skipped_not_crashed(fixtures):
    """Questions with NaN/null span offsets (unresolved, empty, or in_title)
    must be skipped and counted -- never crash the diagnostics."""
    questions, chunks, rankings, preds = fixtures
    only_null = preds[preds["question_id"].isin(["text1_q04", "text1_q05"])]
    table, meta = retrieval_vs_answer(only_null, rankings, chunks, questions, K)
    assert meta["n_evaluated"] == 0
    assert meta["n_skipped_unresolved_span"] == 2
    assert table["n"].tolist() == [0, 0]
    assert all(math.isnan(a) for a in table["accuracy"])


def test_duplicate_question_ids_rejected(fixtures):
    questions, chunks, rankings, preds = fixtures
    doubled = pd.concat([preds, preds], ignore_index=True)
    with pytest.raises(SchemaError, match="duplicate"):
        retrieval_vs_answer(doubled, rankings, chunks, questions, K)
    with pytest.raises(SchemaError, match="duplicate"):
        retrieval_trap(doubled, rankings, chunks, questions, K)


def test_missing_ranking_entry_fails_loudly(fixtures):
    questions, chunks, rankings, preds = fixtures
    broken = rankings.drop(index="text1_q00")
    with pytest.raises(SchemaError, match="no entry for question"):
        retrieval_vs_answer(preds, broken, chunks, questions, K)


def test_unknown_chunk_in_ranking_fails_loudly(fixtures):
    questions, chunks, rankings, preds = fixtures
    broken = chunks.drop(index="text1_overlap_c001")
    with pytest.raises(SchemaError, match="unknown chunk_id"):
        retrieval_vs_answer(preds, rankings, broken, questions, K)
