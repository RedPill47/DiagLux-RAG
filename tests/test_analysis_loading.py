"""Loading and schema validation: fail loudly on contract violations."""

import pandas as pd
import pytest

from _analysis_fixtures import (
    make_pred,
    make_questions,
    write_jsonl,
    write_preds,
    write_questions,
)
from diaglux.analysis.loading import (
    SchemaError,
    join_runs_questions,
    load_preds_file,
    load_questions,
    load_runs,
)


@pytest.fixture()
def runs_setup(tmp_path):
    questions = make_questions()
    qpath = tmp_path / "questions.jsonl"
    write_questions(qpath)
    runs_dir = tmp_path / "runs"
    # Config "aaa": all four correct (oracle).
    rows_a = [make_pred(q, q["gold_letter"], system="oracle") for q in questions]
    # Config "bbb": 2 correct, 1 wrong, 1 unparseable (bm25 @ k=5).
    rows_b = [
        make_pred(questions[0], questions[0]["gold_letter"],
                  system="bm25", setting="text_restricted", k=5),
        make_pred(questions[1], questions[1]["gold_letter"],
                  system="bm25", setting="text_restricted", k=5),
        make_pred(questions[2],
                  chr(65 + questions[2]["permutation"].index("misunderstand")),
                  system="bm25", setting="text_restricted", k=5),
        make_pred(questions[3], None,
                  system="bm25", setting="text_restricted", k=5),
    ]
    write_preds(runs_dir, "aaa", rows_a)
    write_preds(runs_dir, "bbb", rows_b)
    return {"questions": questions, "qpath": qpath, "runs_dir": runs_dir}


def test_load_questions_and_runs_roundtrip(runs_setup):
    questions = load_questions(runs_setup["qpath"])
    assert len(questions) == 4
    # Flattened span convenience columns are present.
    for col in ("critical_start", "critical_end", "critical_status",
                "distractor_start", "distractor_end", "distractor_status"):
        assert col in questions.columns
    assert questions["critical_status"].tolist() == ["exact"] * 4

    runs = load_runs(runs_setup["runs_dir"])
    assert len(runs) == 8
    assert sorted(runs["config_id"].unique()) == ["aaa", "bbb"]
    assert set(runs.attrs["configs"]) == {"aaa", "bbb"}

    joined = join_runs_questions(runs, questions)
    assert len(joined) == 8
    assert "cognitive_type" in joined.columns
    # Per-config accuracy is what we constructed.
    acc = joined.groupby("config_id")["is_correct"].mean()
    assert acc["aaa"] == 1.0
    assert acc["bbb"] == 0.5


def test_load_preds_file_reads_sidecar(runs_setup):
    df = load_preds_file(runs_setup["runs_dir"] / "preds_bbb.jsonl")
    assert df.attrs["config"]["config_id"] == "bbb"
    assert df["config_id"].unique().tolist() == ["bbb"]
    # k is a nullable int column.
    assert df["k"].tolist() == [5, 5, 5, 5]


def test_missing_pred_key_fails_loudly(tmp_path, runs_setup):
    questions = runs_setup["questions"]
    row = make_pred(questions[0], "A")
    del row["parse_status"]
    runs_dir = tmp_path / "runs2"
    write_preds(runs_dir, "ccc", [row])
    with pytest.raises(SchemaError, match="missing required field.*parse_status"):
        load_preds_file(runs_dir / "preds_ccc.jsonl")


def test_missing_question_key_fails_loudly(tmp_path):
    records = make_questions()
    del records[1]["permutation"]
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    with pytest.raises(SchemaError, match="missing required field.*permutation"):
        load_questions(path)


def test_missing_sidecar_fails_loudly(tmp_path, runs_setup):
    row = make_pred(runs_setup["questions"][0], "A")
    runs_dir = tmp_path / "runs3"
    write_preds(runs_dir, "ddd", [row], write_sidecar=False)
    with pytest.raises(SchemaError, match="config sidecar"):
        load_preds_file(runs_dir / "preds_ddd.jsonl")


def test_unparseable_with_is_correct_true_rejected(tmp_path, runs_setup):
    row = make_pred(runs_setup["questions"][0], None)
    row["is_correct"] = True
    runs_dir = tmp_path / "runs4"
    write_preds(runs_dir, "eee", [row])
    with pytest.raises(SchemaError, match="is_correct"):
        load_preds_file(runs_dir / "preds_eee.jsonl")


def test_gold_letter_permutation_mismatch_rejected(tmp_path):
    records = make_questions()
    records[0]["gold_letter"] = "A"  # permutation says B
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    with pytest.raises(SchemaError, match="gold_letter"):
        load_questions(path)


def test_join_rejects_semantic_choice_inconsistent_with_permutation(runs_setup, tmp_path):
    q = runs_setup["questions"][0]  # letter A -> distractor_span
    row = make_pred(q, "A")
    row["semantic_choice"] = "misunderstand"  # internally consistent lie
    row["is_correct"] = False
    runs_dir = tmp_path / "runs5"
    write_preds(runs_dir, "fff", [row])
    preds = load_preds_file(runs_dir / "preds_fff.jsonl")  # passes file-level checks
    questions = load_questions(runs_setup["qpath"])
    with pytest.raises(SchemaError, match="does not match permutation"):
        join_runs_questions(preds, questions)


def test_join_rejects_unknown_question_id(runs_setup, tmp_path):
    q = dict(runs_setup["questions"][0])
    q["question_id"] = "text9_q99"
    runs_dir = tmp_path / "runs6"
    write_preds(runs_dir, "ggg", [make_pred(q, "A")])
    preds = load_preds_file(runs_dir / "preds_ggg.jsonl")
    questions = load_questions(runs_setup["qpath"])
    with pytest.raises(SchemaError, match="absent from"):
        join_runs_questions(preds, questions)


# ---------------------------------------------------------------------------
# REGRESSION: in_title spans (resolved status, null offsets) must be accepted.
# ---------------------------------------------------------------------------

def test_regression_in_title_span_with_null_offsets_accepted(tmp_path):
    records = make_questions()
    records[0]["critical_span"] = {
        "start": None, "end": None, "status": "exact", "in_title": True,
    }
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    df = load_questions(path)  # must not raise
    row = df.set_index("question_id").loc[records[0]["question_id"]]
    assert row["critical_status"] == "exact"
    assert pd.isna(row["critical_start"])
    assert pd.isna(row["critical_end"])


def test_in_title_span_with_offsets_rejected(tmp_path):
    records = make_questions()
    records[0]["critical_span"] = {
        "start": 5, "end": 10, "status": "exact", "in_title": True,
    }
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    with pytest.raises(SchemaError, match="in_title span must have null start/end"):
        load_questions(path)


def test_resolved_span_with_null_offsets_rejected_without_in_title(tmp_path):
    records = make_questions()
    records[0]["critical_span"] = {"start": None, "end": None, "status": "exact"}
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    with pytest.raises(SchemaError, match="integer start/end"):
        load_questions(path)


def test_unresolved_span_with_offsets_rejected(tmp_path):
    records = make_questions()
    records[0]["distractor_span"] = {"start": 3, "end": 9, "status": "unresolved"}
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    with pytest.raises(SchemaError, match="must be null when status"):
        load_questions(path)


def test_unresolved_span_with_null_offsets_accepted(tmp_path):
    records = make_questions()
    records[0]["critical_span"] = {"start": None, "end": None, "status": "unresolved"}
    records[1]["distractor_span"] = {"start": None, "end": None, "status": "empty"}
    path = write_jsonl(tmp_path / "questions.jsonl", records)
    df = load_questions(path)
    assert len(df) == 4
