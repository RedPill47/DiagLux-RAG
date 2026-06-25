"""Accuracy tables and bootstrap CIs on constructed prediction sets."""

import math

import numpy as np
import pytest

from _analysis_fixtures import make_pred, make_questions, write_preds, write_questions
from diaglux.analysis.accuracy import (
    accuracy_by_cognitive_type,
    accuracy_by_linguistic_category,
    bootstrap_ci,
    overall_accuracy,
)
from diaglux.analysis.loading import join_runs_questions, load_questions, load_runs

N_BOOT = 1000  # small but plenty for test stability (seeded anyway)


@pytest.fixture()
def joined(tmp_path):
    questions = make_questions()
    # Vary cognitive types: 2x Retrieve, 2x Interpret.
    questions[2]["cognitive_type"] = "Interpret"
    questions[3]["cognitive_type"] = "Interpret"
    # One question carries an extra linguistic category.
    questions[0]["linguistic_categories"] = ["LEX", "SYN"]
    qpath = tmp_path / "questions.jsonl"
    from _analysis_fixtures import write_jsonl

    write_jsonl(qpath, questions)

    # 3 correct out of 4 -> accuracy 0.75 (the wrong one is "misunderstand").
    rows = [
        make_pred(questions[0], questions[0]["gold_letter"]),
        make_pred(questions[1], questions[1]["gold_letter"]),
        make_pred(questions[2], questions[2]["gold_letter"]),
        make_pred(questions[3],
                  chr(65 + questions[3]["permutation"].index("misunderstand"))),
    ]
    write_preds(tmp_path / "runs", "acc1", rows)
    runs = load_runs(tmp_path / "runs")
    return join_runs_questions(runs, load_questions(qpath))


def test_known_accuracy_with_ci_bracketing_truth(joined):
    table = overall_accuracy(joined, n_boot=N_BOOT)
    assert len(table) == 1
    row = table.iloc[0]
    assert row["n"] == 4  # cell size reported
    assert row["n_correct"] == 3
    assert row["accuracy"] == pytest.approx(0.75)
    # The bootstrap CI must bracket the point estimate / true value.
    assert row["ci_low"] <= 0.75 <= row["ci_high"]
    assert 0.0 <= row["ci_low"] < row["ci_high"] <= 1.0


def test_cell_sizes_by_cognitive_type(joined):
    table = accuracy_by_cognitive_type(joined, n_boot=200)
    by_type = table.set_index("cognitive_type")
    assert by_type.loc["Retrieve", "n"] == 2
    assert by_type.loc["Interpret", "n"] == 2
    assert by_type.loc["Retrieve", "accuracy"] == pytest.approx(1.0)
    assert by_type.loc["Interpret", "accuracy"] == pytest.approx(0.5)


def test_linguistic_category_multi_membership_cell_sizes(joined):
    table = accuracy_by_linguistic_category(joined, n_boot=200)
    by_cat = table.set_index("linguistic_category")
    # Every question has LEX; question 0 additionally has SYN.
    assert by_cat.loc["LEX", "n"] == 4
    assert by_cat.loc["SYN", "n"] == 1
    assert by_cat.loc["SYN", "accuracy"] == pytest.approx(1.0)


def test_bootstrap_ci_brackets_true_value_on_large_sample():
    rng = np.random.default_rng(42)
    correct = rng.random(400) < 0.6  # empirical mean close to 0.6
    lo, hi = bootstrap_ci(correct, n_boot=2000, seed=7)
    assert lo < correct.mean() < hi
    assert lo < 0.6 < hi
    assert 0.0 <= lo < hi <= 1.0


def test_bootstrap_ci_is_deterministic_given_seed():
    correct = [1, 1, 1, 0, 0, 1, 0, 1]
    assert bootstrap_ci(correct, n_boot=500, seed=11) == bootstrap_ci(
        correct, n_boot=500, seed=11
    )


def test_bootstrap_ci_degenerate_cells():
    lo, hi = bootstrap_ci([], n_boot=100)
    assert math.isnan(lo) and math.isnan(hi)
    lo, hi = bootstrap_ci([1, 1, 1], n_boot=100)
    assert lo == hi == 1.0
