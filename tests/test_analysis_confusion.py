"""Semantic-choice confusion distributions, with unparseable as its own bin."""

import pandas as pd
import pytest

from diaglux.analysis.confusion import (
    CHOICE_ORDER,
    confusion_by_cognitive_type,
    confusion_table,
)

CFG = {"config_id": "x1", "system": "bm25", "setting": "text_restricted",
       "k": 5, "model": "mock-model"}


def _joined_frame():
    """10 predictions for one configuration:
    4 correct, 3 misunderstand, 2 distractor_span, 1 unparseable (None)."""
    choices = (["correct"] * 4 + ["misunderstand"] * 3
               + ["distractor_span"] * 2 + [None])
    cognitive = ["Retrieve"] * 5 + ["Interpret"] * 5
    rows = []
    for i, (choice, cog) in enumerate(zip(choices, cognitive)):
        rows.append({
            **CFG,
            "question_id": f"text1_q{i:02d}",
            "semantic_choice": choice,
            "is_correct": choice == "correct",
            "cognitive_type": cog,
        })
    return pd.DataFrame(rows)


def test_confusion_percentages_hand_computed():
    table = confusion_table(_joined_frame())
    assert len(table) == 1
    row = table.iloc[0]
    assert row["n"] == 10
    assert row["count_correct"] == 4
    assert row["pct_correct"] == pytest.approx(0.4)
    assert row["count_misunderstand"] == 3
    assert row["pct_misunderstand"] == pytest.approx(0.3)
    assert row["count_distractor_span"] == 2
    assert row["pct_distractor_span"] == pytest.approx(0.2)
    assert row["count_no_support"] == 0
    assert row["pct_no_support"] == 0.0


def test_unparseable_is_its_own_category():
    row = confusion_table(_joined_frame()).iloc[0]
    assert "unparseable" in CHOICE_ORDER
    assert row["count_unparseable"] == 1
    assert row["pct_unparseable"] == pytest.approx(0.1)


def test_counts_and_percentages_are_exhaustive():
    row = confusion_table(_joined_frame()).iloc[0]
    assert sum(row[f"count_{c}"] for c in CHOICE_ORDER) == row["n"]
    assert sum(row[f"pct_{c}"] for c in CHOICE_ORDER) == pytest.approx(1.0)


def test_confusion_by_cognitive_type_splits_groups():
    table = confusion_by_cognitive_type(_joined_frame())
    assert len(table) == 2
    by_type = table.set_index("cognitive_type")
    # Retrieve rows: 4 correct + 1 misunderstand.
    assert by_type.loc["Retrieve", "n"] == 5
    assert by_type.loc["Retrieve", "count_correct"] == 4
    assert by_type.loc["Retrieve", "pct_correct"] == pytest.approx(0.8)
    # Interpret rows: 2 misunderstand + 2 distractor_span + 1 unparseable.
    assert by_type.loc["Interpret", "n"] == 5
    assert by_type.loc["Interpret", "count_correct"] == 0
    assert by_type.loc["Interpret", "count_unparseable"] == 1


def test_confusion_separates_configurations():
    df1 = _joined_frame()
    df2 = df1.copy()
    df2["config_id"] = "x2"
    df2["semantic_choice"] = "no_support"
    table = confusion_table(pd.concat([df1, df2], ignore_index=True))
    assert len(table) == 2
    by_cfg = table.set_index("config_id")
    assert by_cfg.loc["x2", "count_no_support"] == 10
    assert by_cfg.loc["x2", "pct_no_support"] == pytest.approx(1.0)
