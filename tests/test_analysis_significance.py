"""McNemar's exact test against hand-computed values."""

import math

import pandas as pd
import pytest

from diaglux.analysis.loading import SchemaError
from diaglux.analysis.significance import (
    compare_configs,
    mcnemar_exact,
    significance_matrix,
)


def test_mcnemar_hand_computed_b8_c2():
    # b=8, c=2 discordant pairs; exact two-sided binomial:
    # p = 2 * sum_{i=0}^{2} C(10, i) / 2^10 = 2 * (1 + 10 + 45) / 1024 = 0.109375
    assert mcnemar_exact(8, 2) == pytest.approx(0.109375)
    assert mcnemar_exact(2, 8) == pytest.approx(0.109375)  # symmetric


def test_mcnemar_extremes():
    assert mcnemar_exact(0, 0) == 1.0  # no discordant pairs
    assert mcnemar_exact(5, 5) == 1.0  # balanced -> capped at 1
    # b=10, c=0: p = 2 * C(10,0)/2^10 = 2/1024
    assert mcnemar_exact(10, 0) == pytest.approx(2.0 / 1024.0)


def test_mcnemar_rejects_negative_counts():
    with pytest.raises(ValueError):
        mcnemar_exact(-1, 3)


def _joined_two_configs():
    """12 paired questions: A right & B wrong on 8 (n01), A wrong & B right
    on 2 (n10), both right on 1, both wrong on 1."""
    a_correct = [True] * 8 + [False] * 2 + [True, False]
    b_correct = [False] * 8 + [True] * 2 + [True, False]
    rows = []
    for i, (ca, cb) in enumerate(zip(a_correct, b_correct)):
        qid = f"text1_q{i:02d}"
        rows.append({"config_id": "cfgA", "question_id": qid, "is_correct": ca})
        rows.append({"config_id": "cfgB", "question_id": qid, "is_correct": cb})
    return pd.DataFrame(rows)


def test_compare_configs_counts_discordant_pairs():
    result = compare_configs(_joined_two_configs(), "cfgA", "cfgB")
    assert result["n"] == 12
    assert result["n01"] == 8  # A correct, B wrong
    assert result["n10"] == 2  # A wrong, B correct
    assert result["accuracy_a"] == pytest.approx(9 / 12)
    assert result["accuracy_b"] == pytest.approx(3 / 12)
    assert result["pvalue"] == pytest.approx(0.109375)


def test_compare_configs_pairs_on_common_questions_only():
    joined = _joined_two_configs()
    # Drop two of cfgB's questions; the comparison must pair on the rest.
    joined = joined[~((joined["config_id"] == "cfgB")
                      & (joined["question_id"].isin(["text1_q10", "text1_q11"])))]
    result = compare_configs(joined, "cfgA", "cfgB")
    assert result["n"] == 10
    assert result["n01"] == 8
    assert result["n10"] == 2


def test_compare_configs_unknown_config_fails():
    with pytest.raises(SchemaError, match="no predictions"):
        compare_configs(_joined_two_configs(), "cfgA", "nope")


def test_significance_matrix_symmetric_nan_diagonal():
    mat = significance_matrix(_joined_two_configs())
    assert list(mat.index) == ["cfgA", "cfgB"]
    assert math.isnan(mat.loc["cfgA", "cfgA"])
    assert math.isnan(mat.loc["cfgB", "cfgB"])
    assert mat.loc["cfgA", "cfgB"] == pytest.approx(0.109375)
    assert mat.loc["cfgA", "cfgB"] == mat.loc["cfgB", "cfgA"]
