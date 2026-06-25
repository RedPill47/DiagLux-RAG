"""McNemar's exact test for paired system comparisons (plan Section 2.6).

All configurations are evaluated on the same 640 questions, so system comparisons
are paired: per question, each of two configs is right or wrong. McNemar's exact
test uses only the discordant pairs (n01 = A right / B wrong, n10 = A wrong /
B right) and an exact two-sided binomial test with p = 0.5.
"""

from __future__ import annotations

import math

import pandas as pd
from scipy.stats import binom

from .loading import SchemaError

__all__ = ["mcnemar_exact", "compare_configs", "significance_matrix"]


def mcnemar_exact(n01: int, n10: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts.

    Equivalent to ``statsmodels.stats.contingency_tables.mcnemar(..., exact=True)``:
    p = 2 * BinomCDF(min(n01, n10); n01 + n10, 0.5), capped at 1.
    Returns 1.0 when there are no discordant pairs.
    """
    if n01 < 0 or n10 < 0:
        raise ValueError("discordant counts must be non-negative")
    n = n01 + n10
    if n == 0:
        return 1.0
    p = 2.0 * float(binom.cdf(min(n01, n10), n, 0.5))
    return min(1.0, p)


def _correct_by_question(joined: pd.DataFrame, config_id: str) -> pd.Series:
    sub = joined[joined["config_id"] == config_id]
    if sub.empty:
        raise SchemaError(f"no predictions for config_id {config_id!r}")
    return sub.set_index("question_id")["is_correct"].astype(bool)


def compare_configs(joined: pd.DataFrame, config_a: str, config_b: str) -> dict:
    """Paired comparison of two configs on their common question set.

    Returns a dict with n (paired questions), accuracies, the discordant counts
    (n01: A correct & B wrong; n10: A wrong & B correct) and the exact McNemar
    p-value. Raises SchemaError if the configs share no questions.
    """
    a = _correct_by_question(joined, config_a)
    b = _correct_by_question(joined, config_b)
    common = a.index.intersection(b.index)
    if len(common) == 0:
        raise SchemaError(
            f"configs {config_a!r} and {config_b!r} share no questions; "
            "paired comparison undefined"
        )
    a, b = a.loc[common], b.loc[common]
    n01 = int((a & ~b).sum())
    n10 = int((~a & b).sum())
    return {
        "config_a": config_a,
        "config_b": config_b,
        "n": int(len(common)),
        "accuracy_a": float(a.mean()),
        "accuracy_b": float(b.mean()),
        "n01": n01,
        "n10": n10,
        "pvalue": mcnemar_exact(n01, n10),
    }


def significance_matrix(
    joined: pd.DataFrame, config_ids: list[str] | None = None
) -> pd.DataFrame:
    """Pairwise McNemar p-value matrix for the chosen configs (default: all).

    Symmetric, with NaN on the diagonal. Index and columns are config_ids.
    """
    if config_ids is None:
        config_ids = sorted(joined["config_id"].unique())
    mat = pd.DataFrame(math.nan, index=config_ids, columns=config_ids, dtype=float)
    for i, ca in enumerate(config_ids):
        for cb in config_ids[i + 1:]:
            p = compare_configs(joined, ca, cb)["pvalue"]
            mat.loc[ca, cb] = p
            mat.loc[cb, ca] = p
    mat.index.name = "config_id"
    return mat
