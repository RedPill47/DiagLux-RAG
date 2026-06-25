"""Chosen-option-type confusion analysis (plan Section 2.5).

Each wrong option encodes a failure hypothesis (misunderstand = comprehension
failure, distractor_span = lure failure, no_support = hallucinated support).
This module computes the distribution of ``semantic_choice`` -- including
``unparseable`` outputs as their own category, per plan Section 2.11 -- by
system, setting, k, and cognitive type.
"""

from __future__ import annotations

import pandas as pd

from .loading import CONFIG_COLS

__all__ = ["CHOICE_ORDER", "confusion_table", "confusion_by_cognitive_type"]

CHOICE_ORDER = ["correct", "misunderstand", "distractor_span", "no_support", "unparseable"]


def _choice_series(joined: pd.DataFrame) -> pd.Series:
    """semantic_choice with null (unparseable output) mapped to 'unparseable'."""
    return joined["semantic_choice"].fillna("unparseable")


def confusion_table(
    joined: pd.DataFrame, group_cols: list[str] | None = None
) -> pd.DataFrame:
    """Counts and within-group fractions of each chosen option type.

    Default grouping is one row per run configuration (system, setting, k, model).
    Output columns: group cols, n, then count_<type> and pct_<type> for each of
    correct / misunderstand / distractor_span / no_support / unparseable.
    """
    if group_cols is None:
        group_cols = CONFIG_COLS
    df = joined.copy()
    df["_choice"] = _choice_series(df)
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        n = len(g)
        counts = g["_choice"].value_counts()
        row = {**dict(zip(group_cols, keys)), "n": int(n)}
        for choice in CHOICE_ORDER:
            c = int(counts.get(choice, 0))
            row[f"count_{choice}"] = c
            row[f"pct_{choice}"] = c / n if n else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def confusion_by_cognitive_type(joined: pd.DataFrame) -> pd.DataFrame:
    """Confusion distribution per run configuration x cognitive type."""
    return confusion_table(joined, CONFIG_COLS + ["cognitive_type"])
