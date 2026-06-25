"""Accuracy tables with bootstrap confidence intervals.

Implements the answer-level metrics of plan Section 4.2: accuracy overall and by
system / setting / k / cognitive type / linguistic category, each with cell sizes
and seeded bootstrap 95% CIs (resampling questions, 10k draws by default), per the
statistical-reporting requirements of plan Section 2.6.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from .loading import CONFIG_COLS

__all__ = [
    "bootstrap_ci",
    "accuracy_table",
    "overall_accuracy",
    "accuracy_by_system",
    "accuracy_by_setting",
    "accuracy_by_k",
    "accuracy_by_cognitive_type",
    "accuracy_by_linguistic_category",
]

N_BOOT_DEFAULT = 10_000
SEED_DEFAULT = 12345


def bootstrap_ci(
    correct,
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for an accuracy, resampling questions with replacement.

    ``correct`` is a boolean/0-1 array with one entry per question. Returns
    (low, high) for a (1 - alpha) interval; (nan, nan) for an empty cell.
    """
    arr = np.asarray(correct, dtype=float)
    n = arr.size
    if n == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    means = arr[idx].mean(axis=1)
    lo, hi = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return float(lo), float(hi)


def accuracy_table(
    df: pd.DataFrame,
    group_cols: list[str],
    n_boot: int = N_BOOT_DEFAULT,
    seed: int = SEED_DEFAULT,
) -> pd.DataFrame:
    """Accuracy per group with cell size (n), n_correct, and bootstrap 95% CI."""
    rows = []
    for keys, g in df.groupby(group_cols, dropna=False, sort=True):
        if not isinstance(keys, tuple):
            keys = (keys,)
        correct = g["is_correct"].to_numpy(dtype=bool)
        lo, hi = bootstrap_ci(correct, n_boot=n_boot, seed=seed)
        rows.append(
            {
                **dict(zip(group_cols, keys)),
                "n": int(len(g)),
                "n_correct": int(correct.sum()),
                "accuracy": float(correct.mean()),
                "ci_low": lo,
                "ci_high": hi,
            }
        )
    return pd.DataFrame(rows)


def overall_accuracy(joined: pd.DataFrame, **kw) -> pd.DataFrame:
    """One row per run configuration (system x setting x k x model)."""
    return accuracy_table(joined, CONFIG_COLS, **kw)


def accuracy_by_system(joined: pd.DataFrame, **kw) -> pd.DataFrame:
    return accuracy_table(joined, ["system", "model"], **kw)


def accuracy_by_setting(joined: pd.DataFrame, **kw) -> pd.DataFrame:
    return accuracy_table(joined, ["setting", "system", "model"], **kw)


def accuracy_by_k(joined: pd.DataFrame, **kw) -> pd.DataFrame:
    return accuracy_table(joined, ["system", "setting", "model", "k"], **kw)


def accuracy_by_cognitive_type(joined: pd.DataFrame, **kw) -> pd.DataFrame:
    return accuracy_table(joined, CONFIG_COLS + ["cognitive_type"], **kw)


def accuracy_by_linguistic_category(joined: pd.DataFrame, **kw) -> pd.DataFrame:
    """Accuracy per linguistic category.

    A question counts for category X if X is in its ``linguistic_categories`` list,
    so a question may contribute to several categories; cell sizes (n) are reported
    alongside every figure. Questions with no categories are excluded here.
    """
    exploded = joined.explode("linguistic_categories").rename(
        columns={"linguistic_categories": "linguistic_category"}
    )
    exploded = exploded[exploded["linguistic_category"].notna()]
    return accuracy_table(exploded, CONFIG_COLS + ["linguistic_category"], **kw)
