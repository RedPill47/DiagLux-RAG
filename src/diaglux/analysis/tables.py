"""Render the paper tables to CSV + GitHub markdown under outputs/analysis/.

One function per paper table, named after the plan sections they serve:

- :func:`table_setting_accuracy`    -- concept Sections 6.1/6.2, plan Section 4.2:
  accuracy by system x setting x k (with cell sizes and bootstrap CIs).
- :func:`table_cognitive_breakdown` -- plan Sections 2.6/4.2: accuracy per
  cognitive type per configuration.
- :func:`table_linguistic_breakdown`-- plan Sections 2.6/4.2: accuracy per
  linguistic category (multi-membership; cell sizes reported).
- :func:`table_confusion`           -- plan Section 2.5: chosen-option-type
  distribution by configuration (and by cognitive type).
- :func:`table_significance`        -- plan Section 2.6: pairwise McNemar
  exact-test p-value matrix.
- :func:`table_retrieval_vs_answer` -- plan Sections 2.4/4.2: the 2x2
  retrieval-success x answer-correctness diagnostic.
- :func:`table_retrieval_trap`      -- plan Sections 2.5/4.2: the retrieval-trap
  table (distractor retrieved & critical missed -> distractor option chosen?).

Every function writes ``<name>.csv`` and ``<name>.md`` (plus ``<name>.meta.json``
for the diagnostics, which carry skip counts) and returns the DataFrame.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from . import accuracy as acc
from . import confusion as conf
from . import diagnostics as diag
from . import significance as sig

__all__ = [
    "df_to_markdown",
    "write_table",
    "table_setting_accuracy",
    "table_cognitive_breakdown",
    "table_linguistic_breakdown",
    "table_confusion",
    "table_confusion_by_cognitive_type",
    "table_significance",
    "table_retrieval_vs_answer",
    "table_retrieval_trap",
]


def _fmt(v, float_digits: int = 3) -> str:
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return f"{v:.{float_digits}f}"
    return str(v)


def df_to_markdown(df: pd.DataFrame, float_digits: int = 3) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no extra deps)."""
    cols = [str(c) for c in df.columns]
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join("---" for _ in cols) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(_fmt(v, float_digits) for v in row) + " |")
    return "\n".join(lines)


def write_table(
    df: pd.DataFrame,
    out_dir: str | Path,
    name: str,
    title: str | None = None,
    notes: list[str] | None = None,
    meta: dict | None = None,
) -> pd.DataFrame:
    """Write ``<name>.csv`` and ``<name>.md`` (and ``<name>.meta.json`` if meta)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / f"{name}.csv", index=False)
    parts = []
    if title:
        parts.append(f"# {title}\n")
    parts.append(df_to_markdown(df))
    for note in notes or []:
        parts.append(f"\n{note}")
    (out_dir / f"{name}.md").write_text("\n".join(parts) + "\n", encoding="utf-8")
    if meta is not None:
        (out_dir / f"{name}.meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8")
    return df


_CI_NOTE = ("n = questions per cell; CI = bootstrap 95% interval "
            "(questions resampled with replacement, seeded).")


def table_setting_accuracy(joined: pd.DataFrame, out_dir, **kw) -> pd.DataFrame:
    """Accuracy by system x setting x k (concept 6.1/6.2; plan 4.2)."""
    df = acc.overall_accuracy(joined, **kw)
    return write_table(df, out_dir, "setting_accuracy",
                       title="Accuracy by system, setting, and k",
                       notes=[_CI_NOTE])


def table_cognitive_breakdown(joined: pd.DataFrame, out_dir, **kw) -> pd.DataFrame:
    """Accuracy per cognitive type per configuration (plan 2.6/4.2)."""
    df = acc.accuracy_by_cognitive_type(joined, **kw)
    return write_table(df, out_dir, "cognitive_breakdown",
                       title="Accuracy by cognitive type",
                       notes=[_CI_NOTE])


def table_linguistic_breakdown(joined: pd.DataFrame, out_dir, **kw) -> pd.DataFrame:
    """Accuracy per linguistic category per configuration (plan 2.6/4.2)."""
    df = acc.accuracy_by_linguistic_category(joined, **kw)
    return write_table(
        df, out_dir, "linguistic_breakdown",
        title="Accuracy by linguistic category",
        notes=[_CI_NOTE,
               "A question counts for every category appearing in its "
               "linguistic_categories, so cells overlap; small cells are "
               "qualitative observations, not claims (plan Section 2.6)."])


def table_confusion(joined: pd.DataFrame, out_dir) -> pd.DataFrame:
    """Chosen-option-type distribution per configuration (plan 2.5)."""
    df = conf.confusion_table(joined)
    return write_table(df, out_dir, "confusion",
                       title="Chosen option type by configuration",
                       notes=["unparseable = output yielded no A-D letter "
                              "(tracked separately, scored incorrect)."])


def table_confusion_by_cognitive_type(joined: pd.DataFrame, out_dir) -> pd.DataFrame:
    """Chosen-option-type distribution per configuration x cognitive type (plan 2.5)."""
    df = conf.confusion_by_cognitive_type(joined)
    return write_table(df, out_dir, "confusion_by_cognitive_type",
                       title="Chosen option type by configuration and cognitive type")


def table_significance(
    joined: pd.DataFrame, out_dir, config_ids: list[str] | None = None
) -> pd.DataFrame:
    """Pairwise McNemar exact p-value matrix (plan 2.6)."""
    mat = sig.significance_matrix(joined, config_ids).reset_index()
    return write_table(
        mat, out_dir, "significance_matrix",
        title="Pairwise McNemar exact test p-values",
        notes=["Paired on the common question set of each pair; "
               "exact two-sided binomial test on discordant pairs."])


def table_retrieval_vs_answer(
    preds: pd.DataFrame, rankings: pd.DataFrame, chunks: pd.DataFrame,
    questions: pd.DataFrame, k: int, out_dir, name: str = "retrieval_vs_answer",
) -> pd.DataFrame:
    """The 2x2 retrieval-success x answer-correctness table (plan 2.4/4.2)."""
    df, meta = diag.retrieval_vs_answer(preds, rankings, chunks, questions, k)
    note = (f"k={meta['k']}; evaluated n={meta['n_evaluated']}; skipped "
            f"{meta['n_skipped_unresolved_span']} question(s) with "
            "unresolved/empty critical span.")
    return write_table(df, out_dir, name,
                       title="Retrieval success vs. answer correctness",
                       notes=[note], meta=meta)


def table_retrieval_trap(
    preds: pd.DataFrame, rankings: pd.DataFrame, chunks: pd.DataFrame,
    questions: pd.DataFrame, k: int, out_dir, name: str = "retrieval_trap",
) -> pd.DataFrame:
    """The retrieval-trap table (plan 2.5/4.2)."""
    df, meta = diag.retrieval_trap(preds, rankings, chunks, questions, k)
    note = (f"k={meta['k']}; evaluated n={meta['n_evaluated']}; skipped "
            f"{meta['n_skipped_unresolved_span']} question(s) with "
            "unresolved/empty critical or distractor span.")
    return write_table(df, out_dir, name,
                       title="Retrieval trap: distractor retrieved vs. option chosen",
                       notes=[note], meta=meta)
