"""DiagLux-RAG analysis module (Phase 4): accuracy tables, CIs, McNemar,
confusion and retrieval diagnostics. See docs/CONTRACTS.md for the input schemas
and DiagLux_RAG_review_and_plan.md Sections 2.5, 2.6, 4.2 for the table specs."""

from .loading import (
    SchemaError,
    CONFIG_COLS,
    load_questions,
    load_preds_file,
    load_runs,
    load_rankings,
    load_chunks,
    join_runs_questions,
)
from .accuracy import (
    bootstrap_ci,
    accuracy_table,
    overall_accuracy,
    accuracy_by_system,
    accuracy_by_setting,
    accuracy_by_k,
    accuracy_by_cognitive_type,
    accuracy_by_linguistic_category,
)
from .significance import mcnemar_exact, compare_configs, significance_matrix
from .confusion import CHOICE_ORDER, confusion_table, confusion_by_cognitive_type
from .diagnostics import (
    TRAP_GROUPS,
    topk_span_hits,
    retrieval_vs_answer,
    retrieval_trap,
)
from .tables import (
    df_to_markdown,
    write_table,
    table_setting_accuracy,
    table_cognitive_breakdown,
    table_linguistic_breakdown,
    table_confusion,
    table_confusion_by_cognitive_type,
    table_significance,
    table_retrieval_vs_answer,
    table_retrieval_trap,
)

__all__ = [
    "SchemaError", "CONFIG_COLS",
    "load_questions", "load_preds_file", "load_runs", "load_rankings",
    "load_chunks", "join_runs_questions",
    "bootstrap_ci", "accuracy_table", "overall_accuracy", "accuracy_by_system",
    "accuracy_by_setting", "accuracy_by_k", "accuracy_by_cognitive_type",
    "accuracy_by_linguistic_category",
    "mcnemar_exact", "compare_configs", "significance_matrix",
    "CHOICE_ORDER", "confusion_table", "confusion_by_cognitive_type",
    "TRAP_GROUPS", "topk_span_hits", "retrieval_vs_answer", "retrieval_trap",
    "df_to_markdown", "write_table",
    "table_setting_accuracy", "table_cognitive_breakdown",
    "table_linguistic_breakdown", "table_confusion",
    "table_confusion_by_cognitive_type", "table_significance",
    "table_retrieval_vs_answer", "table_retrieval_trap",
]
