#!/usr/bin/env python
"""Generate the paper analysis tables from prediction logs (Phase 4).

Reads preds_*.jsonl run logs (+ .config.json sidecars), questions.jsonl, and
optionally rankings/chunk files, and writes every table to CSV + markdown under
--out (default outputs/analysis). Nothing is hand-assembled: all tables come
from the per-question logs (plan Section 5).

Examples:
    python scripts/analyze_results.py
    python scripts/analyze_results.py --tables accuracy confusion
    python scripts/analyze_results.py \
        --rankings outputs/retrieval/rankings_text_restricted_bm25_overlap.jsonl \
        --chunks outputs/processed/corpus_chunks_overlap.jsonl \
        --tables diagnostics
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from diaglux.analysis import (
    join_runs_questions,
    load_chunks,
    load_questions,
    load_rankings,
    load_runs,
    table_cognitive_breakdown,
    table_confusion,
    table_confusion_by_cognitive_type,
    table_linguistic_breakdown,
    table_retrieval_trap,
    table_retrieval_vs_answer,
    table_setting_accuracy,
    table_significance,
)

TABLE_GROUPS = ("accuracy", "confusion", "significance", "diagnostics")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="outputs/runs",
                   help="directory of preds_*.jsonl files (default: outputs/runs)")
    p.add_argument("--questions", default="outputs/processed/questions.jsonl",
                   help="questions.jsonl path (default: outputs/processed/questions.jsonl)")
    p.add_argument("--rankings", nargs="*", default=[], metavar="FILE",
                   help="rankings_*.jsonl file(s) for retrieval diagnostics")
    p.add_argument("--chunks", default=None, metavar="FILE",
                   help="corpus_chunks_*.jsonl file matching the rankings")
    p.add_argument("--out", default="outputs/analysis",
                   help="output directory (default: outputs/analysis)")
    p.add_argument("--tables", nargs="*", choices=list(TABLE_GROUPS) + ["all"],
                   default=["all"],
                   help="which table groups to produce (default: all)")
    p.add_argument("--n-boot", type=int, default=10_000,
                   help="bootstrap draws for CIs (default: 10000)")
    p.add_argument("--seed", type=int, default=12345,
                   help="bootstrap seed (default: 12345)")
    return p


def _diagnostics(joined, args, out_dir) -> None:
    if not args.rankings or not args.chunks:
        print("[diagnostics] skipped: --rankings and --chunks are both required.")
        return
    chunks = load_chunks(args.chunks)
    # Recover a per-question frame with span columns for the diagnostics API
    # (joined already carries the span columns per row).
    qcols = ["question_id", "text_id",
             "critical_start", "critical_end", "critical_status",
             "distractor_start", "distractor_end", "distractor_status"]
    questions = joined[qcols].drop_duplicates(subset="question_id")

    configs = joined[["config_id", "system", "setting", "k"]].drop_duplicates()
    for rf in args.rankings:
        rankings = load_rankings(rf)
        meta = rankings.attrs
        matched = configs[
            (configs["system"] == meta.get("method"))
            & (configs["setting"] == meta.get("setting"))
            & configs["k"].notna()
        ]
        if matched.empty:
            print(f"[diagnostics] {Path(rf).name}: no run config matches "
                  f"method={meta.get('method')!r} setting={meta.get('setting')!r}; skipped.")
            continue
        for cfg in matched.itertuples(index=False):
            preds = joined[joined["config_id"] == cfg.config_id]
            k = int(cfg.k)
            suffix = f"{cfg.config_id}_k{k}"
            table_retrieval_vs_answer(preds, rankings, chunks, questions, k,
                                      out_dir, name=f"retrieval_vs_answer_{suffix}")
            table_retrieval_trap(preds, rankings, chunks, questions, k,
                                 out_dir, name=f"retrieval_trap_{suffix}")
            print(f"[diagnostics] wrote retrieval_vs_answer_{suffix} and "
                  f"retrieval_trap_{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    groups = set(TABLE_GROUPS) if "all" in args.tables else set(args.tables)

    questions = load_questions(args.questions)
    preds = load_runs(args.runs_dir)
    joined = join_runs_questions(preds, questions)
    out_dir = Path(args.out)
    kw = {"n_boot": args.n_boot, "seed": args.seed}

    print(f"Loaded {joined['config_id'].nunique()} config(s), "
          f"{len(questions)} questions, {len(joined)} predictions.")

    if "accuracy" in groups:
        table_setting_accuracy(joined, out_dir, **kw)
        table_cognitive_breakdown(joined, out_dir, **kw)
        table_linguistic_breakdown(joined, out_dir, **kw)
        print("[accuracy] wrote setting_accuracy, cognitive_breakdown, "
              "linguistic_breakdown")
    if "confusion" in groups:
        table_confusion(joined, out_dir)
        table_confusion_by_cognitive_type(joined, out_dir)
        print("[confusion] wrote confusion, confusion_by_cognitive_type")
    if "significance" in groups:
        table_significance(joined, out_dir)
        print("[significance] wrote significance_matrix")
    if "diagnostics" in groups:
        _diagnostics(joined, args, out_dir)

    print(f"Tables written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
