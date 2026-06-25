#!/usr/bin/env python
"""Run retrieval (Phase 3) and/or compute retrieval metrics.

Examples — the standard grid (bm25 + hybrid_rrf, both settings, both query modes):

    python scripts/run_retrieval.py --method bm25       --setting text_restricted --query-mode question_only
    python scripts/run_retrieval.py --method bm25       --setting text_restricted --query-mode question_options
    python scripts/run_retrieval.py --method bm25       --setting open_corpus     --query-mode question_only
    python scripts/run_retrieval.py --method bm25       --setting open_corpus     --query-mode question_options
    python scripts/run_retrieval.py --method hybrid_rrf --setting text_restricted --query-mode question_only
    python scripts/run_retrieval.py --method hybrid_rrf --setting open_corpus     --query-mode question_options
    ... (hybrid_rrf over the remaining setting x query-mode cells)

Dense / weighted hybrid (requires sentence-transformers; lazily imported):

    python scripts/run_retrieval.py --method dense    --model intfloat/multilingual-e5-base --setting open_corpus
    python scripts/run_retrieval.py --method dense    --model BAAI/bge-m3 --setting open_corpus
    python scripts/run_retrieval.py --method hybrid_w --alpha 0.5 --setting open_corpus
    python scripts/run_retrieval.py --method hybrid_w --alpha 0.7 --setting open_corpus
    python scripts/run_retrieval.py --method hybrid_w --alpha 0.3 --setting open_corpus

Metrics only, from an existing rankings file:

    python scripts/run_retrieval.py --metrics-only \
        --rankings outputs/retrieval/rankings_text_restricted_bm25_overlap_question_options.jsonl

By default metrics are also computed right after a retrieval run and written
next to the rankings file as ``metrics_<rankings stem>.csv`` (plus a markdown
summary on stdout). Disable with --no-metrics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow running as a plain script from the repo root even without `pip install -e`.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from diaglux.retrieval import metrics as metrics_mod
from diaglux.retrieval import search
from diaglux.retrieval.tokenize import get_tokenizer


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid_rrf", "hybrid_w"],
                        help="retrieval method (required unless --metrics-only)")
    parser.add_argument("--model", default="intfloat/multilingual-e5-base",
                        help="sentence-transformers model for dense/hybrid "
                             "(candidates: intfloat/multilingual-e5-base, BAAI/bge-m3)")
    parser.add_argument("--alpha", type=float, default=None,
                        help="bm25 weight for hybrid_w (grid: 0.5, 0.7, 0.3)")
    parser.add_argument("--setting", choices=list(search.SETTINGS),
                        help="text_restricted | open_corpus (required unless --metrics-only)")
    parser.add_argument("--chunks", default="outputs/processed/corpus_chunks_overlap.jsonl",
                        help="corpus chunks jsonl (contract schema)")
    parser.add_argument("--questions", default="outputs/processed/questions.jsonl",
                        help="questions jsonl (contract schema)")
    parser.add_argument("--query-mode", choices=list(search.QUERY_MODES),
                        default="question_options", help="query construction ablation")
    parser.add_argument("--analyzer", choices=["word", "char_ngram"], default="word",
                        help="BM25 analyzer (char_ngram = subword ablation)")
    parser.add_argument("--rrf-k", type=int, default=60, help="RRF constant k")
    parser.add_argument("--out-dir", default=str(search.DEFAULT_OUT_DIR),
                        help="output directory for rankings/metrics files")
    parser.add_argument("--metrics-only", action="store_true",
                        help="compute metrics from an existing rankings file (--rankings)")
    parser.add_argument("--rankings", default=None,
                        help="existing rankings file (with --metrics-only)")
    parser.add_argument("--no-metrics", action="store_true",
                        help="skip the metrics computation after a retrieval run")
    args = parser.parse_args(argv)

    if args.metrics_only:
        if not args.rankings:
            parser.error("--metrics-only requires --rankings FILE")
    else:
        if not args.method or not args.setting:
            parser.error("--method and --setting are required (unless --metrics-only)")
        if args.method == "hybrid_w" and args.alpha is None:
            parser.error("--method hybrid_w requires --alpha")
    return args


def emit_metrics(rankings_path: Path, questions, chunks, out_dir: Path) -> None:
    rankings = search.load_jsonl(rankings_path)
    result = metrics_mod.compute_metrics(rankings, questions, chunks)
    csv_path = out_dir / f"metrics_{rankings_path.stem.removeprefix('rankings_')}.csv"
    metrics_mod.metrics_to_csv(result, csv_path)
    print(metrics_mod.metrics_to_markdown(result))
    print(f"\n[metrics written to {csv_path}]")


def main(argv=None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    questions = search.load_jsonl(args.questions)
    chunks = search.load_jsonl(args.chunks)

    if args.metrics_only:
        emit_metrics(Path(args.rankings), questions, chunks, out_dir)
        return 0

    dense_retriever = None
    if args.method in ("dense", "hybrid_rrf", "hybrid_w"):
        # DenseRetriever lazily imports sentence-transformers in __init__.
        from diaglux.retrieval.dense import DenseRetriever

        dense_retriever = DenseRetriever(
            model_name=args.model, cache_dir=out_dir / "emb_cache"
        )

    path = search.run_and_write(
        questions, chunks,
        method=args.method, setting=args.setting, query_mode=args.query_mode,
        out_dir=out_dir, dense_retriever=dense_retriever, alpha=args.alpha,
        tokenizer=get_tokenizer(args.analyzer), rrf_k=args.rrf_k,
        analyzer=args.analyzer,
    )
    print(f"[rankings written to {path}]")

    if not args.no_metrics:
        emit_metrics(path, questions, chunks, out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
