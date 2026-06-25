#!/usr/bin/env python
"""Run one answering configuration (Phase 2 harness).

Examples (the four control runs):

  # Random baseline (no LLM at all)
  python scripts/run_answering.py --system random --seed 13

  # Closed-book control
  python scripts/run_answering.py --system closed_book --provider mock --model mock-1

  # Full-text oracle
  python scripts/run_answering.py --system oracle --provider mock --model mock-1

  # RAG (retrieval method/setting are read from inside the rankings file)
  python scripts/run_answering.py --system rag \
      --rankings outputs/retrieval/rankings_text_restricted_bm25_overlap.jsonl \
      --chunks outputs/processed/corpus_chunks_overlap.jsonl \
      --k 5 --provider openai --model gpt-4o-mini

Real providers need the API key in the environment (OPENAI_API_KEY /
ANTHROPIC_API_KEY, overridable with --api-key-env). Use --limit for smoke
tests. Re-running the same configuration resumes: already-answered
question_ids in the existing preds file are skipped.
"""

from __future__ import annotations

import argparse
import json
import sys


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an answering configuration over questions.jsonl",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Examples", 1)[1] if __doc__ else None,
    )
    parser.add_argument(
        "--system",
        required=True,
        choices=["random", "closed_book", "oracle", "rag"],
        help="Answering system. 'rag' resolves to the retrieval method "
        "named inside the rankings file (e.g. bm25).",
    )
    parser.add_argument("--rankings", default=None,
                        help="rankings_*.jsonl file (required for --system rag)")
    parser.add_argument("--chunks", default=None,
                        help="corpus_chunks_*.jsonl file (required for --system rag)")
    parser.add_argument("--k", type=int, default=5,
                        help="Top-k chunks for rag context (default: 5)")
    parser.add_argument("--model", default=None,
                        help="Model id string (default: 'none' for random, "
                        "'mock-model' for mock provider)")
    parser.add_argument("--provider", choices=["openai", "anthropic", "mock"],
                        default="mock", help="LLM provider (default: mock)")
    parser.add_argument("--base-url", default=None,
                        help="OpenAI-compatible endpoint base URL")
    parser.add_argument("--api-key-env", default=None,
                        help="Env var holding the API key (default: "
                        "OPENAI_API_KEY / ANTHROPIC_API_KEY)")
    parser.add_argument("--questions", default="outputs/processed/questions.jsonl",
                        help="questions.jsonl path")
    parser.add_argument("--limit", type=int, default=None,
                        help="Only answer the first N questions (smoke test)")
    parser.add_argument("--seed", type=int, default=13,
                        help="Seed (drives the random baseline; recorded "
                        "in the config sidecar)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--out-dir", default="outputs/runs",
                        help="Output directory (default: outputs/runs)")
    parser.add_argument("--texts-dir", default=None,
                        help="Override the clean-text directory for oracle runs")
    parser.add_argument("--mock-output", default="A",
                        help="Canned output of the mock provider (default: A)")
    return parser


def main(argv=None) -> int:
    args = build_arg_parser().parse_args(argv)

    # Load API keys from a repo-root .env if present (no-op if python-dotenv
    # is absent or the file does not exist). Never overrides already-set vars.
    try:
        from pathlib import Path

        from dotenv import load_dotenv

        load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)
    except ImportError:
        pass

    from diaglux.answering.clients import make_client
    from diaglux.answering.runner import RunConfig, run

    if args.system == "rag" and (args.rankings is None or args.chunks is None):
        print("error: --system rag requires --rankings and --chunks",
              file=sys.stderr)
        return 2

    if args.model is None:
        args.model = "none" if args.system == "random" else (
            "mock-model" if args.provider == "mock" else None)
    if args.model is None:
        print("error: --model is required for non-mock providers",
              file=sys.stderr)
        return 2

    client = None
    if args.system != "random":
        client = make_client(
            provider=args.provider,
            model=args.model,
            base_url=args.base_url,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            api_key_env=args.api_key_env,
            mock_output=args.mock_output,
        )

    is_rag = args.system == "rag"
    config = RunConfig(
        system=args.system,
        provider=args.provider if args.system != "random" else "none",
        model=args.model,
        setting="none",  # rag: resolved from the rankings file by the runner
        k=args.k if is_rag else None,
        questions_path=args.questions,
        rankings_path=args.rankings if is_rag else None,
        chunks_path=args.chunks if is_rag else None,
        texts_dir=args.texts_dir,
        seed=args.seed,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        base_url=args.base_url,
    )

    summary = run(config, client=client, out_dir=args.out_dir,
                  limit=args.limit, progress=True)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if summary["n_unparseable_new"]:
        print(f"warning: {summary['n_unparseable_new']} unparseable outputs "
              "(parse_status='unparseable', scored incorrect)", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
