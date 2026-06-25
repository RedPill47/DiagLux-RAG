#!/usr/bin/env python
"""Driver for the focused RAG answering grid (Phase 3 answering).

Rather than the full cross-product (every retrieval method x every k x both
settings x every model = tens of thousands of calls), this runs a *focused*
grid that still populates the paper's required tables (concept doc 6.1/6.2) and
the retrieval-method comparison, while bounding cost:

  PRIMARY (best retriever, k-sweep):
    hybrid_w0.5_char  x  k in {1,3,5,10}  x  {text_restricted, open_corpus}
  COMPARISON (fixed k=5):
    {bm25, bm25_char, dense_bge-m3, hybrid_rrf_char}  x  both settings

All use overlap chunks + question_options queries (the configuration the
retrieval study selected; see docs/retrieval_findings_dense.md). The "reduced"
tier trims the k-sweep and comparison for the reasoning model (deepseek-v4-pro),
which is far slower/costlier per call; "full" runs everything (cheap for the
Claude models).

Per-question logging, resumability, and API-error handling come from the runner.
Models and their API quirks (token budget, base_url) are encoded in MODELS so a
run is just:  python scripts/run_rag_grid.py --model sonnet
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

ROOT = Path(__file__).resolve().parents[1]
RANK = ROOT / "outputs" / "retrieval"

# Model registry: provider quirks discovered during control runs.
# max_tokens is the *budget* (billed on actual use); reasoning models need head-
# room for hidden reasoning tokens. temperature is requested 0; the clients drop
# or adapt it where a model rejects 0 (gpt-5.5) or deprecates it (opus).
MODELS = {
    "sonnet":   dict(provider="anthropic", model="claude-sonnet-4-6", max_tokens=2048, tier="full"),
    "opus":     dict(provider="anthropic", model="claude-opus-4-8",  max_tokens=2048, tier="full"),
    "deepseek": dict(provider="openai",    model="deepseek-v4-pro",  max_tokens=8192, tier="reduced",
                     base_url="https://api.deepseek.com", api_key_env="DEEPSEEK_API_KEY"),
    "gpt5.5":   dict(provider="openai",    model="gpt-5.5",          max_tokens=8192, tier="full"),
}

SETTINGS = ("text_restricted", "open_corpus")

# (method -> rankings filename stem); all overlap + question_options.
RANKINGS = {
    "hybrid_w0.5_char": "hybrid_w0.5_char_ngram",
    "hybrid_rrf_char":  "hybrid_rrf_char_ngram",
    "bm25":             "bm25",
    "bm25_char":        "bm25_char_ngram",
    "dense_bge-m3":     "dense_BAAI_bge-m3",
}

PRIMARY_METHOD = "hybrid_w0.5_char"
COMPARISON_METHODS = ["bm25", "bm25_char", "dense_bge-m3", "hybrid_rrf_char"]


def rankings_path(method: str, setting: str) -> Path:
    return RANK / f"rankings_{setting}_{RANKINGS[method]}_overlap_question_options.jsonl"


def build_configs(tier: str):
    """Yield (method, setting, k) for the chosen tier.

    full     : best retriever k in {1,3,5,10} + comparison@k=5 both settings (16)
    reduced  : best retriever k in {1,5,10}   + comparison@k=5 open_corpus (10)
    minimal  : best retriever k in {1,5,10} x both settings only — the headline
               k-sweep, no method comparison (6); cheapest informative slice.
    """
    ks = [1, 3, 5, 10] if tier == "full" else [1, 5, 10]
    for setting in SETTINGS:
        for k in ks:
            yield PRIMARY_METHOD, setting, k
    if tier == "minimal":
        return
    comp_settings = SETTINGS if tier == "full" else ("open_corpus",)
    for setting in comp_settings:
        for method in COMPARISON_METHODS:
            yield method, setting, 5


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--tier", choices=["full", "reduced", "minimal"], default=None,
                    help="override the model's default tier")
    ap.add_argument("--chunks", default="outputs/processed/corpus_chunks_overlap.jsonl")
    ap.add_argument("--out-dir", default="outputs/runs")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap questions per config (for a cheap smoke; resumes to full later)")
    ap.add_argument("--dry-run", action="store_true", help="list configs, do not call any API")
    args = ap.parse_args(argv)

    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env", override=False)
    except ImportError:
        pass

    spec = MODELS[args.model]
    tier = args.tier or spec["tier"]
    configs = list(build_configs(tier))
    print(f"[{args.model}] {spec['model']} | tier={tier} | {len(configs)} configs")
    for method, setting, k in configs:
        rp = rankings_path(method, setting)
        status = "OK" if rp.exists() else "MISSING"
        print(f"  {setting:15} {method:18} k={k:<2} -> {rp.name} [{status}]")
    if args.dry_run:
        return 0

    from diaglux.answering.clients import make_client
    from diaglux.answering.runner import RunConfig, run

    client = make_client(
        provider=spec["provider"], model=spec["model"],
        base_url=spec.get("base_url"), api_key_env=spec.get("api_key_env"),
        temperature=0.0, max_tokens=spec["max_tokens"],
    )

    for method, setting, k in configs:
        rp = rankings_path(method, setting)
        if not rp.exists():
            print(f"  SKIP (missing rankings): {rp.name}", file=sys.stderr)
            continue
        cfg = RunConfig(
            system="rag", provider=spec["provider"], model=spec["model"],
            setting=setting, k=k, rankings_path=str(rp), chunks_path=args.chunks,
            temperature=0.0, max_tokens=spec["max_tokens"], base_url=spec.get("base_url"),
        )
        summary = run(cfg, client=client, out_dir=Path(args.out_dir),
                      limit=args.limit, progress=True)
        print(f"  [{method}/{setting}/k={k}] acc={summary['accuracy_new']} "
              f"new={summary['n_new']} err={summary['n_error_new']} "
              f"term={summary['terminated_early']}")
        if summary["terminated_early"]:
            print("  grid halted by an unrecoverable API error; rerun to resume.",
                  file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
