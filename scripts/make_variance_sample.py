#!/usr/bin/env python
"""Build a seeded, stratified subset of questions for the decoding variance check.

The two non-deterministic answering models (gpt-5.5, forced to its default
temperature; claude-opus-4-8, temperature deprecated) are run several times on
this fixed subset to estimate run-to-run accuracy variance, so cross-model gaps
can be compared against the noise floor (see docs/methods_decoding.md).

Stratified by text_id (default 4 questions/text x 16 texts = 64) with a fixed
seed, so every repeat scores the *same* items and the sample is reproducible.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="outputs/processed/questions.jsonl")
    ap.add_argument("--out", default="outputs/processed/questions_variance_sample.jsonl")
    ap.add_argument("--per-text", type=int, default=4)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args(argv)

    by_text: dict[str, list[dict]] = defaultdict(list)
    for line in Path(args.questions).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rec = json.loads(line)
            by_text[rec["text_id"]].append(rec)

    rng = random.Random(args.seed)
    picked: list[dict] = []
    for text_id in sorted(by_text, key=lambda t: int(t.removeprefix("text"))):
        rows = by_text[text_id]
        k = min(args.per_text, len(rows))
        picked.extend(rng.sample(rows, k))

    picked.sort(key=lambda r: (int(r["text_id"].removeprefix("text")), r["question_id"]))
    out = Path(args.out)
    out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in picked),
        encoding="utf-8",
    )
    print(f"wrote {len(picked)} questions ({args.per_text}/text x {len(by_text)} texts) "
          f"to {out} [seed={args.seed}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
