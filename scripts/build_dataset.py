"""Phase 1 build script: produce the processed dataset artifacts.

Writes (docs/CONTRACTS.md):
  outputs/processed/questions.jsonl
  outputs/processed/corpus_chunks_{paragraph,overlap,sentence}.jsonl
  outputs/processed/alignment_report.md

Usage: python scripts/build_dataset.py [--data-root PATH] [--out-dir PATH] [--seed 13]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from tqdm import tqdm

from diaglux.data.align import locate_span
from diaglux.data.chunking import (
    check_full_coverage,
    chunk_overlap,
    chunk_paragraph,
    chunk_sentence,
)
from diaglux.data.kb import load_questions
from diaglux.data.shuffle import GLOBAL_SEED, shuffle_options, letter_to_semantic
from diaglux.data.tags import extract_tags, strip_tags, tag_categories
from diaglux.data.texts import TEXT_IDS, find_data_root, load_clean_text

STATUS_ORDER = ("exact", "dehyphen", "fuzzy", "multiple", "unresolved", "empty")
CHUNKERS = {
    "paragraph": chunk_paragraph,
    "overlap": chunk_overlap,
    "sentence": chunk_sentence,
}


def span_payload(alignment) -> dict:
    payload = {
        "start": alignment.start,
        "end": alignment.end,
        "status": alignment.status,
    }
    if alignment.in_title:
        payload["in_title"] = True
    if alignment.ratio is not None:
        payload["ratio"] = alignment.ratio
    if alignment.n_matches is not None:
        payload["n_matches"] = alignment.n_matches
    if alignment.partial:
        payload["partial"] = True
    return payload


def write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-root", default=None, help="dir with Texts/ etc.")
    ap.add_argument("--out-dir", default="outputs/processed")
    ap.add_argument("--seed", type=int, default=GLOBAL_SEED, help="global shuffle seed")
    args = ap.parse_args(argv)

    data_root = Path(args.data_root) if args.data_root else find_data_root()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- texts -----------------------------------------------------------
    texts: dict[str, tuple[str, str, str]] = {
        tid: load_clean_text(tid, data_root) for tid in TEXT_IDS
    }

    # ---- questions -------------------------------------------------------
    records = load_questions(data_root)
    title_match = {tid: True for tid in TEXT_IDS}  # load_questions enforces this

    rows = []
    status_counts = {"critical": Counter(), "distractor": Counter()}
    unresolved: list[tuple[str, str, str]] = []  # (question_id, which, preview)
    per_text = Counter()
    per_cognitive = Counter()
    fuzzy_resolved = {"critical": 0, "distractor": 0}
    partial_counts = {"critical": 0, "distractor": 0}

    for rec in tqdm(records, desc="aligning questions", unit="q"):
        title, author, body = texts[rec.text_id]
        crit = locate_span(rec.critical_span_raw, body, title=title, author=author)
        dist = locate_span(rec.distractor_span_raw, body, title=title, author=author)
        status_counts["critical"][crit.status] += 1
        status_counts["distractor"][dist.status] += 1
        if crit.status == "fuzzy":
            fuzzy_resolved["critical"] += 1
        if dist.status == "fuzzy":
            fuzzy_resolved["distractor"] += 1
        if crit.partial:
            partial_counts["critical"] += 1
        if dist.partial:
            partial_counts["distractor"] += 1
        for which, al, raw in (
            ("criticalSpan", crit, rec.critical_span_raw),
            ("distractorSpan", dist, rec.distractor_span_raw),
        ):
            if al.status == "unresolved":
                preview = " ".join(strip_tags(raw).split())[:80]
                unresolved.append((rec.question_id, which, preview))

        presented, permutation, gold_letter = shuffle_options(
            rec.question_id, rec.options, global_seed=args.seed
        )
        # Round-trip guard: letter -> semantic -> stored option text.
        for i, letter in enumerate("ABCD"):
            sem = letter_to_semantic(letter, permutation)
            assert presented[letter] == rec.options[sem], rec.question_id
        assert letter_to_semantic(gold_letter, permutation) == "correct"

        tags = extract_tags(rec.critical_span_raw)
        rows.append(
            {
                "question_id": rec.question_id,
                "text_id": rec.text_id,
                "text_title": rec.text_title,
                "question": rec.question,
                "cognitive_type": rec.cognitive_type,
                "options": rec.options,
                "presented": presented,
                "permutation": permutation,
                "gold_letter": gold_letter,
                "shuffle_seed": args.seed,
                "critical_span": span_payload(crit),
                "distractor_span": span_payload(dist),
                "linguistic_tags": tags,
                "linguistic_categories": tag_categories(tags),
            }
        )
        per_text[rec.text_id] += 1
        per_cognitive[rec.cognitive_type] += 1

    write_jsonl(out_dir / "questions.jsonl", rows)

    # ---- chunks ----------------------------------------------------------
    chunk_counts: dict[str, dict[str, int]] = defaultdict(dict)
    coverage_notes: list[str] = []
    for strategy, chunker in CHUNKERS.items():
        all_chunks = []
        for tid in TEXT_IDS:
            _, _, body = texts[tid]
            chunks = chunker(tid, body)
            for c in chunks:
                assert c.chunk_text == body[c.start_char : c.end_char], c.chunk_id
            chunk_counts[strategy][tid] = len(chunks)
            if strategy == "overlap":
                check_full_coverage(chunks, len(body))  # also asserted inside chunker
                coverage_notes.append(
                    f"- `{tid}` overlap: {len(chunks)} chunks, union covers "
                    f"[0, {len(body)}) — OK"
                )
            all_chunks.extend(chunks)
        write_jsonl(
            out_dir / f"corpus_chunks_{strategy}.jsonl",
            [
                {
                    "chunk_id": c.chunk_id,
                    "text_id": c.text_id,
                    "chunk_text": c.chunk_text,
                    "start_char": c.start_char,
                    "end_char": c.end_char,
                    "n_tokens": c.n_tokens,
                }
                for c in all_chunks
            ],
        )

    # ---- report ----------------------------------------------------------
    lines = [
        "# Alignment and Validation Report",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat(timespec='seconds')}  ",
        f"Data root: `{data_root}`  ",
        f"Questions: {len(rows)}  |  Global shuffle seed: {args.seed}",
        "",
        "## Per-text title match",
        "",
        "| text_id | title | KB title match | questions |",
        "|---|---|---|---|",
    ]
    for tid in TEXT_IDS:
        lines.append(
            f"| {tid} | {texts[tid][0]} | {'OK' if title_match[tid] else 'MISMATCH'} "
            f"| {per_text[tid]} |"
        )
    lines += ["", "## Span alignment status counts", ""]
    lines.append("| status | criticalSpan | distractorSpan |")
    lines.append("|---|---|---|")
    for status in STATUS_ORDER:
        lines.append(
            f"| {status} | {status_counts['critical'].get(status, 0)} "
            f"| {status_counts['distractor'].get(status, 0)} |"
        )
    n_q = len(rows)
    crit_located = n_q - status_counts["critical"].get("unresolved", 0) - status_counts[
        "critical"
    ].get("empty", 0)
    dist_located = n_q - status_counts["distractor"].get("unresolved", 0) - status_counts[
        "distractor"
    ].get("empty", 0)
    lines += [
        "",
        f"Located (exact + dehyphen + fuzzy + multiple + in-title): "
        f"critical {crit_located}/{n_q}, distractor {dist_located}/{n_q}.",
        "",
        f"Of the fuzzy spans, {partial_counts['critical']} critical and "
        f"{partial_counts['distractor']} distractor spans carry `\"partial\": true`: "
        "the annotated span concatenates non-contiguous passages of the text, and the "
        "recorded offsets cover only the longest sentence piece that aligned uniquely. "
        "`partial` is an additive extension of the span schema (consumers ignore "
        "unknown keys).",
        "",
        "## Question counts per cognitive type",
        "",
        "| cognitive type | questions |",
        "|---|---|",
    ]
    for ct in ("Retrieve", "Interpret", "Inferential", "Evaluative"):
        lines.append(f"| {ct} | {per_cognitive.get(ct, 0)} |")
    lines += ["", "## Unresolved spans", ""]
    if unresolved:
        lines.append("| question_id | span | first 80 chars (tags stripped) |")
        lines.append("|---|---|---|")
        for qid, which, preview in unresolved:
            lines.append(f"| {qid} | {which} | {preview} |")
    else:
        lines.append("None — every span was located (or is empty/in-title).")
    lines += ["", "## Chunking", ""]
    lines.append("| strategy | total chunks | per text |")
    lines.append("|---|---|---|")
    for strategy in CHUNKERS:
        per = chunk_counts[strategy]
        total = sum(per.values())
        per_str = ", ".join(f"{tid}:{n}" for tid, n in per.items())
        lines.append(f"| {strategy} | {total} | {per_str} |")
    lines += [
        "",
        "### Overlap coverage check (union of spans must equal each full body)",
        "",
        *coverage_notes,
        "",
        "Note: the clean texts contain no blank lines (hard-wrapped single-"
        "newline layout), so the `paragraph` strategy groups consecutive "
        "lines into natural units flushed at sentence-final line ends "
        "(>= 60 tokens, hard cap 180).",
        "",
    ]
    (out_dir / "alignment_report.md").write_text(
        "\n".join(lines), encoding="utf-8", newline="\n"
    )

    print(f"Wrote {n_q} questions, chunk files for {list(CHUNKERS)}, and "
          f"alignment_report.md to {out_dir}")
    print("critical:", dict(status_counts["critical"]))
    print("distractor:", dict(status_counts["distractor"]))
    print(f"unresolved spans: {len(unresolved)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
