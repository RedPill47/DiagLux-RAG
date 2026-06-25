#!/usr/bin/env python
"""Quantify the query<->evidence orthography gap in LuxDiagRC.

Motivation (review_and_plan §2.9, retrieval_findings_dense.md): character-n-gram
BM25 beats word-BM25 and off-the-shelf dense retrieval for Luxembourgish. This
script measures *why*: how much question<->evidence overlap is invisible to
word-level matching but visible at the subword level, because questions use
informal/nonstandard spelling while the texts use literary orthography.

For every question with a resolved ``critical_span`` we compare the question text
to its gold evidence span and report:

1. Mean word-token Jaccard vs. mean character-3-gram Jaccard (question vs. span).
   If char3 >> word, subword matching sees overlap that word matching misses.
2. Content-word matchability: each question content word is classified against the
   span's word set as exact / subword-only (a spelling variant: char similarity
   >= 0.70 to some span word but not identical) / none. The "subword-only" mass is
   the headroom char-n-gram BM25 can exploit and word-BM25 cannot.

Pure stdlib + the project's BM25 tokenizer; no torch, no network.
"""
from __future__ import annotations

import argparse
import json
import sys
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diaglux.data.texts import load_clean_text  # noqa: E402
from diaglux.retrieval.tokenize import word_tokenize  # noqa: E402

RESOLVED = {"exact", "dehyphen", "fuzzy", "multiple"}
SIM_THRESHOLD = 0.70
MIN_CONTENT_LEN = 3  # ignore very short function words


def char_ngrams(text: str, n: int = 3) -> set[str]:
    s = "".join(word_tokenize(text))  # tokenizer lowercases & strips punctuation
    return {s[i : i + n] for i in range(len(s) - n + 1)} if len(s) >= n else {s}


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def classify_word(w: str, span_words: set[str]) -> str:
    if w in span_words:
        return "exact"
    best = max((SequenceMatcher(None, w, sw).ratio() for sw in span_words), default=0.0)
    return "subword" if best >= SIM_THRESHOLD else "none"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--questions", default="outputs/processed/questions.jsonl")
    ap.add_argument("--data-root", default=None,
                    help="dataset root containing Texts/ (default: auto-discover)")
    ap.add_argument("--out", default="outputs/analysis/orthography_gap.md")
    args = ap.parse_args(argv)

    bodies: dict[str, str] = {}

    def body(text_id: str) -> str:
        if text_id not in bodies:
            bodies[text_id] = load_clean_text(text_id, data_root=args.data_root)[2]
        return bodies[text_id]

    word_jac, char_jac = [], []
    n_exact = n_subword = n_none = 0
    n_q = 0
    for line in Path(args.questions).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        q = json.loads(line)
        cs = q["critical_span"]
        if cs["status"] not in RESOLVED or cs["start"] is None:
            continue
        span_text = body(q["text_id"])[cs["start"]: cs["end"]]
        span_text = unicodedata.normalize("NFC", span_text)
        qwords = set(word_tokenize(q["question"]))
        swords = set(word_tokenize(span_text))
        if not qwords or not swords:
            continue
        n_q += 1
        word_jac.append(jaccard(qwords, swords))
        char_jac.append(jaccard(char_ngrams(q["question"]), char_ngrams(span_text)))
        for w in qwords:
            if len(w) < MIN_CONTENT_LEN:
                continue
            c = classify_word(w, swords)
            n_exact += c == "exact"
            n_subword += c == "subword"
            n_none += c == "none"

    tot = n_exact + n_subword + n_none or 1
    matchable = n_exact + n_subword or 1
    sub_of_matchable = n_subword / matchable
    mw, mc = sum(word_jac) / len(word_jac), sum(char_jac) / len(char_jac)
    md = f"""# Orthography gap: question vs. gold evidence span

Questions analyzed (resolved critical span): **{n_q}**. Span text = clean-text slice
at the aligned critical-span offsets. Tokenizer = the BM25 word tokenizer.

## Headline: among question words that match the evidence, {sub_of_matchable:.0%} are spelling variants

Each question content word (len >= {MIN_CONTENT_LEN}) is matched against the gold evidence
span's word set: **exact** (verbatim), **subword-only** (not identical but char-similarity
>= {SIM_THRESHOLD} to some span word — a spelling variant), or **none**.

| match type | count | share of all |
|---|---|---|
| exact (verbatim in span) | {n_exact} | {n_exact / tot:.1%} |
| subword-only (spelling variant) | {n_subword} | {n_subword / tot:.1%} |
| none (paraphrase / absent) | {n_none} | {n_none / tot:.1%} |

Of the question words that have **any** lexical correspondent in the evidence
(exact + subword = {matchable}), **{sub_of_matchable:.0%} are recoverable only at the subword
level** — invisible to word-BM25, recovered by char-n-gram BM25. This is the measured
mechanism behind the BM25-char > BM25-word ordering: informal/nonstandard question spelling
vs. literary text orthography, plus Luxembourgish compounding and diacritic variation.

The large **none** share ({n_none / tot:.1%}) is genuine paraphrase — question vocabulary with
no lexical form in the evidence at all. This is the signal semantic (dense) retrieval is meant
to capture; that dense nonetheless *underperforms* lexical BM25 here (retrieval_findings_dense.md)
shows off-the-shelf multilingual embedders do not bridge it for Luxembourgish.

## Aggregate set overlap (for completeness)

| overlap metric (question vs. span) | mean Jaccard |
|---|---|
| word-token | {mw:.3f} |
| character-3-gram | {mc:.3f} |

Aggregate Jaccard is only modestly higher at the character level ({mc / mw:.2f}x) and is
diluted by span length; char-BM25's advantage comes not from raw set overlap but from BM25's
IDF weighting of *discriminative* subword n-grams in the {sub_of_matchable:.0%} of matchable
words that word-level tokenization cannot align.
"""
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md, encoding="utf-8")
    print(md)
    print(f"[written to {out}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
