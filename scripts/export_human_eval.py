#!/usr/bin/env python
"""Export, per question, the retrieved chunks each model saw and the answer it gave.

For human evaluation (answer correctness and evidence sufficiency). For a chosen
RAG configuration (retriever, setting, k) and a chosen subset of texts, writes a
readable Markdown file plus a machine-readable JSONL. The retrieved chunks are the
same across the four answering models for a given configuration (retrieval does not
depend on the LLM); what differs per model is the selected answer.

Usage:
  python scripts/export_human_eval.py                       # default: open-corpus, k=5, hybrid char, texts 1-3
  python scripts/export_human_eval.py --texts text5,text13  # pick texts
  python scripts/export_human_eval.py --setting text_restricted --k 10
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from diaglux.data.texts import load_clean_text  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs" / "runs"
PROC = ROOT / "outputs" / "processed"
MODELS = [("claude-opus-4-8", "Opus 4.8"), ("gpt-5.5", "GPT-5.5"),
          ("claude-sonnet-4-6", "Sonnet 4.6"), ("deepseek-v4-pro", "deepseek-v4-pro")]
RETR_LABEL = {"hybrid_w0.5_char_ngram": "hybrid char-BM25+dense (weighted)",
              "bm25_char_ngram": "BM25 char", "bm25": "BM25 word",
              "dense_BAAI_bge-m3": "dense BGE-M3"}


def find_config(model, method, setting, k):
    for cfg in glob.glob(str(RUNS / "preds_*.config.json")):
        d = json.load(open(cfg, encoding="utf-8"))
        c, cl = d["config"], d.get("client", {})
        if (cl.get("model") == model and c["system"] == method
                and c["setting"] == setting and c["k"] == k):
            return cfg.replace(".config.json", ".jsonl")
    return None


def overlaps(chunk, span, q_text_id):
    return (span["start"] is not None and chunk["text_id"] == q_text_id
            and chunk["start_char"] < span["end"] and chunk["end_char"] > span["start"])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--setting", default="open_corpus", choices=["open_corpus", "text_restricted"])
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--method", default="hybrid_w0.5_char_ngram")
    ap.add_argument("--texts", default="text1,text2,text3", help="comma list, or 'all'")
    ap.add_argument("--chunks", default=str(PROC / "corpus_chunks_overlap.jsonl"))
    ap.add_argument("--out-dir", default=str(ROOT / "outputs" / "human_eval"))
    args = ap.parse_args(argv)

    questions = {q["question_id"]: q for q in
                 (json.loads(l) for l in open(PROC / "questions.jsonl", encoding="utf-8"))}
    chunks = {c["chunk_id"]: c for c in (json.loads(l) for l in open(args.chunks, encoding="utf-8"))}
    bodies = {}
    def span_text(text_id, sp):
        if sp["start"] is None:
            return ""
        if text_id not in bodies:
            bodies[text_id] = load_clean_text(text_id)[2]
        return bodies[text_id][sp["start"]:sp["end"]]

    # per-model preds for this config
    preds = {}
    for model, _ in MODELS:
        f = find_config(model, args.method, args.setting, args.k)
        if f:
            preds[model] = {r["question_id"]: r for r in (json.loads(l) for l in open(f, encoding="utf-8"))}
    if not preds:
        sys.exit(f"No runs found for {args.method} / {args.setting} / k={args.k}")

    want = None if args.texts == "all" else set(args.texts.split(","))
    qids = [qid for qid in questions if (want is None or questions[qid]["text_id"] in want)]
    qids.sort(key=lambda q: (int(questions[q]["text_id"].removeprefix("text")), q))

    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)
    tag = f"{args.setting}_k{args.k}_{args.method}"
    md, rows = [], []
    md.append(f"# Human-eval export: retrieved context and model answers\n")
    md.append(f"Configuration: retriever = {RETR_LABEL.get(args.method, args.method)}, "
              f"setting = {args.setting}, k = {args.k}. The retrieved chunks below are the same "
              f"for all four models (retrieval is model-independent); only the answer differs.\n")
    md.append(f"Texts: {args.texts}. Questions: {len(qids)}.\n")

    any_model = next(iter(preds))
    for qid in qids:
        q = questions[qid]
        rec = preds[any_model].get(qid)
        if rec is None:
            continue
        cids = rec["context_chunk_ids"]
        crit, dist = q["critical_span"], q["distractor_span"]
        crit_ret = any(overlaps(chunks[c], crit, q["text_id"]) for c in cids if c in chunks)
        dist_ret = any(overlaps(chunks[c], dist, q["text_id"]) for c in cids if c in chunks)
        perm = q["permutation"]; gold = q["gold_letter"]
        md.append(f"## {qid}  |  {q['text_title']}  |  cognitive: {q['cognitive_type']}")
        md.append(f"**Question:** {q['question']}")
        md.append("**Options (as presented):**")
        for i, L in enumerate("ABCD"):
            sem = perm[i]; star = "  **(gold)**" if L == gold else ""
            md.append(f"- {L}. {q['presented'][L]}  _[{sem}]_{star}")
        md.append(f"**Critical span:** \"{span_text(q['text_id'], crit)}\"  (retrieved: {'YES' if crit_ret else 'no'})")
        md.append(f"**Distractor span:** \"{span_text(q['text_id'], dist)}\"  (retrieved: {'YES' if dist_ret else 'no'})")
        md.append(f"**Retrieved chunks (top {args.k}):**")
        for rank, c in enumerate(cids, 1):
            ch = chunks.get(c)
            txt = ch["chunk_text"].replace("\n", " ").strip() if ch else "(missing)"
            md.append(f"{rank}. `[{c}]` {txt}")
        md.append("**Model answers:**")
        per_model = {}
        for model, label in MODELS:
            r = preds.get(model, {}).get(qid)
            if r:
                ok = "correct" if r["is_correct"] else "WRONG"
                md.append(f"- {label}: {r['parsed_letter']} _({r['semantic_choice']})_ -> {ok}")
                per_model[model] = {"letter": r["parsed_letter"], "semantic": r["semantic_choice"],
                                    "correct": r["is_correct"]}
        md.append("\n---\n")
        rows.append({"question_id": qid, "text_id": q["text_id"], "text_title": q["text_title"],
                     "cognitive_type": q["cognitive_type"], "question": q["question"],
                     "options": {L: {"text": q["presented"][L], "type": perm[i]} for i, L in enumerate("ABCD")},
                     "gold_letter": gold, "critical_span": span_text(q["text_id"], crit),
                     "distractor_span": span_text(q["text_id"], dist),
                     "critical_retrieved": crit_ret, "distractor_retrieved": dist_ret,
                     "retrieved_chunks": [{"rank": i + 1, "chunk_id": c,
                                           "text": chunks[c]["chunk_text"]} for i, c in enumerate(cids) if c in chunks],
                     "model_answers": per_model})

    (out / f"human_eval_{tag}.md").write_text("\n".join(md), encoding="utf-8")
    with open(out / f"human_eval_{tag}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"wrote {out}/human_eval_{tag}.md and .jsonl  ({len(rows)} questions, {len(preds)} models)")


if __name__ == "__main__":
    raise SystemExit(main())
