"""Shared synthetic fixtures for the analysis tests.

Builds schema-exact preds files (+ .config.json sidecars) from the synthetic
questions in answering_testutils. Everything is written under tmp_path.
"""

from __future__ import annotations

import json
from pathlib import Path

from answering_testutils import make_questions, write_questions  # noqa: F401

TIMESTAMP = "2026-06-12T14:00:00Z"


def make_pred(
    question: dict,
    letter: str | None,
    system: str = "oracle",
    setting: str = "none",
    k: int | None = None,
    model: str = "mock-model",
    raw_output: str | None = None,
) -> dict:
    """One contract-schema preds row; semantic_choice/is_correct derived from
    the question's permutation. ``letter=None`` produces an unparseable row."""
    if letter is None:
        semantic = None
        is_correct = False
        parse_status = "unparseable"
        raw = raw_output if raw_output is not None else "Ech weess et net."
    else:
        semantic = question["permutation"][ord(letter) - 65]
        is_correct = semantic == "correct"
        parse_status = "exact"
        raw = raw_output if raw_output is not None else letter
    return {
        "question_id": question["question_id"],
        "system": system,
        "setting": setting,
        "k": k,
        "model": model,
        "context_chunk_ids": [],
        "raw_output": raw,
        "parsed_letter": letter,
        "parse_status": parse_status,
        "semantic_choice": semantic,
        "is_correct": is_correct,
        "timestamp": TIMESTAMP,
    }


def correct_letter(question: dict) -> str:
    return question["gold_letter"]


def wrong_letter(question: dict, semantic: str = "misunderstand") -> str:
    return chr(65 + question["permutation"].index(semantic))


def write_preds(
    runs_dir: Path,
    config_id: str,
    rows: list[dict],
    config: dict | None = None,
    write_sidecar: bool = True,
) -> Path:
    """Write preds_{config_id}.jsonl plus its .config.json sidecar."""
    runs_dir.mkdir(parents=True, exist_ok=True)
    preds_path = runs_dir / f"preds_{config_id}.jsonl"
    with open(preds_path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if write_sidecar:
        sidecar = {"config_id": config_id, "config": config or {"system": rows[0]["system"]}}
        sidecar_path = runs_dir / f"preds_{config_id}.config.json"
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
    return preds_path


def write_jsonl(path: Path, records: list[dict]) -> Path:
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
