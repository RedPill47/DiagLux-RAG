"""Run one answering configuration over questions.jsonl.

Writes ``outputs/runs/preds_{config_id}.jsonl`` (one line per question, schema
in docs/CONTRACTS.md) plus a ``preds_{config_id}.config.json`` sidecar holding
the full configuration, the prompt template hash, and the seed.

``config_id`` = first 10 hex chars of sha256 over the canonical (sorted-keys,
compact) JSON of the configuration. Runs are resumable: question_ids already
present in an existing preds file are skipped, so an interrupted run can be
re-launched with the identical configuration and will only fill the gaps.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from diaglux.answering.clients import LLMClient
from diaglux.answering.context import (
    build_oracle_context,
    build_retrieval_context,
    load_chunks,
    load_rankings,
)
from diaglux.answering.parsing import parse_letter
from diaglux.answering.prompts import build_prompt, prompt_template_hash

LETTERS = ("A", "B", "C", "D")

# Systems that use no retrieval and no retrieval files.
NON_RETRIEVAL_SYSTEMS = {"random", "closed_book", "oracle"}

# Substrings marking an UNRECOVERABLE API error: retrying or continuing is
# futile, so abort the whole run cleanly (progress is already flushed and
# resumable). Anything else is treated as transient and retried.
_TERMINAL_MARKERS = (
    "insufficient_quota", "exceeded your current quota", "billing",
    "invalid_api_key", "invalid api key", "authentication", "permission denied",
)


class TerminalAPIError(RuntimeError):
    """Unrecoverable API error (quota/auth): stop the run rather than churn."""


def _complete_resilient(
    client: LLMClient, prompt: str, retries: int = 3, backoff: float = 2.0
) -> Tuple[Optional[str], Optional[Exception]]:
    """Call ``client.complete`` with retries.

    Returns ``(raw_output, None)`` on success, or ``(None, exc)`` if a
    *transient* error persists after ``retries`` attempts (the caller logs the
    question as an error and continues). Raises :class:`TerminalAPIError` on a
    quota/auth error so the caller can abort the run.
    """
    last: Optional[Exception] = None
    for attempt in range(retries):
        try:
            return client.complete(prompt), None
        except Exception as exc:  # classify by message
            if any(m in str(exc).lower() for m in _TERMINAL_MARKERS):
                raise TerminalAPIError(str(exc)) from exc
            last = exc
            if attempt < retries - 1:
                time.sleep(backoff * (attempt + 1))
    return None, last


@dataclasses.dataclass
class RunConfig:
    """Full configuration of one answering run.

    ``system`` is ``random | closed_book | oracle`` or a retrieval method
    name (e.g. ``bm25``). The CLI placeholder ``rag`` is resolved to the
    method/setting declared inside the rankings file before the config_id is
    computed, so identical resolved configs share one preds file.
    """

    system: str
    provider: str = "mock"
    model: str = "none"
    setting: str = "none"  # text_restricted | open_corpus | none
    k: Optional[int] = None  # null for non-retrieval systems
    questions_path: str = "outputs/processed/questions.jsonl"
    rankings_path: Optional[str] = None
    chunks_path: Optional[str] = None
    texts_dir: Optional[str] = None  # override for oracle clean-text loading
    seed: int = 13
    temperature: float = 0.0
    max_tokens: int = 64
    base_url: Optional[str] = None


def canonical_config(config: RunConfig) -> dict:
    """Config as a plain dict, plus the prompt template hash.

    Path-valued fields are reduced to their basename so the config_id is
    invariant to how the path was supplied (relative vs. absolute, forward vs.
    back slashes). Filenames are unique within this project, so two entry points
    (e.g. ``run_rag_grid`` and ``run_answering``) that reference the same
    rankings file now resolve to the same config_id and resume the same preds
    file rather than forking a duplicate.
    """
    data = dataclasses.asdict(config)
    for key in ("questions_path", "rankings_path", "chunks_path", "texts_dir"):
        if data.get(key) is not None:
            data[key] = Path(str(data[key]).replace("\\", "/")).name
    data["prompt_template_hash"] = prompt_template_hash()
    return data


def compute_config_id(config: RunConfig) -> str:
    """First 10 hex chars of sha256 of the canonical config JSON."""
    blob = json.dumps(
        canonical_config(config),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:10]


def random_letter(seed: int, question_id: str) -> str:
    """Deterministic uniform choice over A-D, independent of question order
    (so resumed runs pick the same letter for the same question)."""
    rng = random.Random(f"{seed}:{question_id}")
    return rng.choice(LETTERS)


def load_questions(path) -> List[dict]:
    questions: List[dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                questions.append(json.loads(line))
    return questions


def _existing_question_ids(preds_path: Path) -> Set[str]:
    done: Set[str] = set()
    if not preds_path.exists():
        return done
    with open(preds_path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                done.add(json.loads(line)["question_id"])
    return done


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _semantic_choice(question: dict, letter: Optional[str]) -> Optional[str]:
    """Map a presented letter back to its semantic option type via the
    stored permutation (permutation[i] = type shown as chr(65+i))."""
    if letter is None:
        return None
    return question["permutation"][ord(letter) - 65]


def resolve_rag_config(config: RunConfig, rankings: Dict[str, dict]) -> RunConfig:
    """Replace the 'rag' placeholder system/setting with the method/setting
    declared inside the rankings file (contract: system column holds the
    retrieval method name, e.g. 'bm25')."""
    if not rankings:
        raise ValueError("Rankings file is empty; cannot resolve rag config")
    first = next(iter(rankings.values()))
    updates = {}
    if config.system == "rag":
        updates["system"] = first["method"]
    if config.setting in (None, "none"):
        updates["setting"] = first["setting"]
    return dataclasses.replace(config, **updates) if updates else config


def run(
    config: RunConfig,
    client: Optional[LLMClient] = None,
    out_dir="outputs/runs",
    limit: Optional[int] = None,
    progress: bool = False,
) -> dict:
    """Execute (or resume) one configuration. Returns a summary dict.

    ``client`` is required for every system except ``random``.
    ``limit`` truncates the question list (smoke tests).
    """
    is_retrieval = config.system not in NON_RETRIEVAL_SYSTEMS  # includes "rag"

    rankings: Optional[Dict[str, dict]] = None
    chunks: Optional[Dict[str, str]] = None
    if is_retrieval:
        if config.rankings_path is None or config.chunks_path is None:
            raise ValueError(
                f"System {config.system!r} requires rankings_path and chunks_path"
            )
        if config.k is None:
            raise ValueError("Retrieval runs need k")
        rankings = load_rankings(config.rankings_path)
        config = resolve_rag_config(config, rankings)
        chunks = load_chunks(config.chunks_path)

    if config.system != "random" and client is None:
        raise ValueError(f"System {config.system!r} requires an LLM client")

    config_id = compute_config_id(config)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds_path = out_dir / f"preds_{config_id}.jsonl"
    sidecar_path = out_dir / f"preds_{config_id}.config.json"

    sidecar = {"config_id": config_id, "config": canonical_config(config)}
    if client is not None:
        sidecar["client"] = client.describe()
    sidecar_path.write_text(
        json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    questions = load_questions(config.questions_path)
    if limit is not None:
        questions = questions[:limit]
    done = _existing_question_ids(preds_path)

    n_new = n_skipped = n_correct = n_unparseable = n_error = 0
    terminated = False

    with open(preds_path, "a", encoding="utf-8") as handle:
        for question in questions:
            qid = question["question_id"]
            if qid in done:
                n_skipped += 1
                continue

            context: Optional[str] = None
            context_chunk_ids: List[str] = []

            if config.system == "random":
                letter = random_letter(config.seed, qid)
                raw_output = letter
                parse_status = "exact"
            else:
                if config.system == "oracle":
                    context, context_chunk_ids = build_oracle_context(
                        question["text_id"], texts_dir=config.texts_dir
                    )
                elif config.system == "closed_book":
                    context = None
                else:  # retrieval system
                    assert rankings is not None and chunks is not None
                    record = rankings.get(qid)
                    if record is None:
                        raise KeyError(
                            f"question_id {qid} missing from rankings file "
                            f"{config.rankings_path}"
                        )
                    context, context_chunk_ids = build_retrieval_context(
                        record, chunks, config.k
                    )
                prompt = build_prompt(question, context)
                try:
                    raw_output, api_err = _complete_resilient(client, prompt)
                except TerminalAPIError as exc:
                    print(
                        f"\nABORTING after {n_new} new answers — unrecoverable API "
                        f"error (quota/auth): {exc}\nProgress is saved; top up / fix "
                        "credentials and re-run the identical command to resume.",
                        file=sys.stderr,
                    )
                    terminated = True
                    break
                if api_err is not None:
                    raw_output = f"<<error: {type(api_err).__name__}: {str(api_err)[:200]}>>"
                    letter, parse_status = None, "error"
                else:
                    letter, parse_status = parse_letter(raw_output)

            semantic = _semantic_choice(question, letter)
            is_correct = letter is not None and letter == question["gold_letter"]

            row = {
                "question_id": qid,
                "system": config.system,
                "setting": config.setting,
                "k": config.k,
                "model": config.model,
                "context_chunk_ids": context_chunk_ids,
                "raw_output": raw_output,
                "parsed_letter": letter,
                "parse_status": parse_status,
                "semantic_choice": semantic,
                "is_correct": is_correct,
                "timestamp": _utc_now_iso(),
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

            n_new += 1
            n_correct += int(is_correct)
            n_unparseable += int(parse_status == "unparseable")
            n_error += int(parse_status == "error")
            if progress and n_new % 25 == 0:
                print(f"  {n_new} answered ({n_unparseable} unparseable, {n_error} errors)")

    # Re-write the sidecar so any client-side parameter adaptation made during
    # the run (e.g. a model that forced max_completion_tokens / its default
    # temperature) is reflected in the recorded effective config.
    if client is not None and n_new:
        sidecar["client"] = client.describe()
        sidecar_path.write_text(
            json.dumps(sidecar, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

    return {
        "config_id": config_id,
        "preds_path": str(preds_path),
        "config_path": str(sidecar_path),
        "system": config.system,
        "setting": config.setting,
        "n_questions": len(questions),
        "n_new": n_new,
        "n_skipped_resumed": n_skipped,
        "n_correct_new": n_correct,
        "n_unparseable_new": n_unparseable,
        "n_error_new": n_error,
        "terminated_early": terminated,
        "accuracy_new": (n_correct / n_new) if n_new else None,
    }
