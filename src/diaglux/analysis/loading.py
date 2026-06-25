"""Loading and schema validation for analysis inputs.

Reads the artifacts defined in ``docs/CONTRACTS.md``:

- ``outputs/runs/preds_{config_id}.jsonl`` (+ ``preds_{config_id}.config.json`` sidecars)
- ``outputs/processed/questions.jsonl``
- ``outputs/retrieval/rankings_{setting}_{method}_{strategy}.jsonl``
- ``outputs/processed/corpus_chunks_{strategy}.jsonl``

Every loader validates the schema record by record and raises :class:`SchemaError`
(with file/line context) on the first violation -- analysis must fail loudly rather
than silently produce wrong tables.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

import pandas as pd

__all__ = [
    "SchemaError",
    "SEMANTIC_TYPES",
    "COGNITIVE_TYPES",
    "SETTINGS",
    "PARSE_STATUSES",
    "SPAN_STATUSES",
    "RESOLVED_SPAN_STATUSES",
    "LETTERS",
    "CONFIG_COLS",
    "load_questions",
    "load_preds_file",
    "load_runs",
    "load_rankings",
    "load_chunks",
    "join_runs_questions",
]


class SchemaError(ValueError):
    """A contract violation in an input file (see docs/CONTRACTS.md)."""


SEMANTIC_TYPES = ("correct", "misunderstand", "distractor_span", "no_support")
COGNITIVE_TYPES = ("Retrieve", "Interpret", "Inferential", "Evaluative")
SETTINGS = ("text_restricted", "open_corpus", "none")
PARSE_STATUSES = ("exact", "extracted", "unparseable", "error")
SPAN_STATUSES = ("exact", "dehyphen", "fuzzy", "multiple", "unresolved", "empty")
RESOLVED_SPAN_STATUSES = ("exact", "dehyphen", "fuzzy", "multiple")
LETTERS = ("A", "B", "C", "D")

#: Columns that identify one run configuration in the joined frame.
CONFIG_COLS = ["config_id", "system", "setting", "k", "model"]

_QUESTION_ID_RE = re.compile(r"^text\d+_q\d{2}$")
_CHUNK_ID_RE = re.compile(r"^text\d+_[A-Za-z0-9]+_c\d{3}$")


def _fail(path: Path | str, line: int | None, msg: str) -> None:
    loc = f"{path}" if line is None else f"{path}:{line}"
    raise SchemaError(f"{loc}: {msg}")


def _iter_jsonl(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            raw = raw.strip()
            if not raw:
                continue
            try:
                obj = json.loads(raw)
            except json.JSONDecodeError as exc:
                _fail(path, lineno, f"invalid JSON: {exc}")
            if not isinstance(obj, dict):
                _fail(path, lineno, "each JSONL line must be a JSON object")
            yield lineno, obj


def _require_keys(rec: dict, keys: list[str], path: Path, line: int) -> None:
    missing = [k for k in keys if k not in rec]
    if missing:
        _fail(path, line, f"missing required field(s): {missing}")


def _check_span(span: Any, field: str, path: Path, line: int) -> dict:
    if not isinstance(span, dict):
        _fail(path, line, f"{field} must be an object, got {type(span).__name__}")
    _require_keys(span, ["start", "end", "status"], path, line)
    status = span["status"]
    if status not in SPAN_STATUSES:
        _fail(path, line, f"{field}.status {status!r} not in {SPAN_STATUSES}")
    start, end = span["start"], span["end"]
    if status in RESOLVED_SPAN_STATUSES:
        if span.get("in_title"):
            # Span matched the story title, which is excluded from the clean body;
            # offsets are null and the span cannot overlap any body chunk.
            if start is not None or end is not None:
                _fail(path, line, f"{field}: in_title span must have null start/end")
            return span
        if not (isinstance(start, int) and isinstance(end, int) and not isinstance(start, bool)):
            _fail(path, line, f"{field}: resolved span must have integer start/end, got {start!r}/{end!r}")
        if not (0 <= start < end):
            _fail(path, line, f"{field}: need 0 <= start < end, got {start}/{end}")
    else:
        if start is not None or end is not None:
            _fail(path, line, f"{field}: start/end must be null when status is {status!r}")
    return span


# ---------------------------------------------------------------------------
# questions.jsonl
# ---------------------------------------------------------------------------

_QUESTION_KEYS = [
    "question_id", "text_id", "question", "cognitive_type", "options", "presented",
    "permutation", "gold_letter", "critical_span", "distractor_span",
    "linguistic_tags", "linguistic_categories",
]


def load_questions(path: str | Path) -> pd.DataFrame:
    """Load and validate questions.jsonl; one row per question.

    Adds flattened convenience columns ``critical_start/critical_end/critical_status``
    and ``distractor_start/distractor_end/distractor_status``.
    """
    path = Path(path)
    records: list[dict] = []
    seen: set[str] = set()
    for line, rec in _iter_jsonl(path):
        _require_keys(rec, _QUESTION_KEYS, path, line)
        qid = rec["question_id"]
        if not isinstance(qid, str) or not _QUESTION_ID_RE.match(qid):
            _fail(path, line, f"bad question_id {qid!r} (expected 'text<N>_q<NN>')")
        if qid in seen:
            _fail(path, line, f"duplicate question_id {qid!r}")
        seen.add(qid)
        if not qid.startswith(rec["text_id"] + "_"):
            _fail(path, line, f"question_id {qid!r} inconsistent with text_id {rec['text_id']!r}")
        if rec["cognitive_type"] not in COGNITIVE_TYPES:
            _fail(path, line, f"cognitive_type {rec['cognitive_type']!r} not in {COGNITIVE_TYPES}")
        perm = rec["permutation"]
        if not isinstance(perm, list) or sorted(perm) != sorted(SEMANTIC_TYPES):
            _fail(path, line, f"permutation {perm!r} is not a permutation of {SEMANTIC_TYPES}")
        expected_gold = chr(65 + perm.index("correct"))
        if rec["gold_letter"] != expected_gold:
            _fail(path, line,
                  f"gold_letter {rec['gold_letter']!r} inconsistent with permutation "
                  f"(expected {expected_gold!r})")
        opts = rec["options"]
        if not isinstance(opts, dict) or sorted(opts) != sorted(SEMANTIC_TYPES):
            _fail(path, line, f"options must have exactly the keys {SEMANTIC_TYPES}")
        presented = rec["presented"]
        if not isinstance(presented, dict) or sorted(presented) != sorted(LETTERS):
            _fail(path, line, f"presented must have exactly the keys {LETTERS}")
        crit = _check_span(rec["critical_span"], "critical_span", path, line)
        dist = _check_span(rec["distractor_span"], "distractor_span", path, line)
        cats = rec["linguistic_categories"]
        if not isinstance(cats, list) or not all(isinstance(c, str) for c in cats):
            _fail(path, line, "linguistic_categories must be a list of strings")
        if not isinstance(rec["linguistic_tags"], list):
            _fail(path, line, "linguistic_tags must be a list")
        flat = dict(rec)
        flat["critical_start"] = crit["start"]
        flat["critical_end"] = crit["end"]
        flat["critical_status"] = crit["status"]
        flat["distractor_start"] = dist["start"]
        flat["distractor_end"] = dist["end"]
        flat["distractor_status"] = dist["status"]
        records.append(flat)
    if not records:
        _fail(path, None, "no question records found")
    return pd.DataFrame.from_records(records)


# ---------------------------------------------------------------------------
# preds_{config_id}.jsonl (+ .config.json sidecar)
# ---------------------------------------------------------------------------

_PRED_KEYS = [
    "question_id", "system", "setting", "k", "model", "context_chunk_ids",
    "raw_output", "parsed_letter", "parse_status", "semantic_choice",
    "is_correct", "timestamp",
]

_PREDS_NAME_RE = re.compile(r"^preds_(?P<config_id>.+)\.jsonl$")


def _sidecar_path(preds_path: Path) -> Path:
    return preds_path.with_name(preds_path.name[: -len(".jsonl")] + ".config.json")


def load_preds_file(path: str | Path) -> pd.DataFrame:
    """Load one preds_*.jsonl file plus its .config.json sidecar.

    Returns one row per question with an added ``config_id`` column; the parsed
    sidecar dict is stored in ``df.attrs['config']``.
    """
    path = Path(path)
    m = _PREDS_NAME_RE.match(path.name)
    if not m:
        _fail(path, None, "preds file name must match 'preds_{config_id}.jsonl'")
    config_id = m.group("config_id")

    sidecar = _sidecar_path(path)
    if not sidecar.exists():
        _fail(path, None, f"missing required config sidecar {sidecar.name}")
    try:
        config = json.loads(sidecar.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        _fail(sidecar, None, f"invalid JSON in config sidecar: {exc}")
    if not isinstance(config, dict):
        _fail(sidecar, None, "config sidecar must be a JSON object")

    records: list[dict] = []
    seen: set[str] = set()
    for line, rec in _iter_jsonl(path):
        _require_keys(rec, _PRED_KEYS, path, line)
        qid = rec["question_id"]
        if not isinstance(qid, str) or not _QUESTION_ID_RE.match(qid):
            _fail(path, line, f"bad question_id {qid!r}")
        if qid in seen:
            _fail(path, line, f"duplicate question_id {qid!r} within one preds file")
        seen.add(qid)
        if rec["setting"] not in SETTINGS:
            _fail(path, line, f"setting {rec['setting']!r} not in {SETTINGS}")
        k = rec["k"]
        if k is not None and (not isinstance(k, int) or isinstance(k, bool) or k < 1):
            _fail(path, line, f"k must be a positive integer or null, got {k!r}")
        letter = rec["parsed_letter"]
        if letter is not None and letter not in LETTERS:
            _fail(path, line, f"parsed_letter {letter!r} must be one of {LETTERS} or null")
        if rec["parse_status"] not in PARSE_STATUSES:
            _fail(path, line, f"parse_status {rec['parse_status']!r} not in {PARSE_STATUSES}")
        if (rec["parse_status"] == "unparseable") != (letter is None):
            _fail(path, line,
                  "parsed_letter must be null if and only if parse_status is 'unparseable'")
        choice = rec["semantic_choice"]
        if choice is not None and choice not in SEMANTIC_TYPES:
            _fail(path, line, f"semantic_choice {choice!r} not in {SEMANTIC_TYPES} or null")
        if (letter is None) != (choice is None):
            _fail(path, line, "semantic_choice must be null exactly when parsed_letter is null")
        if not isinstance(rec["is_correct"], bool):
            _fail(path, line, f"is_correct must be a boolean, got {rec['is_correct']!r}")
        if choice is None and rec["is_correct"]:
            _fail(path, line, "unparseable predictions must have is_correct == false")
        if choice is not None and rec["is_correct"] != (choice == "correct"):
            _fail(path, line, "is_correct inconsistent with semantic_choice")
        if not isinstance(rec["context_chunk_ids"], list):
            _fail(path, line, "context_chunk_ids must be a list")
        rec = dict(rec)
        rec["config_id"] = config_id
        records.append(rec)
    if not records:
        _fail(path, None, "no prediction records found")
    df = pd.DataFrame.from_records(records)
    df["k"] = df["k"].astype("Int64")  # nullable int (null for non-retrieval systems)
    df.attrs["config"] = config
    return df


def load_runs(runs_dir: str | Path) -> pd.DataFrame:
    """Load every preds_*.jsonl under ``runs_dir`` into one DataFrame.

    One row per (config, question). Sidecar configs are collected in
    ``df.attrs['configs']`` keyed by config_id.
    """
    runs_dir = Path(runs_dir)
    files = sorted(runs_dir.glob("preds_*.jsonl"))
    if not files:
        _fail(runs_dir, None, "no preds_*.jsonl files found")
    frames, configs = [], {}
    for f in files:
        df = load_preds_file(f)
        configs[df["config_id"].iloc[0]] = df.attrs["config"]
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["k"] = out["k"].astype("Int64")
    out.attrs["configs"] = configs
    return out


# ---------------------------------------------------------------------------
# rankings + chunks (for retrieval diagnostics)
# ---------------------------------------------------------------------------

_RANKING_KEYS = ["question_id", "setting", "method", "query_mode", "chunk_strategy", "ranking"]


def load_rankings(path: str | Path) -> pd.DataFrame:
    """Load one rankings_*.jsonl file; one row per question.

    The ``ranking`` column holds the full list of chunk_ids ordered by rank.
    File-level metadata (setting/method/query_mode/chunk_strategy, validated to be
    uniform across the file) is stored in ``df.attrs``.
    """
    path = Path(path)
    records: list[dict] = []
    meta: dict[str, str] | None = None
    seen: set[str] = set()
    for line, rec in _iter_jsonl(path):
        _require_keys(rec, _RANKING_KEYS, path, line)
        qid = rec["question_id"]
        if not isinstance(qid, str) or not _QUESTION_ID_RE.match(qid):
            _fail(path, line, f"bad question_id {qid!r}")
        if qid in seen:
            _fail(path, line, f"duplicate question_id {qid!r}")
        seen.add(qid)
        if rec["setting"] not in ("text_restricted", "open_corpus"):
            _fail(path, line, f"setting {rec['setting']!r} invalid for rankings")
        this_meta = {key: rec[key] for key in ("setting", "method", "query_mode", "chunk_strategy")}
        if meta is None:
            meta = this_meta
        elif meta != this_meta:
            _fail(path, line, f"inconsistent file metadata: {this_meta} != {meta}")
        ranking = rec["ranking"]
        if not isinstance(ranking, list) or not ranking:
            _fail(path, line, "ranking must be a non-empty list")
        items = []
        for item in ranking:
            if not isinstance(item, dict) or "chunk_id" not in item or "rank" not in item:
                _fail(path, line, "each ranking item needs 'chunk_id' and 'rank'")
            items.append((item["rank"], item["chunk_id"]))
        ranks = [r for r, _ in items]
        if sorted(ranks) != list(range(1, len(ranks) + 1)):
            _fail(path, line, f"ranks must be 1..n without gaps, got {sorted(ranks)[:5]}...")
        ordered = [cid for _, cid in sorted(items)]
        records.append({"question_id": qid, "ranking": ordered})
    if not records:
        _fail(path, None, "no ranking records found")
    df = pd.DataFrame.from_records(records).set_index("question_id")
    df.attrs.update(meta or {})
    return df


_CHUNK_KEYS = ["chunk_id", "text_id", "chunk_text", "start_char", "end_char", "n_tokens"]


def load_chunks(path: str | Path) -> pd.DataFrame:
    """Load a corpus_chunks_*.jsonl file, indexed by chunk_id."""
    path = Path(path)
    records: list[dict] = []
    seen: set[str] = set()
    for line, rec in _iter_jsonl(path):
        _require_keys(rec, _CHUNK_KEYS, path, line)
        cid = rec["chunk_id"]
        if not isinstance(cid, str) or not _CHUNK_ID_RE.match(cid):
            _fail(path, line, f"bad chunk_id {cid!r}")
        if cid in seen:
            _fail(path, line, f"duplicate chunk_id {cid!r}")
        seen.add(cid)
        s, e = rec["start_char"], rec["end_char"]
        if not (isinstance(s, int) and isinstance(e, int) and 0 <= s < e):
            _fail(path, line, f"need 0 <= start_char < end_char, got {s!r}/{e!r}")
        if not cid.startswith(rec["text_id"] + "_"):
            _fail(path, line, f"chunk_id {cid!r} inconsistent with text_id {rec['text_id']!r}")
        records.append(rec)
    if not records:
        _fail(path, None, "no chunk records found")
    return pd.DataFrame.from_records(records).set_index("chunk_id")


# ---------------------------------------------------------------------------
# Join
# ---------------------------------------------------------------------------

def join_runs_questions(preds: pd.DataFrame, questions: pd.DataFrame) -> pd.DataFrame:
    """Join predictions with question metadata; one row per (config, question).

    Cross-validates each prediction against the question's permutation:
    ``semantic_choice`` must equal ``permutation[parsed_letter]``. Fails loudly on
    unknown question_ids, duplicate (config, question) pairs, or inconsistencies.
    """
    unknown = set(preds["question_id"]) - set(questions["question_id"])
    if unknown:
        raise SchemaError(
            f"predictions reference {len(unknown)} question_id(s) absent from "
            f"questions.jsonl, e.g. {sorted(unknown)[:5]}"
        )
    dup = preds.duplicated(subset=["config_id", "question_id"])
    if dup.any():
        bad = preds.loc[dup, ["config_id", "question_id"]].iloc[0].tolist()
        raise SchemaError(f"duplicate (config_id, question_id) pair: {bad}")

    qcols = [
        "question_id", "text_id", "cognitive_type", "permutation", "gold_letter",
        "linguistic_tags", "linguistic_categories",
        "critical_start", "critical_end", "critical_status",
        "distractor_start", "distractor_end", "distractor_status",
    ]
    joined = preds.merge(questions[qcols], on="question_id", how="left", validate="m:1")

    for row in joined.itertuples(index=False):
        # None in raw JSON arrives as NaN in pandas columns (pandas >= 3
        # converts None to NaN even in object/str columns), so use pd.isna.
        if pd.isna(row.parsed_letter):
            continue
        expected = row.permutation[ord(row.parsed_letter) - 65]
        if row.semantic_choice != expected:
            raise SchemaError(
                f"config {row.config_id}, question {row.question_id}: semantic_choice "
                f"{row.semantic_choice!r} does not match permutation for letter "
                f"{row.parsed_letter!r} (expected {expected!r})"
            )
    joined.attrs["configs"] = preds.attrs.get("configs", {})
    return joined
