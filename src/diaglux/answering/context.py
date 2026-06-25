"""Context construction for each answering system.

Systems (docs/CONTRACTS.md preds schema):

- ``oracle``      - the full clean text body of the question's source text.
- retrieval (rag) - top-k chunk texts from a rankings file + chunk file.
- ``closed_book`` - no context (the prompt's closed-book variant).
- ``random``      - no LLM at all; handled in the runner with a seeded
                    uniform choice over A-D.

The clean-text definition is owned by the data module
(``diaglux.data.load_clean_text``). Because that module is developed in
parallel, we import it lazily and defensively; if it is not available yet we
apply the contract rule inline: line 1 = title, line 2 = author, rest = body,
NFC-normalised, original whitespace otherwise preserved.
"""

from __future__ import annotations

import json
import unicodedata
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

# docs/CONTRACTS.md names dataset/dataset/Texts; the checkout in this repo
# has dataset/Texts. Try both, in contract order.
CANDIDATE_TEXTS_DIRS = (
    Path("dataset/dataset/Texts"),
    Path("dataset/Texts"),
)

# Separator between concatenated retrieved chunks.
CHUNK_SEPARATOR = "\n\n"


def _inline_load_clean_text(
    text_id: str, texts_dir: Optional[Path] = None
) -> Tuple[str, str, str]:
    """Contract fallback loader: (title, author, NFC body)."""
    if texts_dir is not None:
        candidates = [Path(texts_dir)]
    else:
        candidates = [d for d in CANDIDATE_TEXTS_DIRS]
    for directory in candidates:
        path = directory / f"{text_id}.txt"
        if path.exists():
            break
    else:
        tried = ", ".join(str(d / f"{text_id}.txt") for d in candidates)
        raise FileNotFoundError(f"Clean text source not found; tried: {tried}")
    raw = path.read_text(encoding="utf-8")
    lines = raw.split("\n")
    title = lines[0].strip()
    author = lines[1].strip() if len(lines) > 1 else ""
    body = "\n".join(lines[2:])
    return title, author, unicodedata.normalize("NFC", body)


def load_clean_text(
    text_id: str, texts_dir: Optional[Path] = None
) -> Tuple[str, str, str]:
    """Return (title, author, clean body) for ``text_id``.

    Prefers ``diaglux.data.load_clean_text`` (single shared definition).
    Falls back to the inline contract rule when the data module is not yet
    importable or does not yet expose the loader. An explicit ``texts_dir``
    always uses the inline loader (the data module would not know about it).
    """
    if texts_dir is None:
        try:
            from diaglux import data as _data  # lazy: built in parallel

            loader = getattr(_data, "load_clean_text", None)
            if loader is not None:
                return loader(text_id)
        except Exception:
            pass
    return _inline_load_clean_text(text_id, texts_dir)


def build_oracle_context(
    text_id: str, texts_dir: Optional[Path] = None
) -> Tuple[str, List[str]]:
    """Full clean body as context. Pseudo chunk id ``{text_id}_full`` is
    recorded in ``context_chunk_ids`` so oracle rows are self-describing."""
    _title, _author, body = load_clean_text(text_id, texts_dir)
    return body, [f"{text_id}_full"]


def load_rankings(path) -> Dict[str, dict]:
    """Load a rankings_{setting}_{method}_{strategy}.jsonl file keyed by
    question_id. Duplicate question_ids are an error."""
    rankings: Dict[str, dict] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            qid = record["question_id"]
            if qid in rankings:
                raise ValueError(f"Duplicate question_id in rankings file: {qid}")
            rankings[qid] = record
    return rankings


def load_chunks(path) -> Dict[str, str]:
    """Load a corpus_chunks_{strategy}.jsonl file as chunk_id -> chunk_text."""
    chunks: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            chunks[record["chunk_id"]] = record["chunk_text"]
    return chunks


def build_retrieval_context(
    ranking_record: Mapping,
    chunks_by_id: Mapping[str, str],
    k: int,
) -> Tuple[str, List[str]]:
    """Concatenate the top-k chunk texts (by rank) with CHUNK_SEPARATOR.

    Rankings files store the FULL ranking; top-k is a slice. Returns
    (context string, ordered chunk ids).
    """
    if k <= 0:
        raise ValueError(f"k must be positive, got {k}")
    ranked = sorted(ranking_record["ranking"], key=lambda entry: entry["rank"])
    top = ranked[:k]
    chunk_ids = [entry["chunk_id"] for entry in top]
    missing = [cid for cid in chunk_ids if cid not in chunks_by_id]
    if missing:
        raise KeyError(
            f"Chunk ids in ranking but missing from chunk file: {missing}"
        )
    context = CHUNK_SEPARATOR.join(chunks_by_id[cid] for cid in chunk_ids)
    return context, chunk_ids
