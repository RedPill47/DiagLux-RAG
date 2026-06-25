"""Canonical clean-text loading.

``load_clean_text`` is the single definition of the clean-text coordinate
system (docs/CONTRACTS.md): the body is the raw file content with the title
line (line 1) and author line (line 2) removed, NFC-normalized, with all
other whitespace preserved exactly. Every span offset and chunk offset in
the pipeline indexes into this body string.
"""

from __future__ import annotations

import os
import unicodedata
from pathlib import Path

TEXT_IDS = tuple(f"text{i}" for i in range(1, 17))

# Candidate locations of the raw dataset, relative to a base directory.
# CONTRACTS.md names dataset/dataset/...; the checked-out tree has dataset/...
_CANDIDATE_SUBDIRS = ("dataset/dataset", "dataset")


def find_data_root(base: str | os.PathLike | None = None) -> Path:
    """Locate the directory containing Texts/, Annotations/, KnowledgeBaseAnnot.txt.

    Honors the DIAGLUX_DATA_ROOT environment variable, then searches the
    repo root (three levels above this package) and the current working
    directory for ``dataset/dataset`` and ``dataset``.
    """
    candidates: list[Path] = []
    if base is not None:
        b = Path(base)
        candidates.extend([b] + [b / sub for sub in _CANDIDATE_SUBDIRS])
    env = os.environ.get("DIAGLUX_DATA_ROOT")
    if env:
        candidates.append(Path(env))
    repo_root = Path(__file__).resolve().parents[3]
    for top in (repo_root, Path.cwd()):
        candidates.extend(top / sub for sub in _CANDIDATE_SUBDIRS)
    for cand in candidates:
        if (cand / "Texts").is_dir() and (cand / "KnowledgeBaseAnnot.txt").is_file():
            return cand
    raise FileNotFoundError(
        "Could not locate the raw dataset (Texts/ + KnowledgeBaseAnnot.txt). "
        "Searched: " + ", ".join(str(c) for c in candidates)
    )


def load_clean_text(
    text_id: str, data_root: str | os.PathLike | None = None
) -> tuple[str, str, str]:
    """Return ``(title, author, body)`` for a clean text.

    - title  = line 1, stripped, NFC-normalized
    - author = line 2, stripped, NFC-normalized (do NOT rely on ALL-CAPS:
      author casing is mixed across files)
    - body   = everything after the author line's newline, NFC-normalized,
      whitespace otherwise preserved (the canonical coordinate system)
    """
    root = data_root if data_root is not None else find_data_root()
    path = Path(root) / "Texts" / f"{text_id}.txt"
    raw = path.read_text(encoding="utf-8")
    # Tolerate (absent in practice) CRLF endings without disturbing offsets
    # for the LF files actually shipped.
    if "\r\n" in raw:
        raw = raw.replace("\r\n", "\n")
    try:
        title_line, rest = raw.split("\n", 1)
        author_line, body = rest.split("\n", 1)
    except ValueError as exc:  # fewer than 3 lines
        raise ValueError(f"{path} does not have title/author/body structure") from exc
    title = unicodedata.normalize("NFC", title_line.strip())
    author = unicodedata.normalize("NFC", author_line.strip())
    body = unicodedata.normalize("NFC", body)
    return title, author, body
