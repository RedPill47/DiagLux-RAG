"""Parsing of dataset/.../KnowledgeBaseAnnot.txt.

The file has 16 lines; each line is a single dict *entry* of the form
``'Title': [40 question dicts]``. Wrap the line in ``{}`` and parse with
``ast.literal_eval``. All 640 question dicts share an identical 9-key
schema (verified); schema validation is kept as a regression guard.
"""

from __future__ import annotations

import ast
import os
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path

from diaglux.data.texts import TEXT_IDS, find_data_root, load_clean_text

EXPECTED_KEYS = frozenset(
    {
        "textTitle",
        "cognitiveType",
        "criticalSpan",
        "distractorSpan",
        "question",
        "A_Correct",
        "B_Misunderstand",
        "C_Distractor_Span",
        "D_No_Support",
    }
)

COGNITIVE_TYPES = ("Retrieve", "Interpret", "Inferential", "Evaluative")

# Stored option field -> semantic option type (docs/CONTRACTS.md).
SEMANTIC_FROM_FIELD = {
    "A_Correct": "correct",
    "B_Misunderstand": "misunderstand",
    "C_Distractor_Span": "distractor_span",
    "D_No_Support": "no_support",
}


@dataclass
class QuestionRecord:
    question_id: str
    text_id: str
    text_title: str
    question: str
    cognitive_type: str
    critical_span_raw: str  # annotated-text substring, tags included
    distractor_span_raw: str
    options: dict[str, str] = field(default_factory=dict)  # semantic type -> text


def parse_kb_line(line: str) -> tuple[str, list[dict]]:
    """Parse one KnowledgeBaseAnnot.txt line into (title, question dicts).

    Validates the 9-key schema and the cognitive type of every entry.
    """
    entry = ast.literal_eval("{" + line.strip().rstrip(",") + "}")
    if not isinstance(entry, dict) or len(entry) != 1:
        raise ValueError("KB line did not parse to a single-entry dict")
    title, questions = next(iter(entry.items()))
    title = unicodedata.normalize("NFC", title)
    if not isinstance(questions, list):
        raise ValueError(f"KB entry for {title!r} is not a list")
    for i, q in enumerate(questions):
        keys = set(q)
        if keys != EXPECTED_KEYS:
            raise ValueError(
                f"KB schema violation in {title!r} question {i}: "
                f"missing {sorted(EXPECTED_KEYS - keys)}, extra {sorted(keys - EXPECTED_KEYS)}"
            )
        if q["cognitiveType"] not in COGNITIVE_TYPES:
            raise ValueError(
                f"Unknown cognitiveType {q['cognitiveType']!r} in {title!r} question {i}"
            )
    return title, questions


def title_to_text_id(data_root: str | os.PathLike | None = None) -> dict[str, str]:
    """Map each clean text's (NFC, stripped) title to its text_id."""
    root = data_root if data_root is not None else find_data_root()
    mapping: dict[str, str] = {}
    for text_id in TEXT_IDS:
        title, _, _ = load_clean_text(text_id, root)
        if title in mapping:
            raise ValueError(f"Duplicate title {title!r} ({mapping[title]} vs {text_id})")
        mapping[title] = text_id
    return mapping


def load_questions(
    data_root: str | os.PathLike | None = None,
) -> list[QuestionRecord]:
    """Parse the full knowledge base into QuestionRecords, ordered by text_id.

    ``question_id`` = ``"{text_id}_q{idx:02d}"`` with idx the 0-based position
    inside the text's question list.
    """
    root = Path(data_root) if data_root is not None else find_data_root()
    titles = title_to_text_id(root)
    kb_path = root / "KnowledgeBaseAnnot.txt"
    by_text: dict[str, list[QuestionRecord]] = {}
    for line in kb_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        title, questions = parse_kb_line(line)
        if title not in titles:
            raise ValueError(f"KB title {title!r} matches no clean text")
        text_id = titles[title]
        records = []
        for idx, q in enumerate(questions):
            inner_title = unicodedata.normalize("NFC", q["textTitle"])
            if inner_title != title:
                raise ValueError(
                    f"textTitle mismatch in {text_id} q{idx}: {inner_title!r} != {title!r}"
                )
            records.append(
                QuestionRecord(
                    question_id=f"{text_id}_q{idx:02d}",
                    text_id=text_id,
                    text_title=title,
                    question=unicodedata.normalize("NFC", q["question"]),
                    cognitive_type=q["cognitiveType"],
                    critical_span_raw=unicodedata.normalize("NFC", q["criticalSpan"]),
                    distractor_span_raw=unicodedata.normalize("NFC", q["distractorSpan"]),
                    options={
                        sem: unicodedata.normalize("NFC", q[fieldname])
                        for fieldname, sem in SEMANTIC_FROM_FIELD.items()
                    },
                )
            )
        if text_id in by_text:
            raise ValueError(f"Duplicate KB entry for {text_id}")
        by_text[text_id] = records
    missing = [t for t in TEXT_IDS if t not in by_text]
    if missing:
        raise ValueError(f"KB has no entry for: {missing}")
    out: list[QuestionRecord] = []
    for text_id in TEXT_IDS:
        out.extend(by_text[text_id])
    return out
