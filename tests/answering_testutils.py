"""Shared synthetic fixtures for the answering-harness tests.

Builds a tiny questions.jsonl (contract schema, docs/CONTRACTS.md), matching
clean-text files, a chunk file, and a rankings file. Nothing here touches
outputs/processed/ or the real dataset; everything is written under tmp_path.

Source kept ASCII-only; Luxembourgish diacritics use \\u escapes so the
normalization form of every literal is unambiguous.
"""

from __future__ import annotations

import json
from pathlib import Path

SEMANTIC_TYPES = ("correct", "misunderstand", "distractor_span", "no_support")

# "glécklech" with a PRECOMPOSED e-acute (NFC).
GLECKLECH_NFC = "gl\u00e9cklech"
# "glécklech" with DECOMPOSED e + combining acute (NFD).
GLECKLECH_NFD = "glécklech"

# (question_id, text_id, question, permutation) -- gold_letter derived.
_QUESTION_SPECS = [
    (
        "text1_q00",
        "text1",
        "Wat huet d'Catherine am Gaart fonnt?",
        ["distractor_span", "correct", "no_support", "misunderstand"],
    ),
    (
        "text1_q01",
        "text1",
        "Firwat ass de Jang fortgaang?",
        ["correct", "misunderstand", "distractor_span", "no_support"],
    ),
    (
        "text2_q00",
        "text2",
        "Wéi eng Faarf hat den Auto?",
        ["no_support", "misunderstand", "distractor_span", "correct"],
    ),
    (
        "text2_q01",
        "text2",
        "Wat mengt d'Erzielerin um Enn vum Text?",
        ["misunderstand", "no_support", "correct", "distractor_span"],
    ),
]

_TEXT_TITLES = {
    "text1": "Catherine, ech sinn esou " + GLECKLECH_NFC,
    "text2": "De Wee heem",
}


def make_question(question_id, text_id, question, permutation):
    options = {
        "correct": f"Richteg Antwert fir {question_id}.",
        "misunderstand": f"Falsch verstane Antwert fir {question_id}.",
        "distractor_span": f"Distractor-Span Antwert fir {question_id}.",
        "no_support": f"Net gestetzt Antwert fir {question_id}.",
    }
    presented = {
        chr(65 + i): options[semantic] for i, semantic in enumerate(permutation)
    }
    gold_letter = chr(65 + permutation.index("correct"))
    return {
        "question_id": question_id,
        "text_id": text_id,
        "text_title": _TEXT_TITLES[text_id],
        "question": question,
        "cognitive_type": "Retrieve",
        "options": options,
        "presented": presented,
        "permutation": list(permutation),
        "gold_letter": gold_letter,
        "shuffle_seed": 13,
        "critical_span": {"start": 10, "end": 60, "status": "exact"},
        "distractor_span": {"start": 70, "end": 110, "status": "fuzzy"},
        "linguistic_tags": ["LEX-FALSE-FRIEND"],
        "linguistic_categories": ["LEX"],
    }


def make_questions():
    return [make_question(*spec) for spec in _QUESTION_SPECS]


def write_questions(path: Path) -> list[dict]:
    questions = make_questions()
    with open(path, "w", encoding="utf-8") as handle:
        for question in questions:
            handle.write(json.dumps(question, ensure_ascii=False) + "\n")
    return questions


# Body of text1 as written to disk (NFD) and as the loader must return (NFC).
TEXT1_BODY_RAW = (
    "D'Catherine souz am Gaart.\n"
    f"Si war {GLECKLECH_NFD} an huet e Bridfchen fonnt.\n"
)
TEXT1_BODY_NFC = (
    "D'Catherine souz am Gaart.\n"
    f"Si war {GLECKLECH_NFC} an huet e Bridfchen fonnt.\n"
)
TEXT2_BODY = "Den Auto war blo.\nEt war schonn donkel.\n"


def write_texts_dir(directory: Path) -> Path:
    """Two tiny raw text files: line1 title, line2 author, rest body.
    text1's body contains an NFD-decomposed character to exercise NFC."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "text1.txt").write_text(
        _TEXT_TITLES["text1"] + "\nCathy Clement\n" + TEXT1_BODY_RAW,
        encoding="utf-8",
    )
    (directory / "text2.txt").write_text(
        _TEXT_TITLES["text2"] + "\nJosy Braun\n" + TEXT2_BODY,
        encoding="utf-8",
    )
    return directory


def write_chunks(path: Path) -> list[dict]:
    chunks = [
        {"chunk_id": f"{tid}_overlap_c{idx:03d}", "text_id": tid,
         "chunk_text": f"Chunk {idx} vum {tid}.",
         "start_char": idx * 40, "end_char": idx * 40 + 39, "n_tokens": 10}
        for tid in ("text1", "text2")
        for idx in range(3)
    ]
    with open(path, "w", encoding="utf-8") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    return chunks


def write_rankings(path: Path, questions) -> list[dict]:
    """Full ranking over the question's own text chunks; the entry list is
    deliberately NOT sorted by rank to exercise rank-based sorting."""
    records = []
    for question in questions:
        tid = question["text_id"]
        entries = [
            {"chunk_id": f"{tid}_overlap_c001", "score": 9.0, "rank": 2},
            {"chunk_id": f"{tid}_overlap_c002", "score": 12.5, "rank": 1},
            {"chunk_id": f"{tid}_overlap_c000", "score": 3.0, "rank": 3},
        ]
        records.append({
            "question_id": question["question_id"],
            "setting": "text_restricted",
            "method": "bm25",
            "query_mode": "question_options",
            "chunk_strategy": "overlap",
            "ranking": entries,
        })
    with open(path, "w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return records


def read_jsonl(path: Path) -> list[dict]:
    with open(path, "r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
