"""Synthetic fixtures for retrieval tests (contract schemas, no real data).

Not collected by pytest (no ``test_`` prefix); imported by the
``test_retrieval_*.py`` modules. Three fake texts' worth of chunks plus
questions, matching docs/CONTRACTS.md schemas exactly. Tests must run without
``outputs/processed/`` existing, without torch, and without network.
"""

from __future__ import annotations

import numpy as np


def make_chunk(chunk_id, text_id, chunk_text, start_char, end_char, n_tokens):
    return {
        "chunk_id": chunk_id,
        "text_id": text_id,
        "chunk_text": chunk_text,
        "start_char": start_char,
        "end_char": end_char,
        "n_tokens": n_tokens,
    }


CHUNKS = [
    # text1 — "d'Schoul" exercises clitic apostrophes
    make_chunk("text1_overlap_c000", "text1", "d'Schoul ass grouss an hell", 0, 27, 6),
    make_chunk("text1_overlap_c001", "text1", "de Mëtteg iesse mir eng Zopp", 27, 55, 6),
    make_chunk("text1_overlap_c002", "text1", "owes liest hien e Buch iwwer Geschicht", 55, 94, 7),
    # text2
    make_chunk("text2_overlap_c000", "text2", "Catherine ass glécklech mat hirem Hobby", 0, 40, 6),
    make_chunk("text2_overlap_c001", "text2", "si spillt gär Poker mat hire Frënn", 40, 74, 7),
    # text3
    make_chunk("text3_overlap_c000", "text3", "den Hond leeft séier duerch de Bësch", 0, 36, 7),
    make_chunk("text3_overlap_c001", "text3", "d'Kaz schléift am Gaart", 36, 59, 5),
]


def make_question(
    question_id,
    text_id,
    question,
    critical_span,
    distractor_span,
    presented=None,
):
    presented = presented or {
        "A": "éischt Optioun",
        "B": "zweet Optioun",
        "C": "drëtt Optioun",
        "D": "véiert Optioun",
    }
    return {
        "question_id": question_id,
        "text_id": text_id,
        "text_title": f"Titel {text_id}",
        "question": question,
        "cognitive_type": "Retrieve",
        "options": {
            "correct": presented["A"],
            "misunderstand": presented["B"],
            "distractor_span": presented["C"],
            "no_support": presented["D"],
        },
        "presented": presented,
        "permutation": ["correct", "misunderstand", "distractor_span", "no_support"],
        "gold_letter": "A",
        "shuffle_seed": 13,
        "critical_span": critical_span,
        "distractor_span": distractor_span,
        "linguistic_tags": [],
        "linguistic_categories": [],
    }


QUESTIONS = [
    # critical span sits inside text1_overlap_c000; distractor inside c002
    make_question(
        "text1_q00", "text1", "Wou ass d'Schoul?",
        critical_span={"start": 0, "end": 27, "status": "exact"},
        distractor_span={"start": 60, "end": 90, "status": "fuzzy"},
        presented={
            "A": "grouss an hell",
            "B": "kleng an donkel",
            "C": "e Buch iwwer Geschicht",
            "D": "um Mound",
        },
    ),
    # critical span inside text2_overlap_c001; distractor unresolved (null offsets)
    make_question(
        "text2_q00", "text2", "Wat spillt si gär?",
        critical_span={"start": 40, "end": 74, "status": "exact"},
        distractor_span={"start": None, "end": None, "status": "unresolved"},
    ),
    # unresolved critical span -> skipped by evidence metrics
    make_question(
        "text3_q00", "text3", "Wien leeft duerch de Bësch?",
        critical_span={"start": None, "end": None, "status": "unresolved"},
        distractor_span={"start": None, "end": None, "status": "empty"},
    ),
]


class FakeEmbedder:
    """Deterministic bag-of-words embedder (no torch, no network).

    Vocabulary defaults to all words in the fixture chunks. Records every
    call so tests can assert on prefixing and on cache behaviour.
    """

    def __init__(self, vocab=None):
        if vocab is None:
            vocab = sorted({
                w.strip(".,!?'").lower()
                for c in CHUNKS
                for w in c["chunk_text"].split()
            })
        self.vocab = {w: i for i, w in enumerate(vocab)}
        self.calls = []

    def __call__(self, texts):
        self.calls.append(list(texts))
        mat = np.zeros((len(texts), len(self.vocab) + 1), dtype=np.float64)
        for row, text in enumerate(texts):
            for word in text.lower().split():
                word = word.strip(".,!?")
                if word in self.vocab:
                    mat[row, self.vocab[word]] += 1.0
        mat[:, -1] = 1e-6  # never a zero vector
        return mat
