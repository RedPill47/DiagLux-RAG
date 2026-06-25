"""Chunking: offsets, overlap coverage, paragraph and sentence splitting."""

import pytest

from diaglux.data.chunking import (
    check_full_coverage,
    chunk_overlap,
    chunk_paragraph,
    chunk_sentence,
)

# ~520 whitespace tokens, hard-wrapped like the real texts (no blank lines).
WORDS = [f"wuert{i}." if i % 12 == 11 else f"wuert{i}" for i in range(520)]
LINES = [" ".join(WORDS[i : i + 10]) for i in range(0, len(WORDS), 10)]
BODY = "\n".join(LINES) + "\n"


def _offsets_consistent(chunks, body):
    for c in chunks:
        assert c.chunk_text == body[c.start_char : c.end_char]
        assert c.n_tokens == len(c.chunk_text.split())


def test_overlap_full_coverage_and_offsets():
    chunks = chunk_overlap("textX", BODY, window=150, overlap=0.5)
    _offsets_consistent(chunks, BODY)
    check_full_coverage(chunks, len(BODY))  # must not raise
    assert chunks[0].start_char == 0
    assert chunks[-1].end_char == len(BODY)
    # ~150-token windows with 50% overlap over 520 tokens -> ceil((520-150)/75)+1
    assert len(chunks) == 6
    assert all(c.n_tokens <= 150 for c in chunks)
    # consecutive chunks actually overlap
    for a, b in zip(chunks, chunks[1:]):
        assert b.start_char < a.end_char


def test_overlap_coverage_check_detects_gap():
    chunks = chunk_overlap("textX", BODY)
    with pytest.raises(AssertionError):
        check_full_coverage(chunks[1:], len(BODY))


def test_paragraph_blank_line_split():
    body = "Éischte Paragraph mat e puer Wierder.\n\nZweete Paragraph, och kuerz.\n"
    chunks = chunk_paragraph("textX", body)
    _offsets_consistent(chunks, body)
    assert [c.chunk_text for c in chunks] == [
        "Éischte Paragraph mat e puer Wierder.",
        "Zweete Paragraph, och kuerz.",
    ]


def test_paragraph_natural_units_without_blank_lines():
    chunks = chunk_paragraph("textX", BODY)
    _offsets_consistent(chunks, BODY)
    assert len(chunks) > 1  # hard-wrapped text must still be segmented
    assert all(c.n_tokens <= 180 for c in chunks)
    # spans are ordered and non-overlapping
    for a, b in zip(chunks, chunks[1:]):
        assert a.end_char <= b.start_char


def test_sentence_split():
    body = "Hien ass frou. Si freet: wierklech? Jo!\nEt war eng laang Nuecht …\n"
    chunks = chunk_sentence("textX", body)
    _offsets_consistent(chunks, body)
    assert [c.chunk_text for c in chunks] == [
        "Hien ass frou.",
        "Si freet: wierklech?",
        "Jo!",
        "Et war eng laang Nuecht …",
    ]


def test_sentence_split_does_not_break_inside_token():
    body = "Et kascht 1000-mol méi.Esou ass et gutt.\n"
    chunks = chunk_sentence("textX", body)
    # "méi.Esou" has no whitespace after the period -> no split there
    assert chunks[0].chunk_text == "Et kascht 1000-mol méi.Esou ass et gutt."
