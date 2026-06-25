"""Context building: oracle clean text and top-k retrieval contexts."""

import pytest

from answering_testutils import (
    TEXT1_BODY_NFC,
    TEXT2_BODY,
    make_questions,
    write_chunks,
    write_rankings,
    write_texts_dir,
)
from diaglux.answering.context import (
    CHUNK_SEPARATOR,
    build_oracle_context,
    build_retrieval_context,
    load_chunks,
    load_rankings,
)


@pytest.fixture()
def texts_dir(tmp_path):
    return write_texts_dir(tmp_path / "Texts")


@pytest.fixture()
def ranking_and_chunks(tmp_path):
    questions = make_questions()
    chunks_path = tmp_path / "corpus_chunks_overlap.jsonl"
    rankings_path = tmp_path / "rankings_text_restricted_bm25_overlap.jsonl"
    write_chunks(chunks_path)
    write_rankings(rankings_path, questions)
    return load_rankings(rankings_path), load_chunks(chunks_path)


def test_oracle_context_is_clean_nfc_body(texts_dir):
    context, chunk_ids = build_oracle_context("text1", texts_dir=texts_dir)
    # Title + author lines stripped; body NFC-normalised (file is NFD).
    assert context == TEXT1_BODY_NFC
    assert chunk_ids == ["text1_full"]


def test_oracle_context_other_text(texts_dir):
    context, chunk_ids = build_oracle_context("text2", texts_dir=texts_dir)
    assert context == TEXT2_BODY
    assert chunk_ids == ["text2_full"]


def test_oracle_context_missing_text_raises(texts_dir):
    with pytest.raises(FileNotFoundError):
        build_oracle_context("text99", texts_dir=texts_dir)


def test_top_k_selection_follows_rank_not_file_order(ranking_and_chunks):
    rankings, chunks = ranking_and_chunks
    # Fixture entries are deliberately NOT stored sorted by rank:
    # rank 1 = c002, rank 2 = c001, rank 3 = c000.
    context, chunk_ids = build_retrieval_context(
        rankings["text1_q00"], chunks, k=2
    )
    assert chunk_ids == ["text1_overlap_c002", "text1_overlap_c001"]
    assert context == "Chunk 2 vum text1." + CHUNK_SEPARATOR + "Chunk 1 vum text1."


def test_top_k_full_when_k_exceeds_ranking_length(ranking_and_chunks):
    rankings, chunks = ranking_and_chunks
    _context, chunk_ids = build_retrieval_context(
        rankings["text2_q00"], chunks, k=10
    )
    assert chunk_ids == [
        "text2_overlap_c002", "text2_overlap_c001", "text2_overlap_c000",
    ]


def test_k_one_takes_only_rank_one(ranking_and_chunks):
    rankings, chunks = ranking_and_chunks
    context, chunk_ids = build_retrieval_context(
        rankings["text1_q01"], chunks, k=1
    )
    assert chunk_ids == ["text1_overlap_c002"]
    assert CHUNK_SEPARATOR not in context


def test_nonpositive_k_raises(ranking_and_chunks):
    rankings, chunks = ranking_and_chunks
    with pytest.raises(ValueError):
        build_retrieval_context(rankings["text1_q00"], chunks, k=0)


def test_missing_chunk_id_raises_keyerror(ranking_and_chunks):
    rankings, chunks = ranking_and_chunks
    chunks = dict(chunks)
    del chunks["text1_overlap_c002"]
    with pytest.raises(KeyError):
        build_retrieval_context(rankings["text1_q00"], chunks, k=2)


def test_load_rankings_rejects_duplicate_question_id(tmp_path):
    questions = make_questions()
    path = tmp_path / "rankings_dup.jsonl"
    write_rankings(path, [questions[0], questions[0]])
    with pytest.raises(ValueError, match="Duplicate question_id"):
        load_rankings(path)
