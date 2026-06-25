"""BM25: exact-match chunk ranks first; full rankings; sane scores."""

import math

import numpy as np

from _retrieval_fixtures import CHUNKS
from diaglux.retrieval.bm25 import BM25Index
from diaglux.retrieval.tokenize import word_tokenize


def test_exact_match_chunk_ranks_first():
    index = BM25Index.from_chunks(CHUNKS)
    query = word_tokenize("d'Schoul ass grouss an hell")  # == text1_overlap_c000
    ranked = index.rank(query)
    assert ranked[0][0] == "text1_overlap_c000"
    assert ranked[0][1] > 0.0


def test_full_ranking_covers_every_doc():
    index = BM25Index.from_chunks(CHUNKS)
    ranked = index.rank(word_tokenize("Poker"))
    assert len(ranked) == len(CHUNKS)
    assert {cid for cid, _ in ranked} == {c["chunk_id"] for c in CHUNKS}
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)


def test_term_present_beats_term_absent():
    index = BM25Index.from_chunks(CHUNKS)
    scores = dict(index.rank(word_tokenize("Poker")))
    assert scores["text2_overlap_c001"] > scores["text1_overlap_c000"]
    assert scores["text1_overlap_c000"] == 0.0  # no query term present


def test_unknown_terms_score_zero_everywhere():
    index = BM25Index.from_chunks(CHUNKS)
    assert np.all(index.score(word_tokenize("xylophon zzz")) == 0.0)


def test_idf_is_lucene_variant_and_nonnegative():
    docs = [["a", "b"], ["a", "c"], ["a", "d"], ["b", "c"]]
    index = BM25Index(docs)
    n, df = 4, 3  # term "a" in 3 of 4 docs -> classic Robertson IDF would be negative
    assert index.idf("a") == math.log(1 + (n - df + 0.5) / (df + 0.5))
    assert index.idf("a") > 0.0
    assert index.idf("unseen") == 0.0


def test_shorter_doc_wins_length_normalization():
    # same tf, shorter document must score higher (b = 0.75)
    docs = [["zopp", "x", "x", "x", "x", "x"], ["zopp", "x"]]
    index = BM25Index(docs, doc_ids=["long", "short"])
    ranked = index.rank(["zopp"])
    assert ranked[0][0] == "short"


def test_doc_ids_default_to_indices():
    index = BM25Index([["a"], ["b"]])
    assert index.doc_ids == [0, 1]
