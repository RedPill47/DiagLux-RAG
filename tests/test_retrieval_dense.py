"""DenseRetriever with an injected fake embedder (no torch, no network)."""

import sys

import numpy as np
import pytest

from _retrieval_fixtures import CHUNKS, FakeEmbedder
from diaglux.retrieval.dense import DenseRetriever, default_prefixes, model_slug


def test_module_import_does_not_import_torch():
    import diaglux.retrieval.dense  # noqa: F401

    assert "torch" not in sys.modules
    assert "sentence_transformers" not in sys.modules


def test_injected_embedder_never_triggers_lazy_import():
    DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(), cache_dir=None)
    assert "sentence_transformers" not in sys.modules


def test_rank_matching_chunk_first():
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(), cache_dir=None)
    r.index_chunks(CHUNKS)
    ranked = r.rank("si spillt gär Poker mat hire Frënn")
    assert ranked[0][0] == "text2_overlap_c001"
    assert len(ranked) == len(CHUNKS)  # full ranking
    scores = [s for _, s in ranked]
    assert scores == sorted(scores, reverse=True)
    assert all(-1.0001 <= s <= 1.0001 for s in scores)  # cosine range


def test_candidate_restriction():
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(), cache_dir=None)
    r.index_chunks(CHUNKS)
    cands = [c["chunk_id"] for c in CHUNKS if c["text_id"] == "text1"]
    ranked = r.rank("d'Schoul ass grouss", candidate_ids=cands)
    assert [cid for cid, _ in ranked] != []
    assert {cid for cid, _ in ranked} == set(cands)


def test_e5_prefixes_applied():
    emb = FakeEmbedder()
    r = DenseRetriever(model_name="intfloat/multilingual-e5-base", embed_fn=emb, cache_dir=None)
    assert (r.query_prefix, r.passage_prefix) == ("query: ", "passage: ")
    r.index_chunks(CHUNKS[:2])
    r.rank("Wou ass d'Schoul?")
    passage_call, query_call = emb.calls
    assert all(t.startswith("passage: ") for t in passage_call)
    assert query_call[0].startswith("query: ")


def test_bge_m3_has_no_prefixes():
    assert default_prefixes("BAAI/bge-m3") == ("", "")
    emb = FakeEmbedder()
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=emb, cache_dir=None)
    r.index_chunks(CHUNKS[:1])
    assert emb.calls[0][0] == CHUNKS[0]["chunk_text"]


def test_prefix_override():
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(),
                       cache_dir=None, query_prefix="Q: ", passage_prefix="P: ")
    assert (r.query_prefix, r.passage_prefix) == ("Q: ", "P: ")


def test_embedding_cache_roundtrip(tmp_path):
    emb1 = FakeEmbedder()
    r1 = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=emb1, cache_dir=tmp_path)
    r1.index_chunks(CHUNKS)
    cache_file = r1.cache_path(CHUNKS)
    assert cache_file.exists()
    assert len(emb1.calls) == 1  # one passage-encoding call

    # second retriever: cache hit, embedder never called for passages
    emb2 = FakeEmbedder()
    r2 = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=emb2, cache_dir=tmp_path)
    r2.index_chunks(CHUNKS)
    assert emb2.calls == []
    assert r1.rank("Poker") == r2.rank("Poker")


def test_cache_key_changes_with_chunks(tmp_path):
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(), cache_dir=tmp_path)
    assert r.cache_path(CHUNKS) != r.cache_path(CHUNKS[:3])
    assert r.cache_path(CHUNKS).name.startswith("BAAI_bge-m3_")


def test_model_slug():
    assert model_slug("intfloat/multilingual-e5-base") == "intfloat_multilingual-e5-base"
    assert "/" not in model_slug("a/b/c")


def test_embeddings_are_normalized():
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(), cache_dir=None)
    r.index_chunks(CHUNKS)
    norms = np.linalg.norm(r._chunk_emb, axis=1)
    assert np.allclose(norms, 1.0)


def test_score_before_index_raises():
    r = DenseRetriever(model_name="BAAI/bge-m3", embed_fn=FakeEmbedder(), cache_dir=None)
    with pytest.raises(RuntimeError):
        r.score("Moien")
