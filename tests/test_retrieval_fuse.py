"""Fusion: RRF on a hand-computed example; weighted min-max fusion."""

import pytest

from diaglux.retrieval.fuse import reciprocal_rank_fusion, weighted_minmax_fusion


def test_rrf_hand_computed():
    # ranking 1: a > b > c (with scores); ranking 2: b > c > a (ids only)
    r1 = [("a", 9.0), ("b", 5.0), ("c", 1.0)]
    r2 = ["b", "c", "a"]
    fused = dict(reciprocal_rank_fusion([r1, r2], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 63)
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["c"] == pytest.approx(1 / 63 + 1 / 62)
    order = [d for d, _ in reciprocal_rank_fusion([r1, r2], k=60)]
    assert order == ["b", "a", "c"]  # b: .0325, a: .0323, c: .0320


def test_rrf_document_missing_from_one_ranking():
    fused = dict(reciprocal_rank_fusion([["a", "b"], ["a"]], k=60))
    assert fused["a"] == pytest.approx(2 / 61)
    assert fused["b"] == pytest.approx(1 / 62)  # only one contribution


def test_rrf_returns_sorted_full_ranking():
    out = reciprocal_rank_fusion([["a", "b", "c"], ["c", "b", "a"]], k=60)
    scores = [s for _, s in out]
    assert scores == sorted(scores, reverse=True)
    assert {d for d, _ in out} == {"a", "b", "c"}


def test_weighted_minmax_hand_computed():
    bm25 = {"a": 2.0, "b": 1.0, "c": 0.0}   # min-max -> a:1, b:.5, c:0
    dense = {"a": 0.1, "b": 0.9, "c": 0.5}  # min-max -> a:0, b:1, c:.5
    fused = dict(weighted_minmax_fusion(bm25, dense, alpha=0.7))
    assert fused["a"] == pytest.approx(0.7 * 1.0 + 0.3 * 0.0)
    assert fused["b"] == pytest.approx(0.7 * 0.5 + 0.3 * 1.0)
    assert fused["c"] == pytest.approx(0.7 * 0.0 + 0.3 * 0.5)
    assert [d for d, _ in weighted_minmax_fusion(bm25, dense, alpha=0.7)] == ["a", "b", "c"]
    # alpha = 0.3 flips the order toward dense
    assert [d for d, _ in weighted_minmax_fusion(bm25, dense, alpha=0.3)][0] == "b"


def test_weighted_minmax_union_and_missing_docs():
    # doc only in dense gets bm25's minimum (0 after normalization)
    fused = dict(weighted_minmax_fusion({"a": 2.0, "b": 1.0}, {"a": 0.5, "z": 1.0}, alpha=0.5))
    assert set(fused) == {"a", "b", "z"}
    assert fused["z"] == pytest.approx(0.5 * 0.0 + 0.5 * 1.0)


def test_weighted_minmax_constant_scores_normalize_to_zero():
    fused = dict(weighted_minmax_fusion({"a": 3.0, "b": 3.0}, {"a": 1.0, "b": 0.0}, alpha=0.5))
    assert fused["a"] == pytest.approx(0.5)  # bm25 side contributes 0 for everyone
    assert fused["b"] == pytest.approx(0.0)


def test_weighted_minmax_alpha_validation():
    with pytest.raises(ValueError):
        weighted_minmax_fusion({"a": 1.0}, {"a": 1.0}, alpha=1.5)
