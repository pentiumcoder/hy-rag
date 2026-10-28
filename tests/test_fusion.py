from __future__ import annotations

import pytest

from hyrag.retrieval.fusion import reciprocal_rank_fusion


def test_unweighted_rrf_matches_the_formula() -> None:
    fused = dict(reciprocal_rank_fusion({"a": ["x", "y", "z"], "b": ["y", "w"]}, k=60))
    assert fused["x"] == pytest.approx(1 / 61)
    assert fused["y"] == pytest.approx(1 / 62 + 1 / 61)
    assert fused["z"] == pytest.approx(1 / 63)
    assert fused["w"] == pytest.approx(1 / 62)


def test_agreement_beats_a_single_top_rank() -> None:
    ranked = reciprocal_rank_fusion({"dense": ["solo", "both"], "sparse": ["both", "other"]})
    assert [cid for cid, _ in ranked][:2] == ["both", "solo"]


def test_weights_shift_the_balance() -> None:
    rankings = {"dense": ["d1", "d2"], "sparse": ["s1", "s2"]}
    dense_heavy = reciprocal_rank_fusion(rankings, {"dense": 0.7, "sparse": 0.3})
    assert [cid for cid, _ in dense_heavy] == ["d1", "d2", "s1", "s2"]
    assert dict(dense_heavy)["d1"] == pytest.approx(0.7 / 61)
    sparse_heavy = reciprocal_rank_fusion(rankings, {"dense": 0.3, "sparse": 0.7})
    assert sparse_heavy[0][0] == "s1"
    # a zero weight removes a retriever's influence on the order but keeps its items
    dense_only = reciprocal_rank_fusion(rankings, {"dense": 1.0, "sparse": 0.0})
    assert [cid for cid, _ in dense_only][:2] == ["d1", "d2"]


def test_small_k_rewards_top_ranks_more() -> None:
    rankings = {"a": ["p", "q"], "b": ["q", "r"], "c": ["p", "s"]}
    for k in (1, 60):
        fused = dict(reciprocal_rank_fusion(rankings, k=k))
        assert fused["p"] > fused["q"]  # p is first twice; q is first once and second once
    gap_small = dict(reciprocal_rank_fusion({"a": ["p", "q"]}, k=1))
    gap_large = dict(reciprocal_rank_fusion({"a": ["p", "q"]}, k=60))
    assert gap_small["p"] / gap_small["q"] > gap_large["p"] / gap_large["q"]


def test_ties_are_deterministic_and_empty_input_is_fine() -> None:
    tied = reciprocal_rank_fusion({"a": ["m", "n"], "b": ["n", "m"]})
    assert [cid for cid, _ in tied] == ["m", "n"]  # equal score and best rank -> by id
    assert reciprocal_rank_fusion({}) == []
    assert reciprocal_rank_fusion({"a": []}) == []


def test_lopsided_weights_silence_the_weaker_retriever() -> None:
    """Why the default is 0.5/0.5 (ADR-0002): at k=60 the top ranks differ by ~15%."""
    dense = [f"d{i}" for i in range(1, 11)]
    rankings = {"dense": dense, "sparse": ["s1"]}
    lopsided = [c for c, _ in reciprocal_rank_fusion(rankings, {"dense": 0.7, "sparse": 0.3})]
    assert lopsided.index("s1") == 10  # BM25's best hit lands below all ten dense hits
    equal = [c for c, _ in reciprocal_rank_fusion(rankings, {"dense": 0.5, "sparse": 0.5})]
    assert equal.index("s1") == 1  # ties dense #1 on score, loses only the tie-break on id
