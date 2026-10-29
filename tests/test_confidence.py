from __future__ import annotations

import math

import pytest

from hyrag.generation.abstention import build_abstention, suggested_documents
from hyrag.generation.confidence import (
    WEIGHTS,
    citation_coverage,
    combine,
    lexical_coverage,
    retrieval_confidence,
)
from hyrag.models import Claim, ScoredChunk
from tests.conftest import make_chunk


def _scored(
    text: str, source: str = "a.md", rerank: float | None = None, rank: int = 1
) -> ScoredChunk:
    return ScoredChunk(
        chunk=make_chunk(text, source=source, index=rank),
        rank=rank,
        score=rerank if rerank is not None else 0.5,
        rerank_score=rerank,
    )


def test_weights_follow_the_documented_formula() -> None:
    assert WEIGHTS == {"retrieval": 0.4, "citation_coverage": 0.4, "completeness": 0.2}
    conf = combine(0.9, 0.5, 1.0, "rerank")
    assert conf.score == pytest.approx(0.4 * 0.9 + 0.4 * 0.5 + 0.2 * 1.0)
    assert conf.retrieval_signal == "rerank"


@pytest.mark.parametrize(
    ("logit", "expected"),
    [(-4.0, 0.5), (0.0, 1 / (1 + math.exp(-2))), (-11.0, 1 / (1 + math.exp(3.5)))],
)
def test_rerank_logits_map_through_a_shifted_sigmoid(logit: float, expected: float) -> None:
    contexts = [_scored("x", rerank=logit), _scored("y", rerank=logit - 5, rank=2)]
    value, signal = retrieval_confidence("q", contexts, center=-4.0, scale=2.0)
    assert signal == "rerank"
    assert value == pytest.approx(expected)


def test_the_best_rerank_score_wins_even_if_not_ranked_first() -> None:
    contexts = [_scored("x", rerank=-9.0), _scored("y", rerank=3.0, rank=2)]
    assert retrieval_confidence("q", contexts)[0] > 0.9


def test_without_rerank_scores_confidence_is_key_term_coverage() -> None:
    contexts = [
        _scored("PTO days carry over until March."),
        _scored("Webhooks are signed.", rank=2),
    ]
    assert lexical_coverage("How many PTO days carry over to Mars?", contexts) == (
        ["pto", "days", "carry", "over"],
        ["mars"],
    )
    value, signal = retrieval_confidence("How many PTO days carry over to Mars?", contexts)
    assert (value, signal) == (pytest.approx(0.8), "lexical")
    assert retrieval_confidence("the of and", contexts) == (0.0, "lexical")
    assert retrieval_confidence("anything", []) == (0.0, "none")


def test_citation_coverage_scores_facts_only() -> None:
    claims = [
        Claim(index=1, text="a", citations=[1], label="SUPPORTED"),
        Claim(index=2, text="b", citations=[1], label="PARTIAL"),
        Claim(index=3, text="c", citations=[2], label="UNSUPPORTED"),
        Claim(index=4, text="d", label="UNCITED"),
        Claim(index=5, text="e", citations=[1], label="UNVERIFIED"),
        Claim(index=6, text="The context does not cover f.", kind="gap"),
    ]
    assert citation_coverage(claims) == pytest.approx((1 + 0.5 + 0 + 0 + 0.5) / 5)
    assert citation_coverage(claims[-1:]) == 0.0
    assert citation_coverage([]) == 0.0


def test_suggested_documents_are_distinct_and_capped() -> None:
    contexts = [
        _scored("one", source="hr/pto.md", rerank=2.0, rank=1),
        _scored("two", source="hr/pto.md", rerank=1.0, rank=2),
        _scored("three", source="hr/faq.txt", rerank=0.5, rank=3),
        _scored("four", source="api/hooks.html", rerank=0.1, rank=4),
        _scored("five", source="ops/oncall.md", rerank=0.0, rank=5),
    ]
    refs = suggested_documents(contexts)
    assert [r.source for r in refs] == ["hr/pto.md", "hr/faq.txt", "api/hooks.html"]
    assert refs[0].score == 2.0


def test_abstention_falls_back_to_term_overlap() -> None:
    contexts = [_scored("The Growth plan costs $499 per month.", source="pricing.md")]
    a = build_abstention(
        "low_retrieval_confidence",
        "too weak",
        "Does the Growth plan have a student discount?",
        contexts,
    )
    assert a.reason == "low_retrieval_confidence" and a.message == "too weak"
    assert a.found == ["Passages that mention: growth, plan."]
    assert a.missing == ["No retrieved passage mentions: student, discount."]
    assert [d.source for d in a.suggested_documents] == ["pricing.md"]


def test_abstention_prefers_the_generators_own_explanation() -> None:
    contexts = [_scored("The Growth plan costs $499 per month.")]
    a = build_abstention(
        "insufficient_context",
        "generator abstained",
        "How much is the Growth plan?",
        contexts,
        found=["Growth plan pricing."],
        missing=["Nothing."],
    )
    assert (a.found, a.missing) == (["Growth plan pricing."], ["Nothing."])
    covered = build_abstention("no_results", "none", "growth plan", contexts)
    assert covered.missing == ["No retrieved passage answers the question directly."]
    assert build_abstention("no_results", "none", "growth", []).suggested_documents == []
