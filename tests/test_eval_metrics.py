from __future__ import annotations

import math

import pytest

from hyrag.evaluation.dataset import Evidence, GoldenQuestion
from hyrag.evaluation.metrics import (
    abstention_scores,
    auroc,
    evidence_ranks,
    key_fact_recall,
    recall_at_k,
    reciprocal_rank,
)
from tests.conftest import make_chunk

QUESTION = GoldenQuestion(
    id="M1",
    type="multi_hop",
    question="q",
    answerable=True,
    reference="r",
    must_include=["NW-4290", "50|fifty"],
    evidence=[
        Evidence(sources=["pricing.md"], quote="50 requests/second"),
        Evidence(sources=["errors.md", "api.md"], quote="NW-4290 RATE_LIMITED"),
    ],
)


def test_evidence_ranks_find_the_first_matching_chunk_per_item() -> None:
    chunks = [
        make_chunk("unrelated", source="a.md"),
        make_chunk("Starter: 50 requests/second", source="pricing.md"),
        make_chunk("NW-4290 RATE_LIMITED is returned", source="api.md"),
        make_chunk("NW-4290 RATE_LIMITED again", source="errors.md"),
    ]
    assert evidence_ranks(QUESTION, chunks) == [2, 3]
    assert evidence_ranks(QUESTION, chunks[:2]) == [2, None]


def test_recall_and_reciprocal_rank() -> None:
    assert recall_at_k([2, 3], 1) == 0.0
    assert recall_at_k([2, 3], 2) == 0.5
    assert recall_at_k([2, None], 5) == 0.5
    assert recall_at_k([], 5) == 0.0
    assert reciprocal_rank([3, 2]) == 0.5
    assert reciprocal_rank([None, None]) == 0.0


def test_key_fact_recall_accepts_alternatives() -> None:
    assert key_fact_recall("You get NW-4290 at fifty rps.", QUESTION.must_include) == 1.0
    assert key_fact_recall("You get nw-4290.", QUESTION.must_include) == 0.5
    assert key_fact_recall(None, QUESTION.must_include) == 0.0
    assert key_fact_recall("anything", []) == 1.0
    assert key_fact_recall("", []) == 0.0


def test_abstention_scores_treat_abstaining_as_the_positive_class() -> None:
    # (should abstain, did abstain)
    s = abstention_scores([(True, True), (True, False), (False, True), (False, False)])
    assert (s.precision, s.recall, s.abstained, s.unanswerable) == (0.5, 0.5, 2, 2)
    assert s.f1 == pytest.approx(0.5)
    never = abstention_scores([(True, False), (False, False)])
    assert (never.precision, never.recall, never.f1) == (1.0, 0.0, 0.0)
    assert abstention_scores([]).f1 == 1.0


def test_auroc() -> None:
    assert auroc([0.9, 0.8], [0.1, 0.2]) == 1.0
    assert auroc([0.1], [0.9]) == 0.0
    assert auroc([0.5, 0.5], [0.5]) == 0.5
    assert math.isnan(auroc([], [0.1]))
