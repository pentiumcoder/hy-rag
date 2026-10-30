from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from hyrag.chunking import get_chunker
from hyrag.config import Settings
from hyrag.evaluation.dataset import Evidence, GoldenQuestion, check_against_corpus, load_golden
from hyrag.ingestion import load_corpus
from hyrag.models import ChunkStrategy
from tests.conftest import CORPUS_DIR, REPO_ROOT, make_chunk

GOLDEN = REPO_ROOT / "eval" / "golden.jsonl"


def test_golden_set_has_the_planned_mix() -> None:
    questions = load_golden(GOLDEN)
    assert len(questions) == 55
    assert Counter(q.type for q in questions) == {
        "lookup": 20,
        "multi_hop": 12,
        "exact_term": 10,
        "unanswerable": 8,
        "ambiguous": 5,
    }
    assert all(len(q.evidence) >= 2 for q in questions if q.type == "multi_hop")
    assert sum(not q.answerable for q in questions) == 8


def test_every_evidence_quote_is_in_the_corpus() -> None:
    assert check_against_corpus(load_golden(GOLDEN), load_corpus(CORPUS_DIR)) == []


@pytest.mark.parametrize("strategy", list(ChunkStrategy))
def test_every_evidence_quote_survives_chunking(
    strategy: ChunkStrategy, settings: Settings
) -> None:
    """A quote split across two chunks could never be retrieved, under any retriever."""
    chunks = get_chunker(strategy, settings).chunk(load_corpus(CORPUS_DIR))
    for q in load_golden(GOLDEN):
        for e in q.evidence:
            assert any(e.matches(c) for c in chunks), (q.id, e.quote)


def test_evidence_matches_by_source_and_normalized_quote() -> None:
    e = Evidence(sources=["hr/pto.md"], quote="Up to  5 unused\nPTO days")
    assert e.matches(make_chunk("Rule: up to 5 unused pto days carry over.", source="hr/pto.md"))
    assert not e.matches(make_chunk("Up to 5 unused PTO days", source="hr/other.md"))
    assert not e.matches(make_chunk("Up to 6 unused PTO days", source="hr/pto.md"))


def test_questions_must_be_consistent() -> None:
    with pytest.raises(ValidationError, match="needs evidence"):
        GoldenQuestion(id="x", type="lookup", question="q", answerable=True, reference="r")
    with pytest.raises(ValidationError, match="only unanswerable"):
        GoldenQuestion(id="x", type="lookup", question="q", answerable=False, reference="r")


def test_load_golden_rejects_duplicates_and_reports_corpus_problems(tmp_path: Path) -> None:
    row = {
        "id": "Q1",
        "type": "lookup",
        "question": "q",
        "answerable": True,
        "reference": "r",
        "evidence": [{"sources": ["hr/pto-policy.md"], "quote": "not in the policy at all"}],
    }
    path = tmp_path / "golden.jsonl"
    path.write_text(json.dumps(row) + "\n\n" + json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_golden(path)

    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    (q,) = load_golden(path)
    unknown = q.model_copy(update={"evidence": [Evidence(sources=["nope.md"], quote="x y")]})
    problems = check_against_corpus([q, unknown], load_corpus(CORPUS_DIR))
    assert problems == [
        "Q1: quote not found: 'not in the policy at all'",
        "Q1: unknown source(s) nope.md",
    ]
