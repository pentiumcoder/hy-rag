from __future__ import annotations

import json

from hyrag.generation.claims import extract_claims
from hyrag.generation.prompts import INSUFFICIENT, build_answer_messages
from hyrag.generation.stub import ExtractiveStubLLM
from hyrag.generation.verifier import CitationVerifier, label_claims
from hyrag.index.tokenize import key_terms
from hyrag.llm import LLMError, user
from hyrag.llm.stub import ScriptedLLM
from hyrag.models import Claim, ScoredChunk
from tests.conftest import make_chunk

PASSAGES = [
    "Up to 5 unused PTO days can be carried over into the next calendar year. Carried-over "
    "days must be used by March 31 or they expire.",
    "Webhooks carry an X-NW-Signature header: an HMAC-SHA256 of the timestamp and the body.",
    "The Growth plan costs $499 per month and includes 50 million events.",
]
QUESTION = "How many PTO days carry over?"


def _contexts() -> list[ScoredChunk]:
    return [
        ScoredChunk(chunk=make_chunk(text, source=f"doc{i}.md", index=i), rank=i, score=1.0)
        for i, text in enumerate(PASSAGES, start=1)
    ]


def _judge_reply(*verdicts: tuple[int, str], completeness: str = "COMPLETE") -> str:
    checks = [{"id": i, "label": label, "reason": f"r{i}"} for i, label in verdicts]
    return json.dumps({"checks": checks, "completeness": completeness})


def test_all_pairs_go_to_the_judge_in_one_call() -> None:
    answer = "Up to 5 days carry over [1]. They expire on March 31 [1][3]."
    judge = ScriptedLLM([_judge_reply((1, "SUPPORTED"), (2, "SUPPORTED"), (3, "UNSUPPORTED"))])
    result = CitationVerifier(judge).verify(QUESTION, answer, extract_claims(answer), _contexts())

    assert len(judge.calls) == 1 and judge.calls[0].json_mode
    assert [(c.claim, c.citation, c.label) for c in result.checks] == [
        (1, 1, "SUPPORTED"),
        (2, 1, "SUPPORTED"),
        (2, 3, "UNSUPPORTED"),
    ]
    assert result.completeness == 1.0 and result.error is None


def test_citations_of_missing_passages_are_rejected_without_the_judge() -> None:
    answer = "Up to 5 days carry over [7]."
    judge = ScriptedLLM()
    result = CitationVerifier(judge).verify(QUESTION, answer, extract_claims(answer), _contexts())
    assert judge.calls == []
    (check,) = result.checks
    assert (check.citation, check.label, check.reason) == (
        7,
        "UNSUPPORTED",
        "there is no passage [7]",
    )
    assert result.completeness is None


def test_label_aliases_missing_verdicts_and_bad_ids() -> None:
    answer = "Five days [1]. March 31 [1]. HMAC [2]."
    reply = json.dumps(
        {
            "checks": [
                {"id": 1, "label": "not supported"},
                {"id": "2", "label": "Partially-Supported", "reason": "half"},
                {"id": "x", "label": "SUPPORTED"},
                {"id": 4, "label": "MAYBE"},
                "junk",
            ],
            "completeness": "partial",
        }
    )
    result = CitationVerifier(ScriptedLLM([reply])).verify(
        QUESTION, answer, extract_claims(answer), _contexts()
    )
    assert [(c.label, c.reason) for c in result.checks] == [
        ("UNSUPPORTED", ""),
        ("PARTIAL", "half"),
        ("UNVERIFIED", "the judge gave no verdict"),
    ]
    assert result.completeness == 0.5


def test_a_failing_judge_leaves_citations_unverified() -> None:
    answer = "Up to 5 days carry over [1]."
    result = CitationVerifier(ScriptedLLM([LLMError("timeout")])).verify(
        QUESTION, answer, extract_claims(answer), _contexts()
    )
    assert result.error == "timeout"
    assert [(c.label, c.reason) for c in result.checks] == [("UNVERIFIED", "verification failed")]


def test_without_a_judge_only_the_free_checks_run() -> None:
    answer = "Up to 5 days carry over [1][9]."
    result = CitationVerifier(None).verify(QUESTION, answer, extract_claims(answer), _contexts())
    assert [c.label for c in result.checks] == ["UNVERIFIED", "UNSUPPORTED"]
    assert result.completeness is None


def test_label_claims_takes_the_best_verdict_per_claim() -> None:
    answer = "Five days [1][3]. Nothing cited here at all. The context does not cover X."
    claims = extract_claims(answer)
    judge = ScriptedLLM([_judge_reply((1, "UNSUPPORTED"), (2, "PARTIAL"))])
    checks = CitationVerifier(judge).verify(QUESTION, answer, claims, _contexts()).checks
    labelled = label_claims(claims, checks)
    assert [c.label for c in labelled] == ["PARTIAL", "UNCITED", None]
    assert all(c.label is None for c in claims)  # the input is not mutated


def test_the_stub_judge_catches_a_planted_hallucination() -> None:
    """A fluent, cited, wrong sentence: the kind of error a reader would not spot."""
    answer = (
        "Up to 5 unused PTO days can be carried over into the next year [1]. "
        "Carried-over days expire on June 30 [1]. "
        "Enterprise customers get unlimited free GPUs [3]."
    )
    claims = extract_claims(answer)
    result = CitationVerifier(ExtractiveStubLLM()).verify(QUESTION, answer, claims, _contexts())
    labels = {c.claim: c.label for c in result.checks}
    assert labels[1] == "SUPPORTED"
    assert labels[2] == "UNSUPPORTED"  # the date is not in the passage
    assert labels[3] == "UNSUPPORTED"  # the passage is about something else
    assert result.completeness == 1.0


def test_stub_generator_answers_extractively_with_citations() -> None:
    stub = ExtractiveStubLLM()
    reply = stub.complete(build_answer_messages(QUESTION, _contexts()))
    assert reply.startswith("Up to 5 unused PTO days can be carried over")
    assert "[1]." in reply and "[2]" not in reply
    claims = extract_claims(reply)
    assert claims and all(c.citations == [1] for c in claims)


def test_stub_generator_abstains_when_nothing_overlaps() -> None:
    reply = ExtractiveStubLLM().complete(
        build_answer_messages("Who is the CEO's favourite painter?", _contexts())
    )
    assert reply.startswith(INSUFFICIENT)
    assert "ceo" in reply and "painter" in reply


def test_stub_ignores_prompts_it_does_not_know() -> None:
    stub = ExtractiveStubLLM()
    assert stub.complete([user("hello")]) == ""
    assert stub.complete([user("hello")], json_mode=True) == "{}"
    assert stub.complete([]) == ""


def test_key_terms_keep_identifiers_whole_and_drop_filler() -> None:
    assert key_terms("How many PTO days does NW-4290 cost? Tell me about the CEO's PTO") == [
        "pto",
        "days",
        "nw-4290",
        "cost",
        "ceo",
    ]


def test_unknown_claim_kinds_are_skipped() -> None:
    gap = Claim(index=1, text="The context does not cover it.", citations=[1], kind="gap")
    result = CitationVerifier(ScriptedLLM()).verify(QUESTION, "x", [gap], _contexts())
    assert result.checks == []


def test_the_judge_sees_every_retrieved_passage_and_rates_answerability() -> None:
    answer = "Up to 5 days carry over [1]."
    reply = json.dumps(
        {
            "checks": [{"id": 1, "label": "SUPPORTED"}],
            "completeness": "COMPLETE",
            "answerable": "partial",
        }
    )
    judge = ScriptedLLM([reply])
    result = CitationVerifier(judge).verify(QUESTION, answer, extract_claims(answer), _contexts())
    assert result.answerability == 0.5
    prompt = judge.calls[0].messages[1]["content"]
    assert all(f'<passage id="{n}"' in prompt for n in (1, 2, 3))  # uncited [2] and [3] too

    unknown = CitationVerifier(ScriptedLLM([_judge_reply((1, "SUPPORTED"))])).verify(
        QUESTION, answer, extract_claims(answer), _contexts()
    )
    assert unknown.answerability is None


def test_the_stub_judge_rates_answerability_by_question_coverage() -> None:
    def answerability(question: str) -> float | None:
        answer = "Up to 5 unused PTO days can be carried over [1]."
        return (
            CitationVerifier(ExtractiveStubLLM())
            .verify(question, answer, extract_claims(answer), _contexts())
            .answerability
        )

    assert answerability(QUESTION) == 1.0
    assert answerability("Which dentist covers the Mars office?") == 0.0
