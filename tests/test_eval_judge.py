from __future__ import annotations

from hyrag.evaluation.dataset import Evidence, GoldenQuestion
from hyrag.evaluation.judge import CorrectnessJudge, grade_by_key_facts
from hyrag.llm import LLMError
from hyrag.llm.stub import ScriptedLLM

QUESTION = GoldenQuestion(
    id="L2",
    type="lookup",
    question="How many PTO days carry over?",
    answerable=True,
    reference="Up to 5 days, used by March 31.",
    must_include=["5", "March 31"],
    evidence=[Evidence(sources=["hr/pto-policy.md"], quote="up to 5 unused pto days")],
)


def test_key_fact_grading() -> None:
    assert grade_by_key_facts(QUESTION, "Up to 5 days, by March 31 [1].").verdict == "CORRECT"
    assert grade_by_key_facts(QUESTION, "Up to 5 days [1].").verdict == "PARTIAL"
    grade = grade_by_key_facts(QUESTION, "Ten days.")
    assert (grade.verdict, grade.score, grade.source) == ("INCORRECT", 0.0, "key_facts")


def test_llm_grading_and_its_fallbacks() -> None:
    llm = ScriptedLLM(['{"verdict": "partial", "reason": "misses the expiry"}'])
    grade = CorrectnessJudge(llm).grade(QUESTION, "Up to 5 days.")
    assert (grade.verdict, grade.reason, grade.source, grade.score) == (
        "PARTIAL",
        "misses the expiry",
        "judge",
        0.5,
    )
    prompt = llm.calls[0].messages[1]["content"]
    assert "Reference answer: Up to 5 days, used by March 31." in prompt

    bad_label = CorrectnessJudge(ScriptedLLM(['{"verdict": "MAYBE"}'])).grade(
        QUESTION, "5, March 31"
    )
    failing = CorrectnessJudge(ScriptedLLM([LLMError("down"), LLMError("down")]))
    for grade in (bad_label, failing.grade(QUESTION, "5, March 31")):
        assert (grade.verdict, grade.source) == ("CORRECT", "key_facts")
    assert CorrectnessJudge(None).grade(QUESTION, "nothing").verdict == "INCORRECT"
