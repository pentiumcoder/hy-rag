from __future__ import annotations

import pytest

from hyrag.evaluation.policy import (
    WRONG_ANSWER_COST,
    GenerationRecord,
    Policy,
    ablation,
    evaluate_policy,
    tune_policy,
)


def record(
    qid: str,
    answerable: bool = True,
    retrieval: float = 0.9,
    generator_abstained: bool = False,
    answerability: float | None = 1.0,
    any_supported: bool = True,
    confidence: float = 0.9,
    correctness: float | None = 1.0,
) -> GenerationRecord:
    return GenerationRecord(
        config="recursive/hybrid",
        qid=qid,
        type="lookup" if answerable else "unanswerable",
        answerable=answerable,
        retrieval=retrieval,
        generator_abstained=generator_abstained,
        answerability=answerability,
        any_supported=any_supported,
        confidence=confidence,
        correctness=correctness if answerable else None,
    )


RECORDS = [
    record("good"),
    record("partial", correctness=0.5, confidence=0.7),
    record("wrong", correctness=0.0, confidence=0.35),
    record("off_topic", answerable=False, retrieval=0.01, confidence=0.1),
    record("declined", answerable=False, generator_abstained=True, retrieval=0.5),
    # A near-miss: fluent answer, retrieval looks fine, but the judge sees no answer in the
    # passages and nothing is supported. This is the case retrieval scores alone miss.
    record(
        "near_miss",
        answerable=False,
        retrieval=0.6,
        answerability=0.0,
        any_supported=False,
        confidence=0.45,
    ),
]


def test_layers_fire_in_order() -> None:
    p = Policy(abstain_threshold=0.2, answer_threshold=0.4, judge_check=True)
    decided = {r.qid: p.decide(r) for r in RECORDS}
    assert decided == {
        "good": None,
        "partial": None,
        "wrong": "confidence_floor",
        "off_topic": "retrieval_gate",
        "declined": "generator",
        "near_miss": "judge",
    }


def test_evaluate_policy_counts_correct_and_wrong_answers() -> None:
    day3 = evaluate_policy(Policy(0.2, 0.0, False), RECORDS)
    assert day3.by_layer == {"retrieval_gate": 1, "generator": 1}
    assert day3.answered == 3
    assert day3.correct == 1.5
    assert day3.wrong == 2.0  # "wrong", plus the near-miss answered
    assert day3.abstained == 2
    assert (day3.abstention_precision, day3.abstention_recall) == (1.0, pytest.approx(2 / 3))
    assert day3.utility == pytest.approx((1.5 - WRONG_ANSWER_COST * 2.0) / len(RECORDS))

    with_judge = evaluate_policy(Policy(0.2, 0.0, True), RECORDS)
    assert with_judge.abstention_recall == 1.0 and with_judge.utility > day3.utility


def test_tuning_finds_the_floor_that_drops_the_wrong_answer() -> None:
    best = tune_policy(RECORDS)[0]
    assert best.policy.judge_check
    assert 0.35 < best.policy.answer_threshold <= 0.7
    assert best.policy.decide(RECORDS[2]) is not None  # "wrong" is withheld
    assert best.policy.decide(RECORDS[0]) is None  # "good" is still answered
    assert best.utility == max(r.utility for r in tune_policy(RECORDS))


def test_ablation_walks_from_no_abstention_to_the_tuned_policy() -> None:
    tuned = tune_policy(RECORDS)[0].policy
    rows = ablation(RECORDS, tuned)
    names = [name for name, _ in rows]
    assert names[0].startswith("no abstention") and names[-1] == "tuned"
    none = rows[0][1]
    assert none.abstained == 0 and none.wrong == 1 + 3  # every unanswerable is answered
    utilities = [r.utility for _, r in rows]
    assert utilities[-1] == max(utilities)
    assert rows[-1][1].as_dict()["policy"] == {
        "abstain_threshold": tuned.abstain_threshold,
        "answer_threshold": tuned.answer_threshold,
        "judge_check": tuned.judge_check,
    }


def test_the_gate_sits_in_the_middle_of_the_equally_good_range() -> None:
    """Any gate between the off-topic score (0.01) and the answerable ones (0.9) is equally
    good on this data; the tuner picks the middle rather than the edge."""
    best = tune_policy(RECORDS)[0].policy
    assert 0.4 <= best.abstain_threshold <= 0.5
    assert best.decide(RECORDS[3]) == "retrieval_gate"  # off-topic never reaches the generator
