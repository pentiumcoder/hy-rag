"""Abstention policies, replayed offline over recorded eval runs.

The eval generates and verifies every question with all abstention rules switched off and
records each signal. Any policy can then be scored without calling a model again, so the
thresholds are tuned on data instead of picked by hand (ADR-0004).

A policy abstains at the first layer that fires:

1. **retrieval gate:** retrieval confidence below ``abstain_threshold``; nothing is generated;
2. **generator:** the generator itself replied that the context is insufficient;
3. **judge:** the judge says the passages do not answer the question and no sentence of the
   answer is supported;
4. **confidence floor:** the verified confidence is below ``answer_threshold``.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass

from pydantic import BaseModel

from hyrag.evaluation.metrics import abstention_scores

# A wrong answer shown as if it were right costs more than an honest "I don't know".
WRONG_ANSWER_COST = 2.0


class GenerationRecord(BaseModel):
    """Everything the policy replay and the report need about one question in one config."""

    config: str
    qid: str
    type: str
    answerable: bool
    retrieval: float
    top_rerank: float | None = None
    generator_abstained: bool
    abstain_reason: str | None = None
    confidence: float = 0.0
    citation_coverage: float = 0.0
    completeness: float = 0.0
    answerability: float | None = None
    any_supported: bool = False
    claims: int = 0
    checks: dict[str, int] = {}
    answer: str | None = None
    correctness: float | None = None  # judge score for answered questions
    verdict: str | None = None
    grade_source: str | None = None
    key_fact_recall: float | None = None
    evidence_recall: float = 0.0  # share of gold evidence among the passages it was given
    generate_ms: float = 0.0
    verify_ms: float = 0.0


@dataclass(frozen=True)
class Policy:
    abstain_threshold: float = 0.2
    answer_threshold: float = 0.0
    judge_check: bool = True

    def decide(self, r: GenerationRecord) -> str | None:
        """The layer that abstains on this record, or None if the answer is shown."""
        if r.retrieval < self.abstain_threshold:
            return "retrieval_gate"
        if r.generator_abstained:
            return "generator"
        if self.judge_check and r.answerability == 0.0 and not r.any_supported:
            return "judge"
        if r.confidence < self.answer_threshold:
            return "confidence_floor"
        return None


@dataclass(frozen=True)
class PolicyResult:
    policy: Policy
    abstention_precision: float
    abstention_recall: float
    abstention_f1: float
    abstained: int
    answered: int  # answerable questions answered
    correct: float  # sum of correctness over answered answerable questions
    wrong: float  # incorrect answers, plus every answered unanswerable question
    utility: float  # correct - WRONG_ANSWER_COST * wrong, per question
    by_layer: dict[str, int]

    def as_dict(self) -> dict[str, object]:
        data = asdict(self)
        data["policy"] = asdict(self.policy)
        return data


def evaluate_policy(policy: Policy, records: Sequence[GenerationRecord]) -> PolicyResult:
    outcomes: list[tuple[bool, bool]] = []
    by_layer: dict[str, int] = {}
    answered = 0
    correct = wrong = 0.0
    for r in records:
        layer = policy.decide(r)
        outcomes.append((not r.answerable, layer is not None))
        if layer is not None:
            by_layer[layer] = by_layer.get(layer, 0) + 1
            continue
        if not r.answerable:
            wrong += 1.0
            continue
        answered += 1
        score = r.correctness if r.correctness is not None else 0.0
        correct += score
        wrong += 1.0 if score == 0.0 else 0.0  # partly right is incomplete, not wrong
    scores = abstention_scores(outcomes)
    n = len(records) or 1
    return PolicyResult(
        policy=policy,
        abstention_precision=scores.precision,
        abstention_recall=scores.recall,
        abstention_f1=scores.f1,
        abstained=scores.abstained,
        answered=answered,
        correct=correct,
        wrong=wrong,
        utility=(correct - WRONG_ANSWER_COST * wrong) / n,
        by_layer=by_layer,
    )


def _grid(start: float, stop: float, step: float) -> list[float]:
    count = round((stop - start) / step)
    return [round(start + i * step, 4) for i in range(count + 1)]


def tune_policy(
    records: Sequence[GenerationRecord],
    abstain_grid: Iterable[float] | None = None,
    answer_grid: Iterable[float] | None = None,
) -> list[PolicyResult]:
    """Every policy on the grid, best first.

    Ranked by utility, then abstention F1, then the lower confidence floor (withhold no
    answer the data does not ask to withhold), then with the judge check on. Among the
    equally good retrieval gates, the middle of that range comes first: gating saves a
    generation call per off-topic question, and the midpoint keeps the widest margin on
    both sides instead of sitting on the edge of what the 55 questions happened to show.
    """
    results = [
        evaluate_policy(Policy(a, b, judge), records)
        for a in (abstain_grid if abstain_grid is not None else _grid(0.0, 0.9, 0.05))
        for b in (answer_grid if answer_grid is not None else _grid(0.0, 0.8, 0.05))
        for judge in (True, False)
    ]
    results.sort(
        key=lambda r: (
            -r.utility,
            -r.abstention_f1,
            r.policy.answer_threshold,
            not r.policy.judge_check,
            r.policy.abstain_threshold,
        )
    )
    best = results[0]
    plateau = [
        r
        for r in results
        if math.isclose(r.utility, best.utility)
        and math.isclose(r.abstention_f1, best.abstention_f1)
        and r.policy.answer_threshold == best.policy.answer_threshold
        and r.policy.judge_check == best.policy.judge_check
    ]
    gates = [r.policy.abstain_threshold for r in plateau]
    middle = (min(gates) + max(gates)) / 2
    chosen = min(
        plateau,
        key=lambda r: (abs(r.policy.abstain_threshold - middle), r.policy.abstain_threshold),
    )
    results.remove(chosen)
    return [chosen, *results]


def ablation(records: Sequence[GenerationRecord], tuned: Policy) -> list[tuple[str, PolicyResult]]:
    """What each layer adds, from no abstention at all up to the tuned policy."""
    day3 = Policy(abstain_threshold=0.2, answer_threshold=0.0, judge_check=False)
    steps = [
        ("no abstention (generator rule ignored)", None),
        ("Day 3: retrieval gate 0.2 + generator", day3),
        ("+ judge answerability check", Policy(0.2, 0.0, True)),
        ("tuned", tuned),
    ]
    out: list[tuple[str, PolicyResult]] = []
    for name, policy in steps:
        if policy is None:
            # Answer everything, even where the generator declined (those count as wrong).
            forced = [r.model_copy(update={"generator_abstained": False}) for r in records]
            out.append((name, evaluate_policy(Policy(0.0, 0.0, False), forced)))
        else:
            out.append((name, evaluate_policy(policy, records)))
    return out
