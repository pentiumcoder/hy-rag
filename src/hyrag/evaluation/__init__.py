"""Evaluation: a golden question set, retrieval and answer metrics, and abstention tuning."""

from hyrag.evaluation.dataset import (
    DEFAULT_GOLDEN,
    Evidence,
    GoldenQuestion,
    check_against_corpus,
    load_golden,
)
from hyrag.evaluation.judge import CorrectnessJudge, Grade, grade_by_key_facts
from hyrag.evaluation.metrics import (
    abstention_scores,
    auroc,
    evidence_ranks,
    key_fact_recall,
    recall_at_k,
    reciprocal_rank,
)
from hyrag.evaluation.policy import GenerationRecord, Policy, evaluate_policy, tune_policy
from hyrag.evaluation.suite import run_suite

__all__ = [
    "DEFAULT_GOLDEN",
    "CorrectnessJudge",
    "Evidence",
    "GenerationRecord",
    "GoldenQuestion",
    "Grade",
    "Policy",
    "abstention_scores",
    "auroc",
    "check_against_corpus",
    "evaluate_policy",
    "evidence_ranks",
    "grade_by_key_facts",
    "key_fact_recall",
    "load_golden",
    "recall_at_k",
    "reciprocal_rank",
    "run_suite",
    "tune_policy",
]
