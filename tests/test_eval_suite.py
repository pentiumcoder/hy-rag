from __future__ import annotations

import json
from pathlib import Path

import pytest

from hyrag.config import Settings
from hyrag.evaluation import (
    CorrectnessJudge,
    Evidence,
    GoldenQuestion,
    run_suite,
)
from hyrag.evaluation.runner import run_generation
from hyrag.evaluation.suite import generation_configs, offline_settings
from hyrag.generation import AnswerPipeline
from hyrag.models import ChunkStrategy, RetrievalMode
from tests.conftest import CORPUS_DIR, REPO_ROOT

QUESTION = GoldenQuestion(
    id="L2",
    type="lookup",
    question="How many PTO days carry over?",
    answerable=True,
    reference="Up to 5 days, used by March 31.",
    must_include=["5", "March 31"],
    evidence=[Evidence(sources=["hr/pto-policy.md"], quote="up to 5 unused pto days")],
)


def test_generation_configs_add_dense_on_the_best_strategy() -> None:
    summary = [
        {"config": "semantic/hybrid", "strategy": "semantic", "mode": "hybrid", "rerank": False},
        {"config": "fixed/hybrid+rerank", "strategy": "fixed", "mode": "hybrid", "rerank": True},
    ]
    configs = generation_configs(summary)
    assert [c.name for c in configs] == [
        "fixed/hybrid",
        "recursive/hybrid",
        "semantic/hybrid",
        "fixed/dense",
    ]
    assert generation_configs([])[-1].strategy is ChunkStrategy.RECURSIVE


def test_run_generation_needs_the_policy_switched_off(settings: Settings) -> None:
    pipeline = AnswerPipeline.__new__(AnswerPipeline)
    pipeline.settings = settings
    with pytest.raises(ValueError, match="abstain_on_unanswerable"):
        run_generation(pipeline, [], [], CorrectnessJudge(None))


def test_offline_settings_never_touch_the_real_data_dir(tmp_path: Path) -> None:
    s = offline_settings(Settings(_env_file=None), tmp_path)  # type: ignore[call-arg]
    assert (s.embedding_provider, s.reranker, s.llm_provider, s.data_dir) == (
        "fake",
        "none",
        "stub",
        tmp_path,
    )


@pytest.fixture(scope="module")
def offline_run(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("eval")
    settings = Settings(corpus_dir=CORPUS_DIR, _env_file=None)  # type: ignore[call-arg]
    messages: list[str] = []
    golden = REPO_ROOT / "eval" / "golden.jsonl"
    # The first 10 questions plus all unanswerable ones, so every layer has work to do.
    lines = golden.read_text(encoding="utf-8").splitlines()
    subset = out / "golden.jsonl"
    subset.write_text(
        "\n".join(lines[:10] + [x for x in lines if '"unanswerable"' in x]) + "\n", encoding="utf-8"
    )
    summary = run_suite(settings, out, golden=subset, offline=True, progress=messages.append)
    assert any(m.startswith("generate ") for m in messages)
    return out, summary


def test_offline_suite_writes_every_output(offline_run: tuple[Path, dict]) -> None:
    out, summary = offline_run
    for name in ("summary.json", "report.md", "runs/retrieval.jsonl", "runs/generation.jsonl"):
        assert (out / name).exists(), name
    assert sorted(summary["charts"]) == [
        "charts/abstention_tradeoff.png",
        "charts/answer_quality.png",
        "charts/retrieval_recall.png",
    ]
    assert json.loads((out / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["mode"] == "offline"
    assert summary["golden"]["questions"] == 18
    assert summary["models"]["generator"] == "stub:extractive"


def test_offline_suite_scores_all_nine_retrieval_configs(offline_run: tuple[Path, dict]) -> None:
    _, summary = offline_run
    configs = summary["retrieval"]["configs"]
    assert {(c["strategy"], c["mode"]) for c in configs} == {
        (s.value, m.value) for s in ChunkStrategy for m in RetrievalMode
    }
    assert all(not c["rerank"] for c in configs)  # no reranker offline
    assert all(0 <= c["recall_at_5"] <= 1 and c["questions"] == 10 for c in configs)
    sparse = [c["recall_at_5"] for c in configs if c["mode"] == "sparse"]
    assert min(sparse) > 0.5  # BM25 finds most evidence even with a meaningless embedder


def test_offline_suite_tunes_and_reports_abstention(offline_run: tuple[Path, dict]) -> None:
    out, summary = offline_run
    generation = summary["generation"]
    assert len(generation["configs"]) == len(generation["configs_run"]) == 4
    assert set(generation["policy"]) == {"abstain_threshold", "answer_threshold", "judge_check"}
    ablation = generation["abstention"]["ablation"]
    assert [row["name"] for row in ablation][-1] == "tuned"
    assert ablation[-1]["utility"] >= max(row["utility"] for row in ablation[:-1])
    assert set(generation["abstention"]["signal_auroc"]) == {
        "retrieval_confidence",
        "generator_answered",
        "judge_answerability",
        "final_confidence",
    }
    report = (out / "report.md").read_text(encoding="utf-8")
    for heading in ("## Retrieval", "## Answers", "## Abstention", "never abstains"):
        assert heading in report


def test_retrieval_only_run_without_charts(tmp_path: Path) -> None:
    settings = Settings(corpus_dir=CORPUS_DIR, _env_file=None)  # type: ignore[call-arg]
    summary = run_suite(settings, tmp_path, generation=False, offline=True, limit=3, charts=False)
    assert "generation" not in summary and summary["charts"] == []
    assert "## Answers" not in (tmp_path / "report.md").read_text(encoding="utf-8")


def test_a_golden_set_that_does_not_match_the_corpus_is_rejected(tmp_path: Path) -> None:
    bad = QUESTION.model_copy(
        update={"evidence": [Evidence(sources=["hr/pto-policy.md"], quote="unlimited PTO")]}
    )
    path = tmp_path / "golden.jsonl"
    path.write_text(bad.model_dump_json() + "\n", encoding="utf-8")
    settings = Settings(corpus_dir=CORPUS_DIR, _env_file=None)  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="quote not found"):
        run_suite(settings, tmp_path / "out", golden=path, offline=True)
