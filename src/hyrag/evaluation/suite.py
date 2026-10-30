"""The whole evaluation: retrieval matrix, generation on selected configs, tuned abstention."""

from __future__ import annotations

import tempfile
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from hyrag import __version__
from hyrag.config import Settings
from hyrag.embeddings import embedder_from_settings
from hyrag.evaluation.dataset import (
    DEFAULT_GOLDEN,
    GoldenQuestion,
    check_against_corpus,
    load_golden,
)
from hyrag.evaluation.judge import CorrectnessJudge
from hyrag.evaluation.policy import WRONG_ANSWER_COST, ablation, tune_policy
from hyrag.evaluation.report import (
    abstention_section,
    summarize_generation,
    summarize_retrieval,
    write_outputs,
)
from hyrag.evaluation.runner import (
    GenerationConfig,
    Progress,
    _quiet,
    run_generation,
    run_retrieval,
    sync_indexes,
)
from hyrag.generation import AnswerPipeline
from hyrag.ingestion import load_corpus
from hyrag.llm import default_model
from hyrag.models import ChunkStrategy, RetrievalMode
from hyrag.retrieval import HybridRetriever, get_reranker


def offline_settings(settings: Settings, data_dir: Path) -> Settings:
    """Fake embedder, no reranker and the extractive stub, in a scratch data directory.

    The scratch directory matters: syncing the real index with the fake embedder would
    rebuild it with fake vectors.
    """
    return settings.model_copy(
        update={
            "embedding_provider": "fake",
            "reranker": "none",
            "llm_provider": "stub",
            "judge_provider": None,
            "llm_cache": False,
            "data_dir": data_dir,
        }
    )


def generation_configs(retrieval_summary: Sequence[dict[str, Any]]) -> list[GenerationConfig]:
    """Hybrid on every strategy, plus dense-only on the strategy with the best hybrid recall."""
    hybrid = [r for r in retrieval_summary if r["mode"] == "hybrid"]
    reranked = [r for r in hybrid if r["rerank"]] or hybrid
    best = ChunkStrategy(reranked[0]["strategy"]) if reranked else ChunkStrategy.RECURSIVE
    configs = [GenerationConfig(s, RetrievalMode.HYBRID) for s in ChunkStrategy]
    return [*configs, GenerationConfig(best, RetrievalMode.DENSE)]


def _models(settings: Settings, offline: bool) -> dict[str, str | None]:
    if offline:
        return {
            "embedder": "fake",
            "reranker": None,
            "generator": "stub:extractive",
            "judge": "stub:extractive",
        }
    provider = settings.llm_provider
    judge_provider = settings.judge_provider or provider
    stub = "stub:extractive"
    return {
        "embedder": settings.embedding_model if settings.embedding_provider != "fake" else "fake",
        "reranker": settings.reranker_model if settings.reranker != "none" else None,
        "generator": stub
        if provider == "stub"
        else settings.generator_model or default_model(provider, "generator"),
        "judge": stub
        if judge_provider == "stub"
        else settings.judge_model or default_model(judge_provider, "judge"),
    }


def run_suite(
    settings: Settings,
    out_dir: Path,
    *,
    golden: Path = DEFAULT_GOLDEN,
    generation: bool = True,
    offline: bool = False,
    limit: int | None = None,
    charts: bool = True,
    progress: Progress = _quiet,
) -> dict[str, Any]:
    questions: list[GoldenQuestion] = load_golden(golden)[:limit]
    problems = check_against_corpus(questions, load_corpus(settings.corpus_dir))
    if problems:
        raise ValueError("golden set does not match the corpus:\n" + "\n".join(problems))

    scratch = (
        tempfile.TemporaryDirectory(prefix="hyrag-eval-", ignore_cleanup_errors=True)
        if offline
        else None
    )
    try:
        if scratch is not None:
            settings = offline_settings(settings, Path(scratch.name))
        embedder = embedder_from_settings(settings)
        reranker = get_reranker(
            settings.reranker, settings.reranker_model, settings.model_cache_dir
        )
        progress("syncing indexes")
        indexes = sync_indexes(settings, list(ChunkStrategy), embedder)
        rows = run_retrieval(settings, questions, indexes, embedder, reranker, progress=progress)
        retrieval_summary = summarize_retrieval(rows)
        summary: dict[str, Any] = {
            "version": __version__,
            "generated_at": datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC"),
            "mode": "offline" if offline else "local models",
            "golden": {
                "path": Path(golden).as_posix(),
                "questions": len(questions),
                "by_type": dict(Counter(q.type for q in questions)),
            },
            "models": _models(settings, offline),
            "retrieval": {
                "k": 5,
                "configs": retrieval_summary,
                "best": retrieval_summary[0]["config"] if retrieval_summary else None,
            },
        }
        records = []
        if generation:
            eval_settings = settings.model_copy(update={"abstain_on_unanswerable": False})
            configs = generation_configs(retrieval_summary)
            retriever = HybridRetriever(
                indexes[configs[0].strategy], embedder, reranker, eval_settings
            )
            pipeline = AnswerPipeline.from_settings(eval_settings, retriever=retriever)
            grader = CorrectnessJudge(None if offline else pipeline.judge)
            records = run_generation(pipeline, questions, configs, grader, progress=progress)
            ranked = tune_policy(records)
            tuned = ranked[0]
            summary["generation"] = {
                "configs_run": [c.name for c in configs],
                "policy": asdict(tuned.policy),
                "configs": summarize_generation(records, tuned.policy),
                "abstention": abstention_section(
                    records, tuned, ablation(records, tuned.policy), WRONG_ANSWER_COST
                ),
            }
        write_outputs(out_dir, summary, rows, records, charts=charts)
        return summary
    finally:
        if scratch is not None:
            scratch.cleanup()
