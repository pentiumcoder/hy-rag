"""Running the golden set through retrieval and the answer pipeline.

Retrieval is cheap and deterministic, so every strategy x mode (x rerank) configuration is
scored. Generation is slow on local models, so only a few configurations run, and each is
done in phases: generate every answer while the generator is loaded, then verify and grade
them all while the judge is loaded. With 8 GB of VRAM that turns ~2 model swaps per question
into 2 per configuration.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from pydantic import BaseModel

from hyrag.config import Settings
from hyrag.embeddings import Embedder
from hyrag.evaluation.dataset import GoldenQuestion
from hyrag.evaluation.judge import CorrectnessJudge
from hyrag.evaluation.metrics import evidence_ranks, key_fact_recall, recall_at_k, reciprocal_rank
from hyrag.evaluation.policy import GenerationRecord
from hyrag.generation import AnswerPipeline
from hyrag.index import IndexManager
from hyrag.ingestion import load_corpus
from hyrag.models import AnswerResponse, AskOverrides, ChunkStrategy, RetrievalMode
from hyrag.retrieval import HybridRetriever, Reranker

Progress = Callable[[str], None]


def _quiet(_: str) -> None:
    return None


@dataclass(frozen=True)
class RetrievalConfig:
    strategy: ChunkStrategy
    mode: RetrievalMode
    rerank: bool

    @property
    def name(self) -> str:
        return f"{self.strategy.value}/{self.mode.value}" + ("+rerank" if self.rerank else "")


@dataclass(frozen=True)
class GenerationConfig:
    strategy: ChunkStrategy
    mode: RetrievalMode

    @property
    def name(self) -> str:
        return f"{self.strategy.value}/{self.mode.value}"


class RetrievalRow(BaseModel):
    config: str
    strategy: ChunkStrategy
    mode: RetrievalMode
    rerank: bool
    qid: str
    type: str
    answerable: bool
    ranks: list[int | None]
    recall_at_1: float
    recall_at_5: float
    reciprocal_rank: float
    top_rerank: float | None = None


def sync_indexes(
    settings: Settings, strategies: Sequence[ChunkStrategy], embedder: Embedder
) -> dict[ChunkStrategy, IndexManager]:
    """Bring each strategy's index up to date with the corpus (a no-op when it already is)."""
    docs = load_corpus(settings.corpus_dir)
    indexes: dict[ChunkStrategy, IndexManager] = {}
    for strategy in strategies:
        index = IndexManager(settings, strategy, embedder)
        index.sync(docs)
        indexes[strategy] = index
    return indexes


def run_retrieval(
    settings: Settings,
    questions: Sequence[GoldenQuestion],
    indexes: dict[ChunkStrategy, IndexManager],
    embedder: Embedder,
    reranker: Reranker | None,
    *,
    modes: Sequence[RetrievalMode] = tuple(RetrievalMode),
    depth: int = 10,
    progress: Progress = _quiet,
) -> list[RetrievalRow]:
    rerank_options = (True, False) if reranker is not None else (False,)
    rows: list[RetrievalRow] = []
    for strategy, index in indexes.items():
        retriever = HybridRetriever(index, embedder, reranker, settings)
        for mode in modes:
            for rerank in rerank_options:
                config = RetrievalConfig(strategy, mode, rerank)
                progress(f"retrieval {config.name}")
                for q in questions:
                    results = retriever.retrieve(q.question, mode, k=depth, rerank=rerank)
                    ranks = evidence_ranks(q, [r.chunk for r in results])
                    reranked = [r.rerank_score for r in results if r.rerank_score is not None]
                    rows.append(
                        RetrievalRow(
                            config=config.name,
                            strategy=strategy,
                            mode=mode,
                            rerank=rerank,
                            qid=q.id,
                            type=q.type,
                            answerable=q.answerable,
                            ranks=ranks,
                            recall_at_1=recall_at_k(ranks, 1),
                            recall_at_5=recall_at_k(ranks, 5),
                            reciprocal_rank=reciprocal_rank(ranks),
                            top_rerank=max(reranked) if reranked else None,
                        )
                    )
    return rows


def _record(
    config: GenerationConfig,
    q: GoldenQuestion,
    draft: AnswerResponse,
    final: AnswerResponse,
) -> GenerationRecord:
    reranked = [c.rerank_score for c in draft.contexts if c.rerank_score is not None]
    ranks = evidence_ranks(q, [c.chunk for c in draft.contexts])
    return GenerationRecord(
        config=config.name,
        qid=q.id,
        type=q.type,
        answerable=q.answerable,
        retrieval=draft.confidence.retrieval,
        top_rerank=max(reranked) if reranked else None,
        generator_abstained=draft.abstained,
        abstain_reason=draft.abstention.reason if draft.abstention else None,
        confidence=final.confidence.score,
        citation_coverage=final.confidence.citation_coverage,
        completeness=final.confidence.completeness,
        answerability=final.confidence.answerability,
        any_supported=any(c.label == "SUPPORTED" for c in final.claims),
        claims=sum(1 for c in final.claims if c.kind == "fact"),
        checks=dict(Counter(c.label for c in final.checks)),
        answer=final.answer,
        evidence_recall=recall_at_k(ranks, len(draft.contexts)) if ranks else 0.0,
        generate_ms=draft.timings_ms.get("generate", 0.0),
        verify_ms=final.timings_ms.get("verify", 0.0),
    )


def run_generation(
    pipeline: AnswerPipeline,
    questions: Sequence[GoldenQuestion],
    configs: Sequence[GenerationConfig],
    grader: CorrectnessJudge,
    *,
    progress: Progress = _quiet,
) -> list[GenerationRecord]:
    """Generate, verify and grade every question with all abstention rules switched off.

    The pipeline must have ``abstain_on_unanswerable=False``; the gate and the confidence
    floor are disabled per request. Policies are applied afterwards (see ``policy.py``).
    """
    if pipeline.settings.abstain_on_unanswerable:
        raise ValueError("run_generation needs a pipeline with abstain_on_unanswerable=False")
    records: list[GenerationRecord] = []
    for config in configs:
        overrides = AskOverrides(
            mode=config.mode,
            strategy=config.strategy,
            abstain_threshold=0.0,
            answer_threshold=0.0,
        )
        drafts: list[AnswerResponse] = []
        for i, q in enumerate(questions, start=1):
            progress(f"generate {config.name} {i}/{len(questions)}")
            drafts.append(pipeline.generate(q.question, overrides))
        for i, (q, draft) in enumerate(zip(questions, drafts, strict=True), start=1):
            progress(f"verify {config.name} {i}/{len(questions)}")
            final = pipeline.verify(draft, overrides)
            record = _record(config, q, draft, final)
            if final.answer and q.answerable:
                grade = grader.grade(q, final.answer)
                record.correctness = grade.score
                record.verdict = grade.verdict
                record.grade_source = grade.source
                record.key_fact_recall = key_fact_recall(final.answer, q.must_include)
            records.append(record)
    return records
