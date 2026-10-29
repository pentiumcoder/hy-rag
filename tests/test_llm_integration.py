"""End-to-end answering with the real local models: bge-small, MiniLM, qwen2.5 and llama3.1.

Skipped by default. Needs Ollama running with both models pulled:

    ollama pull qwen2.5:7b && ollama pull llama3.1:8b
    HYRAG_TEST_LLM=1 pytest -m llm

Assertions are about behaviour the pipeline guarantees (abstaining, flagging an invented
fact), not about the exact wording a 7B model chooses.
"""

from __future__ import annotations

import os
from collections.abc import Iterator

import pytest

from hyrag.config import Settings
from hyrag.embeddings import embedder_from_settings
from hyrag.generation import AnswerPipeline, CitationVerifier, extract_claims, llm_from_settings
from hyrag.index import IndexManager
from hyrag.ingestion import load_corpus
from hyrag.llm.stub import ScriptedLLM
from hyrag.models import ChunkStrategy
from hyrag.retrieval import HybridRetriever, get_reranker
from tests.conftest import CORPUS_DIR, REPO_ROOT

pytestmark = [
    pytest.mark.llm,
    pytest.mark.skipif(
        os.environ.get("HYRAG_TEST_LLM") != "1", reason="set HYRAG_TEST_LLM=1 to run"
    ),
]


@pytest.fixture(scope="module")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AnswerPipeline]:
    settings = Settings(
        corpus_dir=CORPUS_DIR,
        data_dir=tmp_path_factory.mktemp("data"),
        embedding_provider="fastembed",
        reranker="cross-encoder",
        llm_provider="ollama",
        llm_cache=False,
        model_cache_dir=REPO_ROOT / ".cache" / "models",
        _env_file=None,  # type: ignore[call-arg]
    )
    embedder = embedder_from_settings(settings)
    index = IndexManager(settings, ChunkStrategy.RECURSIVE, embedder)
    index.sync(load_corpus(CORPUS_DIR))
    reranker = get_reranker("cross-encoder", cache_dir=settings.model_cache_dir)
    retriever = HybridRetriever(index, embedder, reranker, settings)
    yield AnswerPipeline.from_settings(settings, retriever=retriever)


def test_answers_an_exact_term_question_with_a_supported_citation(
    pipeline: AnswerPipeline,
) -> None:
    r = pipeline.ask("What error do I get when a batch exceeds ingest.max_batch_size?")
    assert not r.abstained and r.answer
    assert "NW-4012" in r.answer
    assert any(c.label == "SUPPORTED" for c in r.claims)
    assert any(c.source == "engineering/error-codes.md" for c in r.citations)
    assert r.confidence.score > 0.6
    assert r.generator == "ollama:qwen2.5:7b" and r.judge == "ollama:llama3.1:8b"


def test_an_off_topic_question_is_refused_before_generation(pipeline: AnswerPipeline) -> None:
    r = pipeline.ask("Who won the 2022 football world cup?")
    assert r.abstained and r.abstention
    assert r.abstention.reason == "low_retrieval_confidence"
    assert r.generator is None


def test_a_near_miss_question_is_not_answered(pipeline: AnswerPipeline) -> None:
    """The pricing page is retrieved, but it has no Team plan."""
    r = pipeline.ask("How much does the Team plan cost per month?")
    assert r.abstained or all(c.label != "SUPPORTED" for c in r.claims), r.answer


def test_the_judge_flags_a_planted_hallucination(pipeline: AnswerPipeline) -> None:
    retriever = pipeline.retriever_for(ChunkStrategy.RECURSIVE)
    contexts = retriever.retrieve("How many PTO days carry over to next year?", k=3)
    source = next(i for i, c in enumerate(contexts, 1) if "carried over" in c.chunk.text)
    answer = (
        f"Up to 5 unused PTO days can be carried over into the next calendar year [{source}]. "
        f"Unused days beyond the limit are paid out at 150% of salary each December [{source}]."
    )
    judge = llm_from_settings(pipeline.settings, "judge")
    result = CitationVerifier(judge).verify(
        "How many PTO days carry over?", answer, extract_claims(answer), contexts
    )
    labels = [c.label for c in result.checks]
    assert labels[0] in {"SUPPORTED", "PARTIAL"}
    assert labels[1] == "UNSUPPORTED"


def test_the_pipeline_survives_a_generator_that_ignores_the_format(
    pipeline: AnswerPipeline,
) -> None:
    sloppy = AnswerPipeline(
        pipeline.retriever_for(ChunkStrategy.RECURSIVE),
        ScriptedLLM(["Sure! You can carry over 5 days, I think."]),
        pipeline.judge,
        pipeline.settings,
    )
    r = sloppy.ask("How many PTO days carry over to next year?")
    assert [c.label for c in r.claims] == ["UNCITED"]
    assert r.confidence.citation_coverage == 0.0
