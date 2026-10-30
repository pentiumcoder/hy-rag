from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

import pytest

from hyrag.config import Settings
from hyrag.embeddings import FakeEmbedder
from hyrag.generation import AnswerPipeline, ExtractiveStubLLM, llm_from_settings
from hyrag.generation.prompts import INSUFFICIENT
from hyrag.index import IndexManager
from hyrag.ingestion import load_corpus
from hyrag.llm import CachedLLM, LLMError
from hyrag.llm.ollama import OllamaClient
from hyrag.llm.stub import ScriptedLLM
from hyrag.models import AskOverrides, ChunkStrategy, RetrievalMode
from hyrag.retrieval import HybridRetriever
from tests.conftest import CORPUS_DIR

ANSWERABLE = "How many unused PTO days can be carried over into the next calendar year?"


class FixedReranker:
    """Scores every passage the same, so retrieval confidence is set by the test."""

    name = "fixed"

    def __init__(self, logit: float) -> None:
        self.logit = logit

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return [self.logit - i * 0.01 for i in range(len(texts))]


@pytest.fixture(scope="module")
def indexed(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = Settings(
        corpus_dir=CORPUS_DIR,
        data_dir=tmp_path_factory.mktemp("data"),
        embedding_provider="fake",
        reranker="none",
        llm_provider="stub",
        _env_file=None,  # type: ignore[call-arg]
    )
    docs = load_corpus(CORPUS_DIR)
    embedder = FakeEmbedder(dim=settings.embedding_dim)
    for strategy in (ChunkStrategy.RECURSIVE, ChunkStrategy.FIXED):
        IndexManager(settings, strategy, embedder).sync(docs)
    return settings


def _pipeline(
    settings: Settings,
    generator: object | None = None,
    judge: object | None = None,
    logit: float | None = None,
) -> AnswerPipeline:
    embedder = FakeEmbedder(dim=settings.embedding_dim)
    index = IndexManager(settings, ChunkStrategy.RECURSIVE, embedder)
    reranker = FixedReranker(logit) if logit is not None else None
    retriever = HybridRetriever(index, embedder, reranker, settings)
    return AnswerPipeline(
        retriever,
        generator or ExtractiveStubLLM(),  # type: ignore[arg-type]
        judge or ExtractiveStubLLM(),  # type: ignore[arg-type]
        settings,
    )


def test_answers_with_verified_citations_offline(indexed: Settings) -> None:
    r = _pipeline(indexed).ask(ANSWERABLE)
    assert not r.abstained and r.abstention is None
    assert r.answer and "5" in r.answer
    assert r.claims and all(c.label == "SUPPORTED" for c in r.claims)
    assert r.citations and r.citations[0].source.startswith("hr/")
    assert not any(c.flagged for c in r.citations)
    assert r.confidence.retrieval_signal == "lexical"
    assert r.confidence.citation_coverage == 1.0
    assert r.confidence.score > 0.8
    assert r.warnings == []
    assert r.generator == r.judge == "stub:extractive"
    assert (r.mode, r.strategy) == (RetrievalMode.HYBRID, ChunkStrategy.RECURSIVE)
    assert set(r.timings_ms) == {"retrieve", "generate", "verify", "total"}
    assert len(r.contexts) == indexed.top_k
    json.loads(r.model_dump_json())  # the API on Day 4 serializes exactly this


def test_low_retrieval_confidence_abstains_without_generating(indexed: Settings) -> None:
    generator = ScriptedLLM()
    r = _pipeline(indexed, generator=generator, logit=-11.0).ask(ANSWERABLE)
    assert generator.calls == []
    assert r.abstained and r.answer is None and r.abstention
    assert r.abstention.reason == "low_retrieval_confidence"
    assert "0.03 < 0.30" in r.abstention.message
    assert r.abstention.suggested_documents
    assert r.generator is None and r.judge is None
    assert r.confidence.retrieval_signal == "rerank"
    assert r.confidence.score == pytest.approx(0.4 * r.confidence.retrieval, abs=1e-3)
    assert set(r.timings_ms) == {"retrieve", "total"}


def test_threshold_override_controls_the_gate(indexed: Settings) -> None:
    pipeline = _pipeline(indexed, logit=-5.0)  # retrieval confidence ~0.38
    assert not pipeline.ask(ANSWERABLE).abstained
    gated = pipeline.ask(ANSWERABLE, AskOverrides(abstain_threshold=0.5))
    assert gated.abstention and gated.abstention.reason == "low_retrieval_confidence"


def test_generator_abstention_becomes_a_structured_response(indexed: Settings) -> None:
    reply = f"{INSUFFICIENT}\nFound: The PTO policy [1].\nMissing: Rules for sabbaticals."
    judge = ScriptedLLM()
    r = _pipeline(indexed, generator=ScriptedLLM([reply]), judge=judge).ask(ANSWERABLE)
    assert r.abstained and r.abstention
    assert r.abstention.reason == "insufficient_context"
    assert (r.abstention.found, r.abstention.missing) == (
        ["The PTO policy."],
        ["Rules for sabbaticals."],
    )
    assert judge.calls == [] and r.generator == "stub:scripted"


def test_an_answer_made_only_of_gaps_is_an_abstention(indexed: Settings) -> None:
    reply = "The Team plan is not mentioned in the provided passages. [1][2]"
    r = _pipeline(indexed, generator=ScriptedLLM([reply])).ask("How much is the Team plan?")
    assert r.abstained and r.abstention and r.abstention.reason == "insufficient_context"
    assert r.abstention.missing == ["The Team plan is not mentioned in the provided passages."]


def test_a_planted_hallucination_is_flagged_and_costs_confidence(indexed: Settings) -> None:
    pipeline = _pipeline(indexed)
    honest = pipeline.ask(ANSWERABLE)
    planted = (
        "Up to 5 unused PTO days can be carried over into the next calendar year [1]. "
        "Unused days are paid out at 150% of salary every December [1]. "
        "Carried-over days never expire [9]."
    )
    r = _pipeline(indexed, generator=ScriptedLLM([planted])).ask(ANSWERABLE)
    assert [c.label for c in r.claims] == ["SUPPORTED", "UNSUPPORTED", "UNSUPPORTED"]
    assert r.citations[0].flagged  # passage [1] was cited for something it does not say
    assert "The answer cites passage [9], which does not exist." in r.warnings
    assert any(w.startswith("Passage [1] does not support sentence(s) 2") for w in r.warnings)
    assert r.confidence.citation_coverage == pytest.approx(1 / 3, abs=1e-3)
    assert r.confidence.score < honest.confidence.score - 0.2


def test_uncited_sentences_are_reported(indexed: Settings) -> None:
    reply = "Up to 5 unused PTO days carry over [1].\n\nAsk your manager first, always."
    r = _pipeline(indexed, generator=ScriptedLLM([reply])).ask(ANSWERABLE)
    assert [c.label for c in r.claims] == ["SUPPORTED", "UNCITED"]
    assert "Sentence(s) 2 cite no passage." in r.warnings


def test_verification_can_be_switched_off(indexed: Settings) -> None:
    judge = ScriptedLLM()
    r = _pipeline(indexed, judge=judge).ask(ANSWERABLE, AskOverrides(verify=False))
    assert judge.calls == []
    assert r.judge is None
    assert {c.label for c in r.claims} == {"UNVERIFIED"}
    assert r.confidence.citation_coverage == 0.5 and r.confidence.completeness == 0.5
    assert "Citations were not verified by the judge." in r.warnings
    assert "verify" not in r.timings_ms


def test_a_judge_failure_is_a_warning_not_a_crash(indexed: Settings) -> None:
    judge = ScriptedLLM([LLMError("judge timed out"), LLMError("judge timed out")])
    r = _pipeline(indexed, judge=judge).ask(ANSWERABLE)
    assert not r.abstained
    assert {c.label for c in r.claims} == {"UNVERIFIED"}
    assert "Citation verification failed: judge timed out" in r.warnings


def test_overrides_switch_mode_strategy_and_depth(indexed: Settings) -> None:
    pipeline = _pipeline(indexed)
    r = pipeline.ask(
        ANSWERABLE,
        AskOverrides(mode=RetrievalMode.SPARSE, strategy=ChunkStrategy.FIXED, top_k=2),
    )
    assert (r.mode, r.strategy, len(r.contexts)) == (RetrievalMode.SPARSE, ChunkStrategy.FIXED, 2)
    assert all(c.chunk.strategy is ChunkStrategy.FIXED for c in r.contexts)
    assert pipeline.retriever_for(ChunkStrategy.FIXED) is pipeline.retriever_for(
        ChunkStrategy.FIXED
    )


def test_an_empty_index_returns_no_results(tmp_path: Path) -> None:
    settings = Settings(data_dir=tmp_path, embedding_provider="fake", _env_file=None)  # type: ignore[call-arg]
    r = _pipeline(settings).ask(ANSWERABLE)
    assert r.abstention and r.abstention.reason == "no_results"
    assert r.confidence.retrieval_signal == "none" and r.confidence.score == 0.0


def test_from_settings_wires_the_stub_and_skips_the_judge_when_disabled(
    indexed: Settings,
) -> None:
    pipeline = AnswerPipeline.from_settings(indexed)
    assert isinstance(pipeline.generator, ExtractiveStubLLM)
    assert isinstance(pipeline.judge, ExtractiveStubLLM)
    no_judge = AnswerPipeline.from_settings(indexed.model_copy(update={"verify_citations": False}))
    assert no_judge.judge is None
    assert no_judge.ask(ANSWERABLE).judge is None


def test_llm_from_settings_builds_cached_ollama_clients(tmp_path: Path) -> None:
    settings = Settings(
        data_dir=tmp_path,
        llm_provider="ollama",
        judge_model="mistral:7b",
        ollama_base_url="http://gpu:11434",
        _env_file=None,  # type: ignore[call-arg]
    )
    generator = llm_from_settings(settings, "generator")
    judge = llm_from_settings(settings, "judge")
    assert isinstance(generator, CachedLLM) and isinstance(generator.inner, OllamaClient)
    assert generator.name == "ollama:qwen2.5:7b"
    assert judge.name == "ollama:mistral:7b"
    assert (tmp_path / "cache" / "llm.sqlite").exists()
    for client in (generator, judge):
        assert isinstance(client, CachedLLM)
        client.close()

    uncached = llm_from_settings(
        settings.model_copy(update={"llm_cache": False, "judge_provider": "stub"}), "judge"
    )
    assert isinstance(uncached, ExtractiveStubLLM)
    raw = llm_from_settings(settings.model_copy(update={"llm_cache": False}))
    assert isinstance(raw, OllamaClient) and raw.base_url == "http://gpu:11434"


def _judge_says(answerable: str, label: str = "UNSUPPORTED") -> ScriptedLLM:
    reply = json.dumps(
        {
            "checks": [{"id": 1, "label": label, "reason": "r"}],
            "completeness": "COMPLETE",
            "answerable": answerable,
        }
    )
    return ScriptedLLM([reply])


def test_generate_then_verify_equals_ask(indexed: Settings) -> None:
    pipeline = _pipeline(indexed)
    draft = pipeline.generate(ANSWERABLE)
    assert not draft.abstained and draft.answer and draft.judge is None
    assert {c.label for c in draft.claims} == {None}  # nothing is verified yet
    assert "verify" not in draft.timings_ms
    final = pipeline.verify(draft)
    assert final.judge == "stub:extractive" and final.claims[0].label == "SUPPORTED"
    assert final.timings_ms["total"] >= draft.timings_ms["total"]
    assert final.confidence.answerability == 1.0
    assert final.model_dump(exclude={"timings_ms"}) == pipeline.ask(ANSWERABLE).model_dump(
        exclude={"timings_ms"}
    )
    assert draft.claims[0].label is None  # verify() returns a new response


def test_verifying_an_abstention_is_a_no_op(indexed: Settings) -> None:
    pipeline = _pipeline(indexed, logit=-11.0)
    abstained = pipeline.generate(ANSWERABLE)
    assert pipeline.verify(abstained) is abstained


def test_the_judge_withdraws_an_answer_the_passages_do_not_support(indexed: Settings) -> None:
    """Item 3 of ADR-0003: a near-miss passage scores well but does not answer the question."""
    reply = "The Team plan costs $99 per month [1]."
    r = _pipeline(indexed, generator=ScriptedLLM([reply]), judge=_judge_says("NO")).ask(
        "How much does the Team plan cost?"
    )
    assert r.abstained and r.answer is None and r.abstention
    assert r.abstention.reason == "unanswerable"
    assert r.abstention.draft_answer == reply
    assert r.claims[0].label == "UNSUPPORTED"  # the evidence is kept for inspection
    assert r.judge == "stub:scripted"
    assert r.confidence.answerability == 0.0


def test_the_judge_check_keeps_answers_with_support_or_when_disabled(indexed: Settings) -> None:
    reply = "Up to 5 unused PTO days carry over [1]."
    supported = _pipeline(
        indexed, generator=ScriptedLLM([reply]), judge=_judge_says("NO", "SUPPORTED")
    ).ask(ANSWERABLE)
    assert not supported.abstained  # the judge contradicts itself; the supported claim wins

    off = indexed.model_copy(update={"abstain_on_unanswerable": False})
    kept = _pipeline(off, generator=ScriptedLLM([reply]), judge=_judge_says("NO")).ask(ANSWERABLE)
    assert not kept.abstained and kept.answer == reply


def test_the_confidence_floor_withdraws_weak_answers(indexed: Settings) -> None:
    reply = "Up to 5 unused PTO days carry over [1]."
    pipeline = _pipeline(indexed, generator=ScriptedLLM([reply, reply]), judge=_judge_says("YES"))
    floored = pipeline.ask(ANSWERABLE, AskOverrides(answer_threshold=0.99))
    assert floored.abstention and floored.abstention.reason == "low_answer_confidence"
    assert "< 0.99" in floored.abstention.message
    assert floored.abstention.draft_answer == reply

    unverified = pipeline.ask(ANSWERABLE, AskOverrides(answer_threshold=0.99, verify=False))
    assert not unverified.abstained  # the floor is calibrated on verified scores only
