from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from hyrag.cli import app
from hyrag.config import get_settings
from hyrag.llm import LLMUnavailableError

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HYRAG_DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "hyrag 0.3.0" in result.output


def test_load_then_reload(corpus_dir: Path) -> None:
    first = runner.invoke(app, ["load", "--corpus", str(corpus_dir)])
    assert first.exit_code == 0, first.output
    assert "added 26" in first.output
    second = runner.invoke(app, ["load", "--corpus", str(corpus_dir)])
    assert "unchanged 26" in second.output


def test_chunk_compare_and_show(corpus_dir: Path) -> None:
    result = runner.invoke(app, ["chunk", "--compare", "--show", "1", "--corpus", str(corpus_dir)])
    assert result.exit_code == 0, result.output
    for strategy in ("fixed", "recursive", "semantic"):
        assert strategy in result.output


def test_chunk_single_strategy(corpus_dir: Path) -> None:
    result = runner.invoke(app, ["chunk", "-s", "fixed", "--corpus", str(corpus_dir)])
    assert result.exit_code == 0, result.output
    assert "fixed" in result.output and "semantic" not in result.output


def test_ingest_then_search(corpus_dir: Path) -> None:
    first = runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])
    assert first.exit_code == 0, first.output
    assert "documents: 26" in first.output and "recursive" in first.output

    again = runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])
    assert again.exit_code == 0, again.output
    assert "unchanged 26" in again.output

    result = runner.invoke(app, ["search", "NW-4012", "--mode", "sparse", "-k", "3"])
    assert result.exit_code == 0, result.output
    assert "error-codes.md" in result.output
    hybrid = runner.invoke(app, ["search", "rotate TLS certificates", "--full", "--no-rerank"])
    assert hybrid.exit_code == 0, hybrid.output
    assert "certificate-rotation.md" in hybrid.output


def test_ingest_all_strategies(corpus_dir: Path) -> None:
    result = runner.invoke(app, ["ingest", "--all", "--corpus", str(corpus_dir)])
    assert result.exit_code == 0, result.output
    for strategy in ("fixed", "recursive", "semantic"):
        assert strategy in result.output
    assert "duplicates" in result.output


def test_search_before_ingest_and_after_embedder_change(
    corpus_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = runner.invoke(app, ["search", "anything"])
    assert empty.exit_code == 1 and "hyrag ingest" in empty.output

    runner.invoke(app, ["ingest", "-s", "fixed", "--corpus", str(corpus_dir)])
    monkeypatch.setenv("HYRAG_EMBEDDING_DIM", "64")
    monkeypatch.setattr("hyrag.embeddings.fake.FakeEmbedder.name", "fake-64")
    get_settings.cache_clear()
    mismatch = runner.invoke(app, ["search", "anything", "-s", "fixed"])
    assert mismatch.exit_code == 1 and "was built with" in mismatch.output


def test_ask_answers_with_citations_and_confidence(corpus_dir: Path) -> None:
    runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])
    result = runner.invoke(
        app,
        ["ask", "How many unused PTO days can be carried over into the next year?", "--context"],
    )
    assert result.exit_code == 0, result.output
    for text in ("Answer", "SUPPORTED", "confidence", "hr/", "stub:extractive", "retrieved"):
        assert text in result.output


def test_ask_json_output_is_the_full_response(corpus_dir: Path) -> None:
    runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])
    result = runner.invoke(
        app, ["ask", "How many PTO days carry over?", "--json", "--no-verify", "-k", "3"]
    )
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["query"] == "How many PTO days carry over?"
    assert len(data["contexts"]) == 3 and data["judge"] is None
    assert set(data["confidence"]) >= {"score", "retrieval", "citation_coverage", "completeness"}


def test_ask_shows_a_structured_abstention(corpus_dir: Path) -> None:
    runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])
    result = runner.invoke(app, ["ask", "Which telescope photographed the quasar zebra?"])
    assert result.exit_code == 0, result.output
    assert "No confident answer" in result.output
    assert "Missing:" in result.output and "Closest documents" in result.output


def test_ask_reports_empty_index_and_unreachable_model(
    corpus_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty = runner.invoke(app, ["ask", "anything"])
    assert empty.exit_code == 1 and "hyrag ingest" in empty.output

    runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])

    def down(*args: object, **kwargs: object) -> None:
        raise LLMUnavailableError("Cannot reach Ollama. Start it with `ollama serve`.")

    monkeypatch.setattr("hyrag.generation.pipeline.AnswerPipeline.ask", down)
    result = runner.invoke(app, ["ask", "How many PTO days carry over?"])
    assert result.exit_code == 1 and "ollama serve" in result.output


def test_ask_reports_an_embedder_mismatch(
    corpus_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["ingest", "--corpus", str(corpus_dir)])
    monkeypatch.setenv("HYRAG_EMBEDDING_DIM", "64")
    monkeypatch.setattr("hyrag.embeddings.fake.FakeEmbedder.name", "fake-64")
    get_settings.cache_clear()
    result = runner.invoke(app, ["ask", "anything"])
    assert result.exit_code == 1 and "was built with" in result.output
