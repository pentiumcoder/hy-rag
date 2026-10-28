from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from hyrag.cli import app
from hyrag.config import get_settings

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
    assert "hyrag 0.2.0" in result.output


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
