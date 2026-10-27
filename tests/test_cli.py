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
    assert "hyrag 0.1.0" in result.output


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
