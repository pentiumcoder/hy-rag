from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import ModuleType

import httpx
import pytest

from hyrag.config import Settings
from tests.conftest import CORPUS_DIR, REPO_ROOT


def _load_seed() -> ModuleType:
    spec = importlib.util.spec_from_file_location("seed", REPO_ROOT / "scripts" / "seed.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


seed = _load_seed()


def _settings(tmp_path: Path, **extra: object) -> Settings:
    return Settings(
        corpus_dir=CORPUS_DIR,
        data_dir=tmp_path,
        embedding_provider="fake",
        reranker="none",
        _env_file=None,  # type: ignore[call-arg]
        **extra,  # type: ignore[arg-type]
    )


def test_index_corpus_is_idempotent(tmp_path: Path) -> None:
    logs: list[str] = []
    settings = _settings(tmp_path)
    first = seed.index_corpus(settings, logs.append)
    assert set(first) == {"fixed", "recursive", "semantic"} and all(n > 50 for n in first.values())
    assert "documents: 26 (26 added, 0 unchanged)" in logs
    logs.clear()
    assert seed.index_corpus(settings, logs.append) == first
    assert "documents: 26 (0 added, 26 unchanged)" in logs
    assert all("(+0)" in line for line in logs if line.startswith("index "))


def test_required_models_follow_the_settings(tmp_path: Path) -> None:
    assert seed.required_models(_settings(tmp_path, llm_provider="ollama")) == [
        "qwen2.5:7b",
        "llama3.1:8b",
    ]
    light = _settings(
        tmp_path, llm_provider="ollama", generator_model="qwen2.5:3b", judge_model="llama3.2:3b"
    )
    assert seed.required_models(light) == ["qwen2.5:3b", "llama3.2:3b"]
    assert seed.required_models(_settings(tmp_path, llm_provider="stub")) == []
    mixed = _settings(tmp_path, llm_provider="stub", judge_provider="ollama")
    assert seed.required_models(mixed) == ["llama3.1:8b"]


def _ollama(installed: list[str], pulls: list[str], fail: bool = False) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/version":
            return httpx.Response(200, json={"version": "0.40.13"})
        if request.url.path == "/api/tags":
            return httpx.Response(200, json={"models": [{"name": n} for n in installed]})
        if request.url.path == "/api/pull":
            pulls.append(json.loads(request.content)["model"])
            events = [{"status": "pulling manifest"}]
            if fail:
                events.append({"error": "model not found"})
            else:
                events += [
                    {"status": "pulling abc", "total": 100, "completed": c} for c in (0, 5, 50, 100)
                ] + [{"status": "success"}]
            return httpx.Response(200, text="\n".join(json.dumps(e) for e in events) + "\n")
        return httpx.Response(404)

    return httpx.MockTransport(handler)


def test_pull_models_skips_present_ones_and_logs_progress(tmp_path: Path) -> None:
    pulls: list[str] = []
    logs: list[str] = []
    settings = _settings(tmp_path, llm_provider="ollama")
    pulled = seed.pull_models(
        settings, logs.append, transport=_ollama(["llama3.1:8b"], pulls), wait=0
    )
    assert pulled == pulls == ["qwen2.5:7b"]
    assert "model llama3.1:8b: already present" in logs
    assert "  pulling abc 50%" in logs and "  pulling abc 100%" in logs and "  success" in logs
    assert sum("pulling abc" in line for line in logs) == 3  # status once, then 50% and 100%


def test_pull_models_matches_latest_tags_and_reports_errors(tmp_path: Path) -> None:
    settings = _settings(tmp_path, llm_provider="ollama", judge_model="mistral")
    pulls: list[str] = []
    seed.pull_models(
        settings, lambda m: None, transport=_ollama(["qwen2.5:7b", "mistral:latest"], pulls), wait=0
    )
    assert pulls == []
    with pytest.raises(RuntimeError, match="model not found"):
        seed.pull_models(
            settings.model_copy(update={"judge_model": "nope"}),
            lambda m: None,
            transport=_ollama(["qwen2.5:7b"], [], fail=True),
            wait=0,
        )


def test_pull_models_gives_up_when_ollama_never_starts(tmp_path: Path) -> None:
    def down(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("refused")

    with pytest.raises(httpx.ConnectError):
        seed.pull_models(
            _settings(tmp_path, llm_provider="ollama"),
            lambda m: None,
            transport=httpx.MockTransport(down),
            wait=0,
        )
    assert seed.pull_models(_settings(tmp_path, llm_provider="stub"), lambda m: None) == []


def test_main_runs_both_steps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(seed, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(seed, "index_corpus", lambda s: calls.append("index"))
    monkeypatch.setattr(seed, "pull_models", lambda s: calls.append("pull"))
    assert seed.main([]) == 0 and calls == ["index", "pull"]
    calls.clear()
    assert seed.main(["--skip-models"]) == 0 and calls == ["index"]
