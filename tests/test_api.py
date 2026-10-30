from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from hyrag import __version__
from hyrag.api import create_app
from hyrag.config import Settings
from hyrag.embeddings import FakeEmbedder
from hyrag.generation import AnswerPipeline
from hyrag.index import IndexManager
from hyrag.ingestion import DocumentStore
from hyrag.llm import LLMError, LLMUnavailableError
from hyrag.models import ChunkStrategy
from tests.conftest import CORPUS_DIR

QUESTION = "How many unused PTO days can be carried over into the next calendar year?"


def _settings(data_dir: Path, **extra: object) -> Settings:
    return Settings(
        corpus_dir=CORPUS_DIR,
        data_dir=data_dir,
        embedding_provider="fake",
        reranker="none",
        llm_provider="stub",
        eval_results_dir=data_dir / "eval",
        _env_file=None,  # type: ignore[call-arg]
        **extra,  # type: ignore[arg-type]
    )


def _ingest(settings: Settings) -> None:
    store = DocumentStore(settings.data_dir)
    store.ingest_directory(CORPUS_DIR)
    docs = store.load_documents()
    for strategy in ChunkStrategy:
        IndexManager(settings, strategy, FakeEmbedder(dim=settings.embedding_dim)).sync(docs)


@pytest.fixture(scope="module")
def indexed_settings(tmp_path_factory: pytest.TempPathFactory) -> Settings:
    settings = _settings(tmp_path_factory.mktemp("data"))
    _ingest(settings)
    return settings


@pytest.fixture
def client(indexed_settings: Settings) -> Iterator[TestClient]:
    with TestClient(create_app(indexed_settings)) as c:
        yield c


def test_health_reports_models_and_indexes(client: TestClient) -> None:
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["version"] == __version__
    assert body["generator"] == body["judge"] == "stub:extractive"
    assert set(body["indexes"]) == {"fixed", "recursive", "semantic"}
    assert all(n > 50 for n in body["indexes"].values())


def test_health_is_degraded_without_an_index(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as c:
        body = c.get("/health").json()
    assert body["status"] == "degraded" and "empty" in body["detail"]


def test_ask_returns_the_full_answer_response(client: TestClient) -> None:
    response = client.post("/v1/ask", json={"query": QUESTION})
    assert response.status_code == 200, response.text
    body = response.json()
    assert not body["abstained"] and "5" in body["answer"]
    assert body["citations"][0]["source"].startswith("hr/")
    assert {"score", "retrieval", "citation_coverage", "completeness"} <= set(body["confidence"])
    assert body["mode"] == "hybrid" and body["strategy"] == "recursive"


def test_ask_overrides_switch_mode_and_strategy(client: TestClient) -> None:
    body = client.post(
        "/v1/ask",
        json={
            "query": QUESTION,
            "overrides": {"mode": "dense", "strategy": "fixed", "top_k": 3, "verify": False},
        },
    ).json()
    assert (body["mode"], body["strategy"], len(body["contexts"])) == ("dense", "fixed", 3)
    assert body["judge"] is None


def test_ask_validates_its_input(client: TestClient) -> None:
    assert client.post("/v1/ask", json={"query": ""}).status_code == 422
    bad = client.post("/v1/ask", json={"query": "q", "overrides": {"top_k": 99}})
    assert bad.status_code == 422


def test_ask_maps_model_failures_to_http_errors(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unavailable(*args: object, **kwargs: object) -> None:
        raise LLMUnavailableError("Start it with `ollama serve`.")

    monkeypatch.setattr(AnswerPipeline, "ask", unavailable)
    response = client.post("/v1/ask", json={"query": QUESTION})
    assert response.status_code == 503 and "ollama serve" in response.json()["detail"]

    def broken(*args: object, **kwargs: object) -> None:
        raise LLMError("bad reply")

    monkeypatch.setattr(AnswerPipeline, "ask", broken)
    assert client.post("/v1/ask", json={"query": QUESTION}).status_code == 502


def test_ask_on_an_empty_index_is_a_conflict(tmp_path: Path) -> None:
    with TestClient(create_app(_settings(tmp_path))) as c:
        response = c.post("/v1/ask", json={"query": QUESTION})
    assert response.status_code == 409 and "Ingest documents first" in response.json()["detail"]


def test_ask_with_a_mismatched_embedder_is_a_conflict(
    indexed_settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("hyrag.embeddings.fake.FakeEmbedder.name", "fake-64")
    other = indexed_settings.model_copy(update={"embedding_dim": 64})
    with TestClient(create_app(other)) as c:
        response = c.post("/v1/ask", json={"query": QUESTION})
        assert c.get("/health").json()["status"] == "degraded"
    assert response.status_code == 409


def test_search_returns_per_stage_scores(client: TestClient) -> None:
    body = client.post(
        "/v1/search", json={"query": "NW-4012", "mode": "sparse", "k": 3, "strategy": "fixed"}
    ).json()
    assert body["strategy"] == "fixed" and len(body["results"]) == 3
    assert "engineering/error-codes.md" in {r["chunk"]["source"] for r in body["results"]}
    top = body["results"][0]
    assert top["sparse_rank"] == 1 and top["dense_rank"] is None


def test_documents_lists_the_store(client: TestClient) -> None:
    docs = client.get("/v1/documents").json()
    assert len(docs) == 26
    assert docs == sorted(docs, key=lambda d: d["source"])
    pdf = next(d for d in docs if d["source"] == "finance/travel-and-expense.pdf")
    assert pdf["format"] == "pdf" and pdf["sections"] == 7 and pdf["chars"] > 500


def test_ingest_uploads_and_indexes_new_documents(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _ingest(settings)
    note = b"# Quantum Coffee Policy\n\nEvery office gets a quantum espresso machine named Q-77.\n"
    with TestClient(create_app(settings)) as c:
        response = c.post("/v1/ingest", files=[("files", ("coffee.md", note, "text/markdown"))])
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["added"] == ["uploads/coffee.md"]
        assert [i["strategy"] for i in body["indexes"]] == ["fixed", "recursive", "semantic"]
        assert all(i["added"] >= 1 for i in body["indexes"])

        again = c.post("/v1/ingest", files=[("files", ("coffee.md", note, "text/markdown"))])
        assert again.json()["unchanged"] == ["uploads/coffee.md"]

        hits = c.post("/v1/search", json={"query": "Q-77 espresso", "mode": "sparse"}).json()
        assert hits["results"][0]["chunk"]["source"] == "uploads/coffee.md"
        assert any(d["source"] == "uploads/coffee.md" for d in c.get("/v1/documents").json())
    assert (tmp_path / "uploads" / "coffee.md").read_bytes() == note


def test_ingest_rejects_unsupported_and_oversized_files(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    exe = client.post("/v1/ingest", files=[("files", ("tool.exe", b"MZ", "application/x"))])
    assert exe.status_code == 415 and ".pdf" in exe.json()["detail"]
    monkeypatch.setattr("hyrag.api.app.MAX_UPLOAD_BYTES", 10)
    big = client.post("/v1/ingest", files=[("files", ("big.md", b"x" * 11, "text/markdown"))])
    assert big.status_code == 413


def test_eval_summary_is_served_once_written(indexed_settings: Settings) -> None:
    with TestClient(create_app(indexed_settings)) as c:
        assert c.get("/v1/eval/summary").status_code == 404
        indexed_settings.eval_results_dir.mkdir(parents=True, exist_ok=True)
        (indexed_settings.eval_results_dir / "summary.json").write_text(
            json.dumps({"retrieval": {"best": "recursive/hybrid+rerank"}}), encoding="utf-8"
        )
        assert c.get("/v1/eval/summary").json()["retrieval"]["best"] == "recursive/hybrid+rerank"


def test_openapi_documents_every_endpoint(client: TestClient) -> None:
    paths = set(client.get("/openapi.json").json()["paths"])
    assert paths == {
        "/health",
        "/v1/ask",
        "/v1/search",
        "/v1/documents",
        "/v1/ingest",
        "/v1/eval/summary",
    }


def test_cors_allows_the_dashboard_dev_server(client: TestClient) -> None:
    response = client.options(
        "/v1/ask",
        headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "POST"},
    )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"
