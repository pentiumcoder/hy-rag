"""Prepare a fresh deployment: index the corpus and make sure Ollama has the models.

Runs before the API starts in Docker (see docker/api.Dockerfile). Every step is idempotent:
an unchanged corpus is not re-embedded, and models Ollama already has are not pulled again,
so restarting the stack costs seconds.

    python scripts/seed.py [--skip-models]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Callable, Iterable

import httpx

from hyrag.config import Settings, get_settings
from hyrag.embeddings import embedder_from_settings
from hyrag.index import IndexManager
from hyrag.ingestion import DocumentStore
from hyrag.llm import default_model
from hyrag.models import ChunkStrategy

Log = Callable[[str], None]


def _log(message: str) -> None:
    print(f"[seed] {message}", flush=True)


def index_corpus(settings: Settings, log: Log = _log) -> dict[str, int]:
    """Load the corpus into the document store and sync every strategy's index."""
    store = DocumentStore(settings.data_dir)
    report = store.ingest_directory(settings.corpus_dir)
    docs = store.load_documents()  # corpus plus anything uploaded through the API
    log(f"documents: {len(docs)} ({len(report.added)} added, {len(report.unchanged)} unchanged)")
    embedder = embedder_from_settings(settings)
    counts: dict[str, int] = {}
    for strategy in ChunkStrategy:
        result = IndexManager(settings, strategy, embedder).sync(docs)
        counts[strategy.value] = result.total_chunks
        log(f"index {strategy.value}: {result.total_chunks} chunks (+{result.added})")
    return counts


def required_models(settings: Settings) -> list[str]:
    models: list[str] = []
    if settings.llm_provider == "ollama":
        models.append(settings.generator_model or default_model("ollama", "generator"))
    if (settings.judge_provider or settings.llm_provider) == "ollama":
        models.append(settings.judge_model or default_model("ollama", "judge"))
    return list(dict.fromkeys(models))


def _wait_for_ollama(client: httpx.Client, log: Log, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while True:
        try:
            client.get("/api/version").raise_for_status()
            return
        except httpx.HTTPError:
            if time.monotonic() > deadline:
                raise
            log("waiting for Ollama...")
            time.sleep(2)


def _installed(client: httpx.Client) -> set[str]:
    names: set[str] = set()
    for model in client.get("/api/tags").json().get("models", []):
        name = str(model.get("name", ""))
        names.add(name)
        if name.endswith(":latest"):
            names.add(name.removesuffix(":latest"))
    return names


def _progress(lines: Iterable[str], log: Log) -> None:
    """Log each new status once, and download progress every 10%, not every chunk."""
    last_status, last_decile = "", -1
    for line in lines:
        if not line.strip():
            continue
        event = json.loads(line)
        if "error" in event:
            raise RuntimeError(f"Ollama pull failed: {event['error']}")
        status = str(event.get("status", ""))
        if status != last_status:
            log(f"  {status}")
            last_status, last_decile = status, -1
        total, done = event.get("total"), event.get("completed")
        if total and done is not None:
            decile = int(done / total * 10)
            if decile > last_decile:
                if decile:
                    log(f"  {status} {decile * 10}%")
                last_decile = decile


def pull_models(
    settings: Settings,
    log: Log = _log,
    *,
    transport: httpx.BaseTransport | None = None,
    wait: float = 120.0,
) -> list[str]:
    """Pull the generator and judge into Ollama unless they are already there."""
    models = required_models(settings)
    if not models:
        return []
    with httpx.Client(
        base_url=settings.ollama_base_url, timeout=None, transport=transport
    ) as client:
        _wait_for_ollama(client, log, wait)
        installed = _installed(client)
        pulled: list[str] = []
        for model in models:
            if model in installed:
                log(f"model {model}: already present")
                continue
            log(f"model {model}: pulling (this happens once)")
            with client.stream("POST", "/api/pull", json={"model": model}) as response:
                response.raise_for_status()
                _progress(response.iter_lines(), log)
            pulled.append(model)
        return pulled


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--skip-models", action="store_true", help="Do not contact Ollama.")
    args = parser.parse_args(argv)
    settings = get_settings()
    index_corpus(settings)
    if not args.skip_models:
        pull_models(settings)
    _log("ready")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
