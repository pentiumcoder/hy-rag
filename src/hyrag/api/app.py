"""FastAPI service: ``/v1/ask``, ``/v1/search``, ``/v1/documents``, ``/v1/ingest``, eval summary.

The pipeline (and with it the embedding and reranking models) is built on the first request
that needs it, not at import, so ``/health`` answers immediately and tests can swap in a
pipeline built from fakes.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Annotated, Any

from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from hyrag import __version__
from hyrag.api.schemas import (
    AskRequest,
    DocumentInfo,
    HealthResponse,
    IndexSyncInfo,
    IngestResponse,
    SearchRequest,
    SearchResponse,
)
from hyrag.config import Settings, get_settings
from hyrag.generation import AnswerPipeline
from hyrag.index import IndexManager, IndexMismatchError
from hyrag.ingestion import SUPPORTED_EXTENSIONS, DocumentStore
from hyrag.llm import LLMError, LLMUnavailableError, default_model
from hyrag.models import AnswerResponse, ChunkStrategy

MAX_UPLOAD_BYTES = 10 * 1024 * 1024


class Services:
    """Everything a request handler needs, created lazily and shared by all requests."""

    def __init__(self, settings: Settings, pipeline: AnswerPipeline | None = None) -> None:
        self.settings = settings
        self._pipeline = pipeline
        self._build_lock = threading.Lock()
        self.write_lock = threading.Lock()  # one ingest at a time; indexes have a single writer

    @property
    def pipeline(self) -> AnswerPipeline:
        with self._build_lock:
            if self._pipeline is None:
                self._pipeline = AnswerPipeline.from_settings(self.settings)
            return self._pipeline

    def index(self, strategy: ChunkStrategy | None) -> IndexManager:
        pipeline = self.pipeline
        return pipeline.retriever_for(strategy or pipeline.default_strategy).index

    def model_names(self) -> tuple[str, str]:
        s = self.settings
        generator = (
            "stub:extractive"
            if s.llm_provider == "stub"
            else s.generator_model or default_model(s.llm_provider, "generator")
        )
        judge_provider = s.judge_provider or s.llm_provider
        judge = (
            "stub:extractive"
            if judge_provider == "stub"
            else s.judge_model or default_model(judge_provider, "judge")
        )
        return generator, judge


def get_services(request: Request) -> Services:
    services: Services = request.app.state.services
    return services


ServicesDep = Annotated[Services, Depends(get_services)]


def _require_index(services: Services, strategy: ChunkStrategy | None) -> None:
    try:
        index = services.index(strategy)
    except IndexMismatchError as exc:
        raise HTTPException(409, str(exc)) from exc
    if len(index) == 0:
        raise HTTPException(409, "The index is empty. Ingest documents first.")


def create_app(settings: Settings | None = None, pipeline: AnswerPipeline | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(
        title="hy-rag",
        version=__version__,
        summary="Hybrid-search RAG over internal docs, with verified citations.",
        description=(
            "Ask questions about the indexed documents. Every answer cites its passages as "
            "`[n]`, each citation is checked by a judge model, and weak questions get a "
            "structured abstention instead of a guess."
        ),
    )
    app.state.services = Services(settings, pipeline)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    @app.get("/health", response_model=HealthResponse, tags=["meta"])
    def health(services: ServicesDep) -> HealthResponse:
        generator, judge = services.model_names()
        indexes: dict[str, int] = {}
        detail = None
        try:
            for strategy in ChunkStrategy:
                indexes[strategy.value] = len(services.index(strategy))
        except IndexMismatchError as exc:
            detail = str(exc)
        default = services.settings.chunk_strategy.value
        ok = detail is None and indexes.get(default, 0) > 0
        return HealthResponse(
            status="ok" if ok else "degraded",
            version=__version__,
            llm_provider=services.settings.llm_provider,
            generator=generator,
            judge=judge,
            indexes=indexes,
            detail=detail or (None if ok else f"The {default} index is empty."),
        )

    @app.post("/v1/ask", response_model=AnswerResponse, tags=["answers"])
    def ask(body: AskRequest, services: ServicesDep) -> AnswerResponse:
        """Answer with `[n]` citations, verified sources and a confidence breakdown.

        `overrides` switches retrieval mode (`hybrid`, `dense`, `sparse`), chunking strategy,
        `top_k`, reranking, verification and the abstention thresholds for this request only.
        """
        _require_index(services, body.overrides.strategy)
        try:
            return services.pipeline.ask(body.query, body.overrides)
        except LLMUnavailableError as exc:
            raise HTTPException(503, str(exc)) from exc
        except LLMError as exc:
            raise HTTPException(502, str(exc)) from exc

    @app.post("/v1/search", response_model=SearchResponse, tags=["answers"])
    def search(body: SearchRequest, services: ServicesDep) -> SearchResponse:
        """Retrieval only: ranked chunks with their dense, BM25, RRF and rerank scores."""
        _require_index(services, body.strategy)
        retriever = services.pipeline.retriever_for(
            body.strategy or services.pipeline.default_strategy
        )
        results = retriever.retrieve(body.query, body.mode, body.k, rerank=body.rerank)
        return SearchResponse(
            query=body.query, mode=body.mode, strategy=retriever.index.strategy, results=results
        )

    @app.get("/v1/documents", response_model=list[DocumentInfo], tags=["documents"])
    def documents(services: ServicesDep) -> list[DocumentInfo]:
        """Every document in the store, from the corpus and from uploads."""
        docs = DocumentStore(services.settings.data_dir).load_documents()
        return [
            DocumentInfo(
                doc_id=d.doc_id,
                source=d.source,
                title=d.title,
                format=d.format,
                sections=len(d.sections),
                chars=d.char_count,
            )
            for d in sorted(docs, key=lambda d: d.source)
        ]

    @app.post("/v1/ingest", response_model=IngestResponse, tags=["documents"])
    def ingest(
        services: ServicesDep,
        files: Annotated[list[UploadFile], File(description="Markdown, text, HTML or PDF")],
    ) -> IngestResponse:
        """Upload documents, then chunk, dedup, embed and index them for every strategy.

        Uploads are stored under `data/uploads/` and appear as `uploads/<name>` sources.
        """
        upload_dir = services.settings.data_dir / "uploads"
        saved: list[Path] = []
        for upload in files:
            name = Path(upload.filename or "").name
            if not name or Path(name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                allowed = ", ".join(sorted(SUPPORTED_EXTENSIONS))
                raise HTTPException(415, f"Unsupported file {name!r}; use one of {allowed}.")
            content = upload.file.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                raise HTTPException(413, f"{name} is larger than 10 MB.")
            upload_dir.mkdir(parents=True, exist_ok=True)
            target = upload_dir / name
            target.write_bytes(content)
            saved.append(target)

        with services.write_lock:
            store = DocumentStore(services.settings.data_dir)
            report = store.ingest_files(saved, services.settings.data_dir)
            docs = store.load_documents()
            synced: list[IndexSyncInfo] = []
            for strategy in ChunkStrategy:
                try:
                    result = services.index(strategy).sync(docs)
                except IndexMismatchError as exc:
                    raise HTTPException(409, str(exc)) from exc
                synced.append(
                    IndexSyncInfo(
                        strategy=strategy,
                        added=result.added,
                        removed=result.removed,
                        unchanged=result.unchanged,
                        duplicates=result.total_duplicates,
                        chunks=result.total_chunks,
                    )
                )
        return IngestResponse(
            added=report.added,
            updated=report.updated,
            unchanged=report.unchanged,
            skipped=report.skipped,
            indexes=synced,
        )

    @app.get("/v1/eval/summary", tags=["evaluation"])
    def eval_summary(services: ServicesDep) -> dict[str, Any]:
        """The latest `eval/results/summary.json` written by `hyrag eval`."""
        path = services.settings.eval_results_dir / "summary.json"
        if not path.exists():
            raise HTTPException(404, "No evaluation results yet. Run `hyrag eval`.")
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return data

    return app
