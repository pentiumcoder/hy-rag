"""Request and response bodies of the HTTP API that are not domain models themselves."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from hyrag.models import AskOverrides, ChunkStrategy, RetrievalMode, ScoredChunk


class AskRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    overrides: AskOverrides = Field(default_factory=AskOverrides)

    model_config = {
        "json_schema_extra": {
            "examples": [
                {"query": "How many PTO days carry over to next year?"},
                {"query": "NW-4290", "overrides": {"mode": "dense", "verify": False}},
            ]
        }
    }


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    mode: RetrievalMode = RetrievalMode.HYBRID
    strategy: ChunkStrategy | None = None
    k: int = Field(default=5, ge=1, le=20)
    rerank: bool = True


class SearchResponse(BaseModel):
    query: str
    mode: RetrievalMode
    strategy: ChunkStrategy
    results: list[ScoredChunk]


class DocumentInfo(BaseModel):
    doc_id: str
    source: str
    title: str
    format: str
    sections: int
    chars: int


class IndexSyncInfo(BaseModel):
    strategy: ChunkStrategy
    added: int
    removed: int
    unchanged: int
    duplicates: int
    chunks: int


class IngestResponse(BaseModel):
    added: list[str]
    updated: list[str]
    unchanged: list[str]
    skipped: list[str]
    indexes: list[IndexSyncInfo]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    llm_provider: str
    generator: str
    judge: str
    indexes: dict[str, int]  # strategy -> indexed chunks
    detail: str | None = None
