"""Runtime settings. Every knob lives here and can be overridden with HYRAG_* env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hyrag.models import ChunkStrategy


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="HYRAG_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    corpus_dir: Path = Path("data/corpus")
    data_dir: Path = Path("data")

    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = Field(default=800, ge=100, le=8000)
    chunk_overlap: int = Field(default=120, ge=0)

    semantic_breakpoint_percentile: float = Field(default=20.0, gt=0, lt=100)
    semantic_min_chars: int = Field(default=200, ge=0)
    semantic_max_chars: int = Field(default=1200, ge=100)

    # Embeddings. "fastembed" runs bge-small locally; "fake" is the deterministic test embedder.
    embedding_provider: Literal["fastembed", "openai", "fake"] = "fastembed"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_dim: int = Field(default=256, ge=8)  # only used by the fake embedder
    embedding_cache: bool = True
    model_cache_dir: Path = Path(".cache/models")

    # Indexing: a new chunk is skipped if its cosine similarity to an indexed one exceeds this.
    dedup_threshold: float = Field(default=0.95, gt=0, le=1)

    # Retrieval: dense and sparse top-k, weighted RRF, then cross-encoder rerank to top_k.
    dense_k: int = Field(default=10, ge=1)
    sparse_k: int = Field(default=10, ge=1)
    rrf_k: int = Field(default=60, ge=1)
    # Equal weights by default: with rrf_k=60, rank scores differ by only ~15% across the top
    # 10, so even 0.7/0.3 would stop a BM25-only hit from outranking any dense hit (ADR-0002).
    dense_weight: float = Field(default=0.5, ge=0)
    sparse_weight: float = Field(default=0.5, ge=0)
    fusion_top_n: int = Field(default=20, ge=1)
    reranker: Literal["cross-encoder", "none"] = "cross-encoder"
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_k: int = Field(default=5, ge=1)

    # Generation. "ollama" runs locally; "stub" is the deterministic extractive model for tests.
    # Model names default per provider (see hyrag.llm.DEFAULT_MODELS) when left empty.
    llm_provider: Literal["ollama", "openai", "anthropic", "stub"] = "ollama"
    generator_model: str | None = None
    # The judge defaults to a different model family than the generator (ADR-0003).
    judge_provider: Literal["ollama", "openai", "anthropic", "stub"] | None = None
    judge_model: str | None = None
    ollama_base_url: str = "http://localhost:11434"
    ollama_num_ctx: int = Field(default=8192, ge=512)
    llm_timeout: float = Field(default=180.0, gt=0)
    llm_temperature: float = Field(default=0.0, ge=0, le=2)
    generator_max_tokens: int = Field(default=512, ge=16)
    judge_max_tokens: int = Field(default=1024, ge=16)
    llm_cache: bool = True

    # Answering: citation verification, confidence and abstention.
    verify_citations: bool = True
    # Below this retrieval confidence nothing is generated and the answer is a structured
    # abstention. Rerank logits map to [0, 1] with sigmoid((logit - center) / scale).
    # 0.3 is the middle of the range that is equally good on the eval set (ADR-0004).
    abstain_threshold: float = Field(default=0.3, ge=0, le=1)
    rerank_center: float = -4.0
    rerank_scale: float = Field(default=2.0, gt=0)
    # After verification: withdraw the answer if the judge says the passages do not answer the
    # question and no sentence is supported, or if confidence is below answer_threshold. The
    # eval found no floor that helps (it drops correct answers as fast as wrong ones), so 0.
    abstain_on_unanswerable: bool = True
    answer_threshold: float = Field(default=0.0, ge=0, le=1)

    # Evaluation and API.
    eval_results_dir: Path = Path("eval/results")
    api_cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    @model_validator(mode="after")
    def _check_sizes(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.semantic_min_chars >= self.semantic_max_chars:
            raise ValueError("semantic_min_chars must be smaller than semantic_max_chars")
        if self.dense_weight + self.sparse_weight == 0:
            raise ValueError("dense_weight and sparse_weight cannot both be zero")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"

    @property
    def index_dir(self) -> Path:
        return self.data_dir / "index"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
