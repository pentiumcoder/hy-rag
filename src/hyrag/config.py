"""Runtime settings. Every knob lives here and can be overridden with HYRAG_* env vars."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

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

    embedding_provider: str = "fake"
    embedding_dim: int = Field(default=256, ge=8)

    @model_validator(mode="after")
    def _check_sizes(self) -> Settings:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        if self.semantic_min_chars >= self.semantic_max_chars:
            raise ValueError("semantic_min_chars must be smaller than semantic_max_chars")
        return self

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
