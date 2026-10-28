"""The single writer for both indexes.

Dense (Chroma) and sparse (BM25) indexes must always contain the same chunks, or hybrid
fusion silently compares different corpora. Everything that writes goes through
:class:`IndexManager`, which keeps a manifest of indexed chunk ids next to both indexes.

Write order is Chroma, then BM25, then the manifest. If a crash interrupts it, the next
open notices the id sets disagree and repairs BM25 and the manifest from Chroma, which is
the source of truth because it also stores every chunk's text and metadata.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from hyrag.chunking import Chunker, get_chunker
from hyrag.config import Settings
from hyrag.embeddings import Embedder
from hyrag.index.bm25_store import BM25Store
from hyrag.index.dedup import Duplicate, find_duplicates
from hyrag.index.vector_store import VectorStore
from hyrag.models import Chunk, ChunkStrategy, Document


class IndexMismatchError(RuntimeError):
    """The index was built with a different embedder than the one in use."""


class Manifest(BaseModel):
    strategy: ChunkStrategy
    embedder: str | None = None
    chunks: dict[str, str] = Field(default_factory=dict)  # chunk_id -> source
    duplicates: dict[str, Duplicate] = Field(default_factory=dict)  # skipped chunk_id -> record
    updated_at: str | None = None


class SyncReport(BaseModel):
    strategy: ChunkStrategy
    added: int = 0
    removed: int = 0
    unchanged: int = 0
    new_duplicates: list[Duplicate] = Field(default_factory=list)
    total_chunks: int = 0
    total_duplicates: int = 0
    rebuilt: bool = False  # the embedder changed, so everything was re-embedded
    repaired: bool = False  # the indexes had drifted and were reconciled on open


def _write_json(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(payload, encoding="utf-8")
    os.replace(tmp, path)


class IndexManager:
    def __init__(
        self,
        settings: Settings,
        strategy: ChunkStrategy | str,
        embedder: Embedder,
        chunker: Chunker | None = None,
    ) -> None:
        self.settings = settings
        self.strategy = ChunkStrategy(strategy)
        self.embedder = embedder
        self._chunker = chunker
        self.dir = settings.index_dir / self.strategy.value
        self.vectors = VectorStore(settings.index_dir / "chroma", f"hyrag_{self.strategy.value}")
        self.bm25 = BM25Store.load(self.dir / "bm25.json")
        self.manifest = self._load_manifest()
        self.repaired = self._reconcile()

    # ----- persistence -------------------------------------------------------------------

    @property
    def manifest_path(self) -> Path:
        return self.dir / "manifest.json"

    @property
    def dedup_report_path(self) -> Path:
        return self.dir / "dedup_report.json"

    def _load_manifest(self) -> Manifest:
        if self.manifest_path.exists():
            return Manifest.model_validate_json(self.manifest_path.read_text(encoding="utf-8"))
        return Manifest(strategy=self.strategy)

    def _save(self) -> None:
        self.bm25.save()
        self.manifest.updated_at = datetime.now(UTC).isoformat(timespec="seconds")
        _write_json(self.manifest_path, self.manifest.model_dump_json(indent=2))
        report = {
            "strategy": self.strategy.value,
            "threshold": self.settings.dedup_threshold,
            "embedder": self.manifest.embedder,
            "duplicates": [d.model_dump() for d in self.manifest.duplicates.values()],
        }
        _write_json(self.dedup_report_path, json.dumps(report, indent=2))

    # ----- consistency -------------------------------------------------------------------

    def consistent(self) -> bool:
        ids = self.vectors.ids()
        return ids == self.bm25.ids() == set(self.manifest.chunks)

    def _reconcile(self) -> bool:
        """Make BM25 and the manifest match Chroma. Returns True if anything was repaired."""
        if self.consistent():
            return False
        chunks = self.vectors.all_chunks()
        self.bm25.clear()
        self.bm25.add((c.chunk_id, c.text) for c in chunks)
        self.manifest.chunks = {c.chunk_id: c.source for c in chunks}
        # A skipped duplicate whose original vanished must be re-evaluated on the next sync.
        self.manifest.duplicates = {
            cid: dup
            for cid, dup in self.manifest.duplicates.items()
            if dup.duplicate_of in self.manifest.chunks
        }
        self._save()
        return True

    def check_embedder(self, embedder: Embedder | None = None) -> None:
        """Raise if ``embedder`` (default: ours) differs from the one that built the index."""
        name = (embedder or self.embedder).name
        built_with = self.manifest.embedder
        if built_with is not None and built_with != name:
            raise IndexMismatchError(
                f"The {self.strategy.value} index was built with {built_with!r} but the current "
                f"embedder is {name!r}. Run `hyrag ingest` to rebuild it."
            )

    # ----- writes ------------------------------------------------------------------------

    @property
    def chunker(self) -> Chunker:
        if self._chunker is None:
            self._chunker = get_chunker(self.strategy, self.settings, embedder=self.embedder)
        return self._chunker

    def reset(self) -> None:
        self.vectors.reset()
        self.bm25.clear()
        self.manifest = Manifest(strategy=self.strategy)
        self._save()

    def sync(self, docs: Sequence[Document]) -> SyncReport:
        """Bring both indexes in line with ``docs``: add new chunks, drop stale ones."""
        report = SyncReport(strategy=self.strategy, repaired=self.repaired)
        if self.manifest.embedder not in (None, self.embedder.name):
            self.reset()
            report.rebuilt = True

        desired = {c.chunk_id: c for c in self.chunker.chunk(docs)}
        indexed = set(self.manifest.chunks)
        to_remove = indexed - desired.keys()
        stale_dups = {d for d in self.manifest.duplicates if d not in desired}
        revive = {
            d
            for d, dup in self.manifest.duplicates.items()
            if dup.duplicate_of in to_remove and d in desired
        }
        known = indexed | set(self.manifest.duplicates)
        to_add = [c for cid, c in desired.items() if cid not in known or cid in revive]

        if to_remove:
            self.vectors.delete(to_remove)
            self.bm25.remove(to_remove)
            for cid in to_remove:
                self.manifest.chunks.pop(cid, None)
        for cid in stale_dups | revive:
            self.manifest.duplicates.pop(cid, None)

        kept: list[Chunk] = []
        if to_add:
            embeddings = self.embedder.embed_documents([c.text for c in to_add])
            keep_idx, duplicates = find_duplicates(
                to_add, embeddings, self.vectors, self.settings.dedup_threshold
            )
            kept = [to_add[i] for i in keep_idx]
            if kept:
                self.vectors.upsert(kept, embeddings[keep_idx])
                self.bm25.add((c.chunk_id, c.text) for c in kept)
            self.manifest.chunks.update({c.chunk_id: c.source for c in kept})
            self.manifest.duplicates.update({d.chunk_id: d for d in duplicates})
            report.new_duplicates = duplicates

        self.manifest.embedder = self.embedder.name
        self._save()
        self.repaired = False

        report.added = len(kept)
        report.removed = len(to_remove)
        report.unchanged = len(indexed - to_remove)
        report.total_chunks = len(self.manifest.chunks)
        report.total_duplicates = len(self.manifest.duplicates)
        return report

    # ----- reads -------------------------------------------------------------------------

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, Chunk]:
        return self.vectors.get(chunk_ids)

    def __len__(self) -> int:
        return len(self.manifest.chunks)
