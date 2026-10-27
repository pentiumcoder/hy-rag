"""Keeps raw uploads next to their processed form so the corpus can be re-indexed
(or re-parsed with an improved loader) without anyone re-uploading files."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from hyrag.ingestion.loaders import iter_corpus_files, load_document
from hyrag.models import Document


@dataclass
class IngestReport:
    added: list[str] = field(default_factory=list)
    updated: list[str] = field(default_factory=list)
    unchanged: list[str] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)  # parsed but no extractable text

    @property
    def changed(self) -> list[str]:
        return self.added + self.updated


class DocumentStore:
    """``data/raw/<source>`` holds originals, ``data/processed/<doc_id>.json`` the parse."""

    MANIFEST = "manifest.json"

    def __init__(self, data_dir: Path) -> None:
        self.raw_dir = Path(data_dir) / "raw"
        self.processed_dir = Path(data_dir) / "processed"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    # manifest: source -> {"doc_id": ..., "doc_hash": ...}
    def _manifest_path(self) -> Path:
        return self.processed_dir / self.MANIFEST

    def manifest(self) -> dict[str, dict[str, str]]:
        path = self._manifest_path()
        if not path.exists():
            return {}
        data: dict[str, dict[str, str]] = json.loads(path.read_text(encoding="utf-8"))
        return data

    def _write_manifest(self, manifest: dict[str, dict[str, str]]) -> None:
        self._manifest_path().write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )

    def ingest_directory(self, root: Path) -> IngestReport:
        """Copy and parse every supported file under ``root``; unchanged files are skipped."""
        root = Path(root)
        return self.ingest_files(iter_corpus_files(root), root)

    def ingest_files(self, paths: list[Path], root: Path) -> IngestReport:
        report = IngestReport()
        manifest = self.manifest()
        for path in paths:
            doc = load_document(path, root)
            previous = manifest.get(doc.source)
            if previous and previous["doc_hash"] == doc.doc_hash:
                report.unchanged.append(doc.source)
                continue
            if not doc.sections:
                report.skipped.append(doc.source)
                continue
            raw_target = self.raw_dir / doc.source
            raw_target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, raw_target)
            self._processed_path(doc.doc_id).write_text(doc.model_dump_json(), encoding="utf-8")
            manifest[doc.source] = {"doc_id": doc.doc_id, "doc_hash": doc.doc_hash}
            (report.updated if previous else report.added).append(doc.source)
        self._write_manifest(manifest)
        return report

    def _processed_path(self, doc_id: str) -> Path:
        return self.processed_dir / f"{doc_id}.json"

    def load_documents(self) -> list[Document]:
        """Read the processed documents (no parsing), sorted by source."""
        docs = [
            Document.model_validate_json(self._processed_path(entry["doc_id"]).read_text("utf-8"))
            for entry in self.manifest().values()
        ]
        return sorted(docs, key=lambda d: d.source)

    def reprocess(self) -> list[Document]:
        """Re-parse every stored raw file, e.g. after a loader improvement."""
        manifest = self.manifest()
        docs: list[Document] = []
        for source in sorted(manifest):
            doc = load_document(self.raw_dir / source, self.raw_dir)
            self._processed_path(doc.doc_id).write_text(doc.model_dump_json(), encoding="utf-8")
            manifest[source] = {"doc_id": doc.doc_id, "doc_hash": doc.doc_hash}
            docs.append(doc)
        self._write_manifest(manifest)
        return docs

    def remove(self, source: str) -> bool:
        manifest = self.manifest()
        entry = manifest.pop(source, None)
        if entry is None:
            return False
        self._processed_path(entry["doc_id"]).unlink(missing_ok=True)
        (self.raw_dir / source).unlink(missing_ok=True)
        self._write_manifest(manifest)
        return True
