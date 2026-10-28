"""Disk cache for embeddings, keyed by (model, kind, text) hash.

Re-chunking with a different strategy produces many identical texts (and re-indexing an
unchanged corpus produces all of them), so the cache makes re-ingestion nearly free.
SQLite keeps it to a single file with atomic writes and no extra dependency.
"""

from __future__ import annotations

import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from hyrag.embeddings.base import Embedder, Matrix, Vector
from hyrag.models import stable_hash


class CachedEmbedder:
    def __init__(self, inner: Embedder, path: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.dim = inner.dim
        self.hits = 0
        self.misses = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute("CREATE TABLE IF NOT EXISTS emb (key TEXT PRIMARY KEY, vec BLOB NOT NULL)")
        self._db.commit()

    def _key(self, kind: str, text: str) -> str:
        return stable_hash(self.name, kind, text, length=32)

    def _lookup(self, keys: list[str]) -> dict[str, Vector]:
        found: dict[str, Vector] = {}
        for start in range(0, len(keys), 500):  # stay under SQLite's variable limit
            batch = keys[start : start + 500]
            marks = ",".join("?" * len(batch))
            rows = self._db.execute(f"SELECT key, vec FROM emb WHERE key IN ({marks})", batch)
            for key, blob in rows:
                found[key] = np.frombuffer(blob, dtype=np.float32)
        return found

    def _embed(self, kind: str, texts: Sequence[str]) -> Matrix:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        keys = [self._key(kind, t) for t in texts]
        with self._lock:
            found = self._lookup(list(dict.fromkeys(keys)))
            missing = [i for i, k in enumerate(keys) if k not in found]
            todo = list(dict.fromkeys(texts[i] for i in missing))
            if todo:
                if kind == "q":
                    fresh = np.stack([self.inner.embed_query(t) for t in todo])
                else:
                    fresh = self.inner.embed_documents(todo)
                new = {self._key(kind, t): fresh[j] for j, t in enumerate(todo)}
                self._db.executemany(
                    "INSERT OR REPLACE INTO emb (key, vec) VALUES (?, ?)",
                    [(k, v.astype(np.float32).tobytes()) for k, v in new.items()],
                )
                self._db.commit()
                found.update(new)
            self.misses += len(todo)  # texts actually sent to the model
            self.hits += len(keys) - len(todo)
        return np.stack([found[k] for k in keys]).astype(np.float32)

    def embed_documents(self, texts: Sequence[str]) -> Matrix:
        return self._embed("d", texts)

    def embed_query(self, text: str) -> Vector:
        vector: Vector = self._embed("q", [text])[0]
        return vector

    def close(self) -> None:
        self._db.close()
