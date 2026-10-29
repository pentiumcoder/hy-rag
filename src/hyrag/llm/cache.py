"""Disk cache for LLM replies, keyed by (model, messages, options) hash.

Local generation takes seconds per call, and the eval suite asks the same questions of the
same contexts again and again while other knobs change. Only deterministic calls
(temperature 0) are cached, so a sampled reply is never replayed as if it were fixed.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Sequence
from pathlib import Path

from hyrag.llm.base import LLMClient, Message
from hyrag.models import stable_hash


class CachedLLM:
    def __init__(self, inner: LLMClient, path: Path) -> None:
        self.inner = inner
        self.name = inner.name
        self.hits = 0
        self.misses = 0
        path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(path), check_same_thread=False)
        self._db.execute(
            "CREATE TABLE IF NOT EXISTS llm (key TEXT PRIMARY KEY, reply TEXT NOT NULL)"
        )
        self._db.commit()

    def _key(self, messages: Sequence[Message], json_mode: bool, max_tokens: int | None) -> str:
        payload = json.dumps(list(messages), sort_keys=True, ensure_ascii=False)
        return stable_hash(self.name, payload, str(json_mode), str(max_tokens), length=32)

    def complete(
        self,
        messages: Sequence[Message],
        *,
        json_mode: bool = False,
        temperature: float = 0.0,
        max_tokens: int | None = None,
    ) -> str:
        if temperature != 0:
            self.misses += 1
            return self.inner.complete(
                messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens
            )
        key = self._key(messages, json_mode, max_tokens)
        with self._lock:
            row = self._db.execute("SELECT reply FROM llm WHERE key = ?", (key,)).fetchone()
        if row is not None:
            self.hits += 1
            return str(row[0])
        reply = self.inner.complete(
            messages, json_mode=json_mode, temperature=temperature, max_tokens=max_tokens
        )
        with self._lock:
            self._db.execute("INSERT OR REPLACE INTO llm (key, reply) VALUES (?, ?)", (key, reply))
            self._db.commit()
            self.misses += 1
        return reply

    def close(self) -> None:
        self._db.close()
