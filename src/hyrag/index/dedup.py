"""Near-duplicate filtering before insert.

Internal docs copy-paste heavily (the same deploy-freeze paragraph lives in the handbook and
the release process). Indexing both wastes context slots: the top-5 fills with the same
text twice. Each new chunk is compared with its nearest *indexed* neighbour and with chunks
accepted earlier in the same batch; above the threshold it is skipped and reported.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from pydantic import BaseModel

from hyrag.embeddings.base import Matrix
from hyrag.index.vector_store import VectorStore
from hyrag.models import Chunk


class Duplicate(BaseModel):
    chunk_id: str
    source: str
    duplicate_of: str
    duplicate_of_source: str
    similarity: float


def find_duplicates(
    chunks: Sequence[Chunk],
    embeddings: Matrix,
    store: VectorStore,
    threshold: float,
) -> tuple[list[int], list[Duplicate]]:
    """Return indices of chunks to keep, plus a record for each skipped duplicate.

    Order matters: the earlier chunk (or the one already indexed) is the original.
    """
    nearest = store.query_many(embeddings, k=1)
    kept: list[int] = []
    duplicates: list[Duplicate] = []
    for i, chunk in enumerate(chunks):
        best_sim, best_id, best_source = -1.0, "", ""
        if nearest[i]:
            hit = nearest[i][0]
            if hit.chunk_id != chunk.chunk_id:
                best_sim, best_id, best_source = hit.similarity, hit.chunk_id, hit.source
        if kept:
            sims = embeddings[kept] @ embeddings[i]
            j = int(np.argmax(sims))
            if float(sims[j]) > best_sim:
                best_sim = float(sims[j])
                best_id, best_source = chunks[kept[j]].chunk_id, chunks[kept[j]].source
        if best_sim > threshold:
            duplicates.append(
                Duplicate(
                    chunk_id=chunk.chunk_id,
                    source=chunk.source,
                    duplicate_of=best_id,
                    duplicate_of_source=best_source,
                    similarity=round(best_sim, 4),
                )
            )
        else:
            kept.append(i)
    return kept, duplicates
