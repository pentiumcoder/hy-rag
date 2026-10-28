"""Hybrid retrieval: dense + BM25, fused with weighted RRF, then cross-encoder reranked."""

from __future__ import annotations

from dataclasses import dataclass

from hyrag.config import Settings, get_settings
from hyrag.embeddings import Embedder, embedder_from_settings
from hyrag.index import IndexManager
from hyrag.models import ChunkStrategy, RetrievalMode, ScoredChunk
from hyrag.retrieval.fusion import reciprocal_rank_fusion
from hyrag.retrieval.rerank import Reranker, get_reranker


@dataclass
class _Candidate:
    chunk_id: str
    first_stage: float
    dense: tuple[int, float] | None = None  # (rank, cosine)
    sparse: tuple[int, float] | None = None  # (rank, bm25)
    rrf: float | None = None
    rerank: float | None = None


class HybridRetriever:
    """Phase 2 contract: ``retrieve(query, mode, k) -> list[ScoredChunk]``.

    * ``dense`` and ``sparse`` each fetch ``fusion_top_n`` candidates on their own, so
      every mode hands the reranker an equally deep pool and comparisons stay fair.
    * ``hybrid`` fetches ``dense_k`` + ``sparse_k`` and keeps the RRF top ``fusion_top_n``.
    """

    def __init__(
        self,
        index: IndexManager,
        embedder: Embedder,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.index = index
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or index.settings
        index.check_embedder(embedder)

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        strategy: ChunkStrategy | str | None = None,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> HybridRetriever:
        settings = settings or get_settings()
        embedder = embedder or embedder_from_settings(settings)
        if reranker is None:
            reranker = get_reranker(
                settings.reranker, settings.reranker_model, settings.model_cache_dir
            )
        index = IndexManager(settings, strategy or settings.chunk_strategy, embedder)
        return cls(index, embedder, reranker, settings)

    def retrieve(
        self,
        query: str,
        mode: RetrievalMode | str = RetrievalMode.HYBRID,
        k: int | None = None,
        rerank: bool = True,
    ) -> list[ScoredChunk]:
        mode = RetrievalMode(mode)
        k = k or self.settings.top_k
        s = self.settings
        single_depth = s.fusion_top_n

        dense_hits: list[tuple[str, float]] = []
        sparse_hits: list[tuple[str, float]] = []
        if mode is not RetrievalMode.SPARSE:
            depth = s.dense_k if mode is RetrievalMode.HYBRID else single_depth
            query_vec = self.embedder.embed_query(query)
            dense_hits = [
                (h.chunk_id, h.similarity) for h in self.index.vectors.query(query_vec, depth)
            ]
        if mode is not RetrievalMode.DENSE:
            depth = s.sparse_k if mode is RetrievalMode.HYBRID else single_depth
            sparse_hits = self.index.bm25.search(query, depth)

        dense_at = {cid: (rank, score) for rank, (cid, score) in enumerate(dense_hits, start=1)}
        sparse_at = {cid: (rank, score) for rank, (cid, score) in enumerate(sparse_hits, start=1)}

        if mode is RetrievalMode.HYBRID:
            fused = reciprocal_rank_fusion(
                {"dense": [c for c, _ in dense_hits], "sparse": [c for c, _ in sparse_hits]},
                weights={"dense": s.dense_weight, "sparse": s.sparse_weight},
                k=s.rrf_k,
            )[: s.fusion_top_n]
            candidates = [
                _Candidate(cid, score, dense_at.get(cid), sparse_at.get(cid), rrf=score)
                for cid, score in fused
            ]
        else:
            hits = dense_hits if mode is RetrievalMode.DENSE else sparse_hits
            candidates = [
                _Candidate(cid, score, dense_at.get(cid), sparse_at.get(cid)) for cid, score in hits
            ]

        chunks = self.index.get_chunks([c.chunk_id for c in candidates])
        candidates = [c for c in candidates if c.chunk_id in chunks]

        if rerank and self.reranker is not None and candidates:
            scores = self.reranker.score(query, [chunks[c.chunk_id].text for c in candidates])
            for cand, score in zip(candidates, scores, strict=True):
                cand.rerank = score
            # Stable sort: equal rerank scores keep their first-stage order.
            candidates.sort(key=lambda c: -(c.rerank or 0.0))

        results: list[ScoredChunk] = []
        for rank, cand in enumerate(candidates[:k], start=1):
            results.append(
                ScoredChunk(
                    chunk=chunks[cand.chunk_id],
                    rank=rank,
                    score=cand.rerank if cand.rerank is not None else cand.first_stage,
                    dense_rank=cand.dense[0] if cand.dense else None,
                    dense_score=cand.dense[1] if cand.dense else None,
                    sparse_rank=cand.sparse[0] if cand.sparse else None,
                    sparse_score=cand.sparse[1] if cand.sparse else None,
                    rrf_score=cand.rrf,
                    rerank_score=cand.rerank,
                )
            )
        return results
