"""Weighted Reciprocal Rank Fusion.

RRF merges ranked lists using only ranks: ``score(d) = sum_i w_i / (k + rank_i(d))``.
BM25 scores are unbounded and cosine similarities are not, so fusing raw scores needs
fragile per-query normalization; ranks need none. ``k`` (60 by convention) damps the
advantage of the very top ranks, so agreement between retrievers beats a single top hit.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]],
    weights: Mapping[str, float] | None = None,
    k: int = 60,
) -> list[tuple[str, float]]:
    """Fuse ranked id lists (best first). Returns ``(id, score)`` sorted best first.

    Missing weights default to 1.0. Ties break on the best single rank, then on id, so the
    output is deterministic.
    """
    scores: dict[str, float] = {}
    best_rank: dict[str, int] = {}
    for name, ids in rankings.items():
        weight = 1.0 if weights is None else weights.get(name, 1.0)
        for rank, item in enumerate(ids, start=1):
            scores[item] = scores.get(item, 0.0) + weight / (k + rank)
            best_rank[item] = min(best_rank.get(item, rank), rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], best_rank[pair[0]], pair[0]))
