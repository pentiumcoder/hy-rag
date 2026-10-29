"""Structured abstention: say what was found, what is missing and where to look next.

An abstention is an answer too. A bare "I don't know" leaves the user nowhere, so every
abstention names the documents that came closest and the parts of the question that the
retrieved passages do (and do not) mention.
"""

from __future__ import annotations

from collections.abc import Sequence

from hyrag.generation.confidence import lexical_coverage
from hyrag.models import Abstention, AbstentionReason, DocumentRef, ScoredChunk


def suggested_documents(contexts: Sequence[ScoredChunk], limit: int = 3) -> list[DocumentRef]:
    """The best-ranked passage of each of the first ``limit`` distinct documents."""
    refs: list[DocumentRef] = []
    seen: set[str] = set()
    for sc in contexts:
        if sc.chunk.source in seen:
            continue
        seen.add(sc.chunk.source)
        refs.append(
            DocumentRef(
                source=sc.chunk.source,
                title=sc.chunk.title,
                section_heading=sc.chunk.section_heading,
                page=sc.chunk.page,
                score=sc.score,
            )
        )
        if len(refs) == limit:
            break
    return refs


def build_abstention(
    reason: AbstentionReason,
    message: str,
    query: str,
    contexts: Sequence[ScoredChunk],
    *,
    found: Sequence[str] = (),
    missing: Sequence[str] = (),
) -> Abstention:
    """``found``/``missing`` from the generator win; otherwise they come from term overlap."""
    terms_found, terms_missing = lexical_coverage(query, contexts)
    if not found and terms_found:
        found = [f"Passages that mention: {', '.join(terms_found)}."]
    if not missing:
        missing = (
            [f"No retrieved passage mentions: {', '.join(terms_missing)}."]
            if terms_missing
            else ["No retrieved passage answers the question directly."]
        )
    return Abstention(
        reason=reason,
        message=message,
        found=list(found),
        missing=list(missing),
        suggested_documents=suggested_documents(contexts),
    )
