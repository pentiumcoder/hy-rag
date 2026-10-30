"""Domain models shared by every layer of the pipeline."""

from __future__ import annotations

import hashlib
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, computed_field

DocFormat = Literal["markdown", "text", "html", "pdf"]


class ChunkStrategy(StrEnum):
    FIXED = "fixed"
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RetrievalMode(StrEnum):
    DENSE = "dense"
    SPARSE = "sparse"
    HYBRID = "hybrid"


def stable_hash(*parts: str, length: int = 16) -> str:
    """Deterministic short hash; Python's built-in hash() is salted per process."""
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()
    return digest[:length]


class Section(BaseModel):
    """A contiguous run of text under one heading (and, for PDFs, on one page)."""

    heading: str | None = None
    text: str
    page: int | None = None
    level: int = 0


class Document(BaseModel):
    """A source file normalized into plain-text sections plus metadata."""

    doc_id: str
    source: str  # POSIX path relative to the corpus root
    title: str
    format: DocFormat
    sections: list[Section]
    doc_hash: str  # hash of the raw bytes, used to skip unchanged files
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "\n\n".join(s.text for s in self.sections if s.text)

    @property
    def char_count(self) -> int:
        return sum(len(s.text) for s in self.sections)


class Chunk(BaseModel):
    """A retrievable unit of text. The id is stable for identical inputs."""

    chunk_id: str
    doc_id: str
    source: str
    title: str
    text: str
    strategy: ChunkStrategy
    chunk_index: int
    section_heading: str | None = None
    page: int | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def char_count(self) -> int:
        return len(self.text)

    @classmethod
    def create(
        cls,
        doc: Document,
        strategy: ChunkStrategy,
        chunk_index: int,
        text: str,
        section_heading: str | None = None,
        page: int | None = None,
    ) -> Chunk:
        return cls(
            chunk_id=stable_hash(doc.source, strategy.value, str(chunk_index), text),
            doc_id=doc.doc_id,
            source=doc.source,
            title=doc.title,
            text=text,
            strategy=strategy,
            chunk_index=chunk_index,
            section_heading=section_heading,
            page=page,
        )


class ScoredChunk(BaseModel):
    """A retrieved chunk plus every score that contributed to its final rank.

    Ranks are 1-based; a ``None`` score means that stage did not see the chunk
    (e.g. ``sparse_*`` is ``None`` for a chunk only the dense retriever found).
    """

    chunk: Chunk
    rank: int
    score: float  # what the final order is sorted by: rerank score if reranked, else first stage
    dense_score: float | None = None  # cosine similarity
    dense_rank: int | None = None
    sparse_score: float | None = None  # BM25
    sparse_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None  # cross-encoder logit


VerdictLabel = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNVERIFIED"]
ClaimLabel = Literal["SUPPORTED", "PARTIAL", "UNSUPPORTED", "UNVERIFIED", "UNCITED"]
AbstentionReason = Literal[
    "no_results",  # retrieval returned nothing
    "low_retrieval_confidence",  # gated before generation
    "insufficient_context",  # the generator declined
    "unanswerable",  # the judge found the passages do not answer it (after generation)
    "low_answer_confidence",  # the verified answer scored below answer_threshold
]


class AskOverrides(BaseModel):
    """Per-request knobs; ``None`` means use the configured setting."""

    mode: RetrievalMode | None = None
    strategy: ChunkStrategy | None = None
    top_k: int | None = Field(default=None, ge=1, le=20)
    rerank: bool | None = None
    verify: bool | None = None
    abstain_threshold: float | None = Field(default=None, ge=0, le=1)
    answer_threshold: float | None = Field(default=None, ge=0, le=1)


class Claim(BaseModel):
    """One sentence (or bullet) of the answer and the passages it cites.

    ``kind="gap"`` marks a sentence stating what the context does *not* cover; it needs no
    citation and is left out of citation coverage.
    """

    index: int  # 1-based position in the answer
    text: str  # with the [n] markers removed
    citations: list[int] = Field(default_factory=list)
    kind: Literal["fact", "gap"] = "fact"
    inherited: bool = False  # citations copied from the citation group closing its block
    label: ClaimLabel | None = None  # best verdict over its citations; None for gaps


class CitationCheck(BaseModel):
    """The judge's verdict on whether passage ``citation`` supports claim ``claim``."""

    claim: int
    citation: int
    label: VerdictLabel
    reason: str = ""


class Citation(BaseModel):
    """A passage the answer cites, resolved to its source. ``flagged`` if any use of it failed."""

    number: int
    chunk_id: str
    source: str
    title: str
    section_heading: str | None = None
    page: int | None = None
    flagged: bool = False


class Confidence(BaseModel):
    """``score = 0.4 * retrieval + 0.4 * citation_coverage + 0.2 * completeness``.

    ``answerability`` (the judge's view of whether the passages answer the question) is not
    part of the score; it feeds the post-answer abstention rule instead.
    """

    score: float
    retrieval: float
    citation_coverage: float
    completeness: float
    retrieval_signal: Literal["rerank", "lexical", "none"]
    answerability: float | None = None  # 1 yes, 0.5 partly, 0 no; None when not judged


class DocumentRef(BaseModel):
    source: str
    title: str
    section_heading: str | None = None
    page: int | None = None
    score: float | None = None


class Abstention(BaseModel):
    """What the system says instead of guessing: what it found, what is missing, where to look."""

    reason: AbstentionReason
    message: str
    found: list[str] = Field(default_factory=list)
    missing: list[str] = Field(default_factory=list)
    suggested_documents: list[DocumentRef] = Field(default_factory=list)
    draft_answer: str | None = None  # the withdrawn answer, when abstaining after generation


class AnswerResponse(BaseModel):
    """Phase 3 contract: everything ``AnswerPipeline.ask`` knows about one question."""

    query: str
    answer: str | None  # keeps the [n] markers; None when abstained
    abstained: bool
    abstention: Abstention | None = None
    claims: list[Claim] = Field(default_factory=list)
    checks: list[CitationCheck] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    confidence: Confidence
    contexts: list[ScoredChunk] = Field(default_factory=list)  # [n] is contexts[n - 1]
    mode: RetrievalMode
    strategy: ChunkStrategy
    generator: str | None = None
    judge: str | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
