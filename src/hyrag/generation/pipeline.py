"""The answer pipeline: retrieve, gate, generate, verify, score."""

from __future__ import annotations

import time
from collections.abc import Sequence

from hyrag.config import Settings, get_settings
from hyrag.generation.abstention import build_abstention
from hyrag.generation.claims import extract_claims, is_insufficient, parse_insufficient
from hyrag.generation.confidence import citation_coverage, combine, retrieval_confidence
from hyrag.generation.prompts import build_answer_messages
from hyrag.generation.stub import ExtractiveStubLLM
from hyrag.generation.verifier import CitationVerifier, label_claims
from hyrag.index import IndexManager
from hyrag.llm import CachedLLM, LLMClient, get_llm
from hyrag.models import (
    AbstentionReason,
    AnswerResponse,
    AskOverrides,
    ChunkStrategy,
    Citation,
    CitationCheck,
    Claim,
    RetrievalMode,
    ScoredChunk,
)
from hyrag.retrieval import HybridRetriever


def llm_from_settings(settings: Settings, role: str = "generator") -> LLMClient:
    """The generator or judge the pipeline uses: configured provider, wrapped in the disk cache."""
    if role == "judge":
        provider, model = settings.judge_provider or settings.llm_provider, settings.judge_model
    else:
        provider, model = settings.llm_provider, settings.generator_model
    if provider == "stub":
        return ExtractiveStubLLM()
    client = get_llm(
        provider,
        model,
        role=role,
        base_url=settings.ollama_base_url,
        timeout=settings.llm_timeout,
        num_ctx=settings.ollama_num_ctx,
    )
    if settings.llm_cache:
        return CachedLLM(client, settings.cache_dir / "llm.sqlite")
    return client


class _Clock:
    def __init__(self) -> None:
        self.start = self.last = time.perf_counter()
        self.timings: dict[str, float] = {}

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = round((now - self.last) * 1000, 1)
        self.last = now

    def done(self) -> dict[str, float]:
        self.timings["total"] = round((time.perf_counter() - self.start) * 1000, 1)
        return self.timings


class AnswerPipeline:
    """Phase 3 contract: ``ask(query, overrides) -> AnswerResponse``.

    1. Retrieve the top-k passages (hybrid by default).
    2. If retrieval confidence is below ``abstain_threshold``, return a structured abstention
       without calling the generator: a weak context is where hallucinations come from.
    3. Generate an answer that cites passages as ``[n]``; the generator may itself abstain.
    4. Verify every citation with the judge in one batched call.
    5. Score confidence from retrieval, citation coverage and completeness.
    """

    def __init__(
        self,
        retriever: HybridRetriever,
        generator: LLMClient,
        judge: LLMClient | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.settings = settings or retriever.settings
        self.generator = generator
        self.judge = judge
        self.default_strategy = retriever.index.strategy
        self._retrievers: dict[ChunkStrategy, HybridRetriever] = {
            retriever.index.strategy: retriever
        }

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        strategy: ChunkStrategy | str | None = None,
        *,
        retriever: HybridRetriever | None = None,
        generator: LLMClient | None = None,
        judge: LLMClient | None = None,
    ) -> AnswerPipeline:
        settings = settings or get_settings()
        retriever = retriever or HybridRetriever.from_settings(settings, strategy)
        generator = generator or llm_from_settings(settings, "generator")
        if judge is None and settings.verify_citations:
            judge = llm_from_settings(settings, "judge")
        return cls(retriever, generator, judge, settings)

    def retriever_for(self, strategy: ChunkStrategy) -> HybridRetriever:
        """Retrievers for other strategies share the embedder and reranker of the first one."""
        if strategy not in self._retrievers:
            base = self._retrievers[self.default_strategy]
            index = IndexManager(self.settings, strategy, base.embedder)
            self._retrievers[strategy] = HybridRetriever(
                index, base.embedder, base.reranker, self.settings
            )
        return self._retrievers[strategy]

    def ask(self, query: str, overrides: AskOverrides | None = None) -> AnswerResponse:
        response = self.generate(query, overrides)
        if response.abstained:
            return response
        return self.verify(response, overrides)

    def generate(self, query: str, overrides: AskOverrides | None = None) -> AnswerResponse:
        """Steps 1-3: retrieve, gate and generate. The answer is not verified yet.

        Split from :meth:`verify` so a batch job (the eval suite) can generate every answer
        while the generator is loaded and verify them all afterwards, instead of swapping
        the two models on the GPU for every question.
        """
        o = overrides or AskOverrides()
        s = self.settings
        mode = o.mode or RetrievalMode.HYBRID
        strategy = o.strategy or self.default_strategy
        threshold = s.abstain_threshold if o.abstain_threshold is None else o.abstain_threshold
        clock = _Clock()

        contexts = self.retriever_for(strategy).retrieve(
            query, mode, o.top_k or s.top_k, rerank=True if o.rerank is None else o.rerank
        )
        clock.lap("retrieve")
        retrieval, signal = retrieval_confidence(
            query, contexts, center=s.rerank_center, scale=s.rerank_scale
        )
        response = AnswerResponse(
            query=query,
            answer=None,
            abstained=True,
            confidence=combine(retrieval, 0.0, 0.0, signal),
            contexts=contexts,
            mode=mode,
            strategy=strategy,
            generator=self.generator.name,
        )

        if not contexts:
            return self._abstain(
                response,
                clock,
                "no_results",
                "The index returned no passages for this question.",
                generated=False,
            )
        if retrieval < threshold:
            return self._abstain(
                response,
                clock,
                "low_retrieval_confidence",
                f"The best passage matches the question too weakly (retrieval confidence "
                f"{retrieval:.2f} < {threshold:.2f}), so no answer was generated.",
                generated=False,
            )

        raw = self.generator.complete(
            build_answer_messages(query, contexts),
            temperature=s.llm_temperature,
            max_tokens=s.generator_max_tokens,
        ).strip()
        clock.lap("generate")
        if not raw or is_insufficient(raw):
            found, missing = parse_insufficient(raw)
            return self._abstain(
                response,
                clock,
                "insufficient_context",
                "The generator found that the retrieved passages do not answer the question.",
                found=found,
                missing=missing,
            )

        claims = extract_claims(raw)
        if not any(c.kind == "fact" for c in claims):
            # Every sentence says what the context lacks: an abstention in answer's clothing.
            return self._abstain(
                response,
                clock,
                "insufficient_context",
                "The generator found that the retrieved passages do not answer the question.",
                missing=[c.text for c in claims],
            )
        response.answer = raw
        response.abstained = False
        response.claims = claims
        response.timings_ms = clock.done()
        return response

    def verify(
        self, response: AnswerResponse, overrides: AskOverrides | None = None
    ) -> AnswerResponse:
        """Steps 4-6: verify citations, score confidence, and apply the post-answer policy.

        After verification the answer is still withdrawn (and kept as ``draft_answer``) if
        the judge finds the passages do not answer the question and no sentence is
        supported, or if the verified confidence is below ``answer_threshold``. Retrieval
        scores cannot tell a near-miss passage from an answering one (ADR-0003); the judge,
        which reads the passages, can (ADR-0004).
        """
        if response.abstained or response.answer is None:
            return response
        o = overrides or AskOverrides()
        s = self.settings
        verify = (s.verify_citations if o.verify is None else o.verify) and self.judge is not None
        floor = s.answer_threshold if o.answer_threshold is None else o.answer_threshold
        start = time.perf_counter()

        verifier = CitationVerifier(
            self.judge if verify else None,
            max_tokens=s.judge_max_tokens,
            temperature=s.llm_temperature,
        )
        verification = verifier.verify(
            response.query, response.answer, response.claims, response.contexts
        )
        claims = label_claims(response.claims, verification.checks)
        completeness = 0.5 if verification.completeness is None else verification.completeness
        timings = dict(response.timings_ms)
        if verify:
            timings["verify"] = round((time.perf_counter() - start) * 1000, 1)
            timings["total"] = round(timings.get("total", 0.0) + timings["verify"], 1)

        confidence = combine(
            response.confidence.retrieval,
            citation_coverage(claims),
            completeness,
            response.confidence.retrieval_signal,
            answerability=verification.answerability,
        )
        result = response.model_copy(
            update={
                "claims": claims,
                "checks": verification.checks,
                "citations": _citations(claims, verification.checks, response.contexts),
                "confidence": confidence,
                "warnings": _warnings(claims, verification.checks, len(response.contexts), verify),
                "judge": self.judge.name if verify and self.judge else None,
                "timings_ms": timings,
            }
        )
        if verification.error:
            result.warnings.append(f"Citation verification failed: {verification.error}")
        if not verify:
            return result

        supported = any(c.label == "SUPPORTED" for c in claims)
        if s.abstain_on_unanswerable and verification.answerability == 0.0 and not supported:
            return self._withdraw(
                result,
                "unanswerable",
                "The judge found that the retrieved passages do not answer the question, and "
                "no sentence of the draft answer is supported by them.",
            )
        if confidence.score < floor:
            return self._withdraw(
                result,
                "low_answer_confidence",
                f"The verified answer scored too low (confidence {confidence.score:.2f} < "
                f"{floor:.2f}) to show as an answer.",
            )
        return result

    def _withdraw(
        self, response: AnswerResponse, reason: AbstentionReason, message: str
    ) -> AnswerResponse:
        """Turn a generated answer into an abstention, keeping the draft for inspection."""
        missing = [c.text for c in response.claims if c.kind == "gap"]
        abstention = build_abstention(
            reason, message, response.query, response.contexts, missing=missing
        )
        abstention.draft_answer = response.answer
        return response.model_copy(
            update={"answer": None, "abstained": True, "abstention": abstention}
        )

    def _abstain(
        self,
        response: AnswerResponse,
        clock: _Clock,
        reason: AbstentionReason,
        message: str,
        *,
        generated: bool = True,
        found: Sequence[str] = (),
        missing: Sequence[str] = (),
    ) -> AnswerResponse:
        response.abstention = build_abstention(
            reason, message, response.query, response.contexts, found=found, missing=missing
        )
        if not generated:
            response.generator = None
        response.judge = None
        response.timings_ms = clock.done()
        return response


def _citations(
    claims: Sequence[Claim], checks: Sequence[CitationCheck], contexts: Sequence[ScoredChunk]
) -> list[Citation]:
    flagged = {c.citation for c in checks if c.label == "UNSUPPORTED"}
    numbers = dict.fromkeys(n for c in claims if c.kind == "fact" for n in c.citations)
    citations: list[Citation] = []
    for n in numbers:
        if not 1 <= n <= len(contexts):
            continue
        chunk = contexts[n - 1].chunk
        citations.append(
            Citation(
                number=n,
                chunk_id=chunk.chunk_id,
                source=chunk.source,
                title=chunk.title,
                section_heading=chunk.section_heading,
                page=chunk.page,
                flagged=n in flagged,
            )
        )
    return citations


def _warnings(
    claims: Sequence[Claim], checks: Sequence[CitationCheck], n_contexts: int, verified: bool
) -> list[str]:
    warnings: list[str] = []
    for n in sorted({c.citation for c in checks if not 1 <= c.citation <= n_contexts}):
        warnings.append(f"The answer cites passage [{n}], which does not exist.")
    uncited = [c.index for c in claims if c.label == "UNCITED"]
    if uncited:
        warnings.append(f"Sentence(s) {', '.join(map(str, uncited))} cite no passage.")
    unsupported: dict[int, list[CitationCheck]] = {}
    for check in checks:
        if check.label == "UNSUPPORTED" and 1 <= check.citation <= n_contexts:
            unsupported.setdefault(check.citation, []).append(check)
    for n, failed in sorted(unsupported.items()):
        sentences = ", ".join(str(c.claim) for c in failed)
        reason = next((c.reason for c in failed if c.reason), "")
        warnings.append(
            f"Passage [{n}] does not support sentence(s) {sentences}"
            + (f": {reason}" if reason else ".")
        )
    if not verified:
        warnings.append("Citations were not verified by the judge.")
    return warnings
