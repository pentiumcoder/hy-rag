"""Citation verification: does each cited passage actually support the claim citing it?

Every (claim, citation) pair goes to the judge in **one** JSON call rather than one call per
pair: on a local GPU each call costs seconds, and when the generator and judge do not both
fit in VRAM each call can also mean a model swap (ADR-0003). Citations of passages that do
not exist are rejected without asking the judge.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from hyrag.generation.prompts import JudgePair, build_judge_messages
from hyrag.llm import LLMClient, LLMError, complete_json
from hyrag.models import Chunk, CitationCheck, Claim, ClaimLabel, ScoredChunk, VerdictLabel

COMPLETENESS = {"COMPLETE": 1.0, "PARTIAL": 0.5, "INCOMPLETE": 0.0}
_RANK: dict[str, int] = {"SUPPORTED": 3, "PARTIAL": 2, "UNVERIFIED": 1, "UNSUPPORTED": 0}
_ALIASES: dict[str, VerdictLabel] = {
    "SUPPORTED": "SUPPORTED",
    "SUPPORTS": "SUPPORTED",
    "PARTIAL": "PARTIAL",
    "PARTIALLY_SUPPORTED": "PARTIAL",
    "UNSUPPORTED": "UNSUPPORTED",
    "NOT_SUPPORTED": "UNSUPPORTED",
    "CONTRADICTED": "UNSUPPORTED",
}


@dataclass
class Verification:
    checks: list[CitationCheck] = field(default_factory=list)
    completeness: float | None = None  # None when the judge was not asked
    error: str | None = None


def _normalize_label(value: Any) -> VerdictLabel | None:
    key = str(value or "").strip().upper().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key)


def _parse_verdicts(data: dict[str, Any]) -> dict[int, tuple[VerdictLabel, str]]:
    verdicts: dict[int, tuple[VerdictLabel, str]] = {}
    items = data.get("checks")
    if not isinstance(items, list):
        return verdicts
    for item in items:
        if not isinstance(item, dict):
            continue
        try:
            check_id = int(item.get("id"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        label = _normalize_label(item.get("label"))
        if label is not None:
            verdicts[check_id] = (label, str(item.get("reason") or "").strip())
    return verdicts


class CitationVerifier:
    """Checks every citation in an answer. With ``judge=None`` only the free checks run."""

    def __init__(
        self, judge: LLMClient | None, *, max_tokens: int = 1024, temperature: float = 0.0
    ) -> None:
        self.judge = judge
        self.max_tokens = max_tokens
        self.temperature = temperature

    def verify(
        self,
        question: str,
        answer: str,
        claims: Sequence[Claim],
        contexts: Sequence[ScoredChunk],
    ) -> Verification:
        checks: list[CitationCheck] = []
        pairs: list[JudgePair] = []
        claim_of: dict[int, int] = {}  # pair id -> claim index
        for claim in claims:
            if claim.kind != "fact":
                continue
            for number in claim.citations:
                if not 1 <= number <= len(contexts):
                    checks.append(
                        CitationCheck(
                            claim=claim.index,
                            citation=number,
                            label="UNSUPPORTED",
                            reason=f"there is no passage [{number}]",
                        )
                    )
                    continue
                pair = JudgePair(len(pairs) + 1, claim.text, number)
                pairs.append(pair)
                claim_of[pair.id] = claim.index

        result = Verification()
        if pairs and self.judge is None:
            for p in pairs:
                checks.append(
                    CitationCheck(claim=claim_of[p.id], citation=p.citation, label="UNVERIFIED")
                )
        elif pairs and self.judge is not None:
            passages: dict[int, Chunk] = {p.citation: contexts[p.citation - 1].chunk for p in pairs}
            messages = build_judge_messages(question, answer, passages, pairs)
            try:
                data = complete_json(
                    self.judge, messages, temperature=self.temperature, max_tokens=self.max_tokens
                )
            except LLMError as exc:
                data = {}
                result.error = str(exc)
            verdicts = _parse_verdicts(data)
            for p in pairs:
                label, reason = verdicts.get(p.id, ("UNVERIFIED", "the judge gave no verdict"))
                if result.error:
                    reason = "verification failed"
                checks.append(
                    CitationCheck(
                        claim=claim_of[p.id], citation=p.citation, label=label, reason=reason
                    )
                )
            completeness = str(data.get("completeness") or "").strip().upper()
            result.completeness = COMPLETENESS.get(completeness)

        result.checks = sorted(checks, key=lambda c: (c.claim, c.citation))
        return result


def label_claims(claims: Sequence[Claim], checks: Sequence[CitationCheck]) -> list[Claim]:
    """Give each fact claim the best verdict over its citations, or ``UNCITED`` if it has none."""
    best: dict[int, VerdictLabel] = {}
    for check in checks:
        current = best.get(check.claim)
        if current is None or _RANK[check.label] > _RANK[current]:
            best[check.claim] = check.label
    labelled: list[Claim] = []
    for claim in claims:
        label: ClaimLabel | None = None
        if claim.kind == "fact":
            label = best.get(claim.index, "UNCITED") if claim.citations else "UNCITED"
        labelled.append(claim.model_copy(update={"label": label}))
    return labelled
