"""Turning a generated answer into checkable claims with ``[n]`` citations.

A claim is one sentence or bullet. Citation markers may sit before or after the final full
stop (``... 5 days [2].`` or ``... 5 days. [2]``); both attach to the sentence they follow.
"""

from __future__ import annotations

import re
from typing import Literal

from hyrag.models import Claim

_CITE = re.compile(r"\[(\d+(?:\s*[,\-–]\s*\d+)*)\]")
# A sentence ends at . ! or ?, plus any citation groups after it and a stray stop after
# those ("... travel. [1][2]."), as long as whitespace or the end of the line follows.
_SENTENCE_END = re.compile(r"[.!?]+(?:\s*\[\d+(?:\s*[,\-–]\s*\d+)*\])*[.!?]*(?=\s+\S|\s*$)")
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)])\s+")
_HEADING = re.compile(r"^\s*#+\s*")
# Sentences about what the context lacks ("The context does not cover ...", "X is not
# mentioned in the passages"). Small models often cite passages on these too.
_SOURCE = r"\b(?:context|passages?|documents?|documentation|sources?)\b"
_GAP = re.compile(
    rf"{_SOURCE}.{{0,60}}?\b(?:does not|do not|doesn't|don't|did not)\b.{{0,30}}?"
    r"\b(?:cover|mention|say|state|include|contain|specify|provide|describe|address|detail)"
    rf"|\bnot (?:mentioned|covered|specified|provided|stated|described|included)\b.{{0,40}}?"
    rf"{_SOURCE}"
    r"|\b(?:there is|there's) no (?:information|mention|detail)",
    re.IGNORECASE,
)
_INSUFFICIENT = re.compile(r"INSUFFICIENT[_ ]CONTEXT", re.IGNORECASE)
_NOTHING = {"", "none", "nothing", "n/a", "-"}


def _expand(group: str) -> list[int]:
    numbers: list[int] = []
    for part in re.split(r"\s*,\s*", group.strip()):
        bounds = re.split(r"\s*[\-–]\s*", part)
        if len(bounds) == 2 and bounds[0].isdigit() and bounds[1].isdigit():
            lo, hi = int(bounds[0]), int(bounds[1])
            numbers.extend(range(lo, hi + 1) if 0 < hi - lo < 20 else [lo, hi])
        elif part.isdigit():
            numbers.append(int(part))
    return numbers


def find_citations(text: str) -> list[int]:
    """Cited passage numbers in order of first appearance; ``[1-3]`` expands to 1, 2, 3."""
    seen: dict[int, None] = {}
    for match in _CITE.finditer(text):
        for n in _expand(match.group(1)):
            seen.setdefault(n, None)
    return list(seen)


def strip_citations(text: str) -> str:
    text = _CITE.sub("", text)
    text = re.sub(r"\s+([.,;:!?])", r"\1", text)
    text = re.sub(r"([.!?])\.+", r"\1", text)  # "day. [1]." leaves "day.."
    return re.sub(r"\s{2,}", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    """Split an answer into sentences and bullet items, keeping each one's citations."""
    sentences: list[str] = []
    for raw_line in text.splitlines():
        line = _HEADING.sub("", _BULLET.sub("", raw_line)).replace("**", "").strip()
        if not line:
            continue
        start = 0
        for match in _SENTENCE_END.finditer(line):
            piece = line[start : match.end()].strip()
            if piece:
                sentences.append(piece)
            start = match.end()
        rest = line[start:].strip()
        if rest:
            sentences.append(rest)
    return sentences


def extract_claims(answer: str) -> list[Claim]:
    """One claim per sentence or bullet.

    Small models often put a single citation group after the last item of a list or
    paragraph and mean it for the whole block. When that group is the block's only citation,
    it is copied to the block's other factual sentences (``inherited=True``). Inheriting is
    safe because every copied (claim, passage) pair still goes to the judge.
    """
    claims: list[Claim] = []
    for block in re.split(r"\n\s*\n", answer):
        block_claims: list[Claim] = []
        for sentence in split_sentences(block):
            citations = find_citations(sentence)
            text = strip_citations(sentence)
            if not re.search(r"\w", text):
                continue
            if not citations and (text.endswith(":") or len(text.split()) < 3):
                continue  # a label such as "Steps:" rather than a statement
            kind: Literal["fact", "gap"] = "gap" if _GAP.search(text) else "fact"
            index = len(claims) + len(block_claims) + 1
            block_claims.append(Claim(index=index, text=text, citations=citations, kind=kind))
        facts = [c for c in block_claims if c.kind == "fact"]
        if len(facts) > 1 and facts[-1].citations and not any(c.citations for c in facts[:-1]):
            for claim in facts[:-1]:
                claim.citations = list(facts[-1].citations)
                claim.inherited = True
        claims.extend(block_claims)
    return claims


def is_insufficient(answer: str) -> bool:
    return bool(_INSUFFICIENT.search(answer))


def parse_insufficient(answer: str) -> tuple[list[str], list[str]]:
    """The ``Found:`` and ``Missing:`` lines of an ``INSUFFICIENT_CONTEXT`` reply."""
    found: list[str] = []
    missing: list[str] = []
    for line in answer.splitlines():
        label, _, value = line.strip().partition(":")
        value = strip_citations(value)
        if value.lower().rstrip(".") in _NOTHING:
            continue
        if label.strip().lower() == "found":
            found.append(value)
        elif label.strip().lower() == "missing":
            missing.append(value)
    return found, missing
