"""Text clean-up applied to every loader's output so downstream code sees one shape."""

from __future__ import annotations

import re
import unicodedata

_INLINE_WS = re.compile(r"[ \t\f\v ]+")
_MANY_BLANK_LINES = re.compile(r"\n{3,}")
# "inges-\ntion" -> "ingestion" (PDF line-wrap hyphenation). Leaves "ingest-\nService" alone.
_SOFT_HYPHEN_BREAK = re.compile(r"(?<=[a-z])-\n(?=[a-z])")
_MD_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__)(.+?)\1")
_MD_INLINE_CODE = re.compile(r"`([^`\n]+)`")


def normalize_text(text: str) -> str:
    """Unicode-normalize, unify newlines, collapse runs of whitespace, trim."""
    text = unicodedata.normalize("NFKC", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\x00", "")
    text = _SOFT_HYPHEN_BREAK.sub("", text)
    lines = [_INLINE_WS.sub(" ", line).strip() for line in text.split("\n")]
    text = "\n".join(lines)
    text = _MANY_BLANK_LINES.sub("\n\n", text)
    return text.strip()


def strip_markdown_inline(text: str) -> str:
    """Drop inline markdown syntax while keeping its readable content.

    Identifiers inside backticks (``ingest.max_batch_size``) are kept verbatim because
    exact-term retrieval depends on them.
    """
    text = _MD_LINK.sub(r"\1", text)
    text = _MD_EMPHASIS.sub(r"\2", text)
    text = _MD_INLINE_CODE.sub(r"\1", text)
    return text
