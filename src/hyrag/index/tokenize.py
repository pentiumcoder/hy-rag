"""Identifier-aware tokenizer for BM25.

Internal docs are full of exact identifiers (``ingest.max_batch_size``, ``NW-4012``,
``X-NW-Signature``) that generic tokenizers shred into common words. This one keeps each
identifier whole *and* emits its parts, so an exact-key query matches the key itself while
a plain-English query ("max batch size") still matches its pieces.
"""

from __future__ import annotations

import re

_TOKEN = re.compile(r"[a-z0-9_]+(?:[.\-/:][a-z0-9_]+)*")
_PARTS = re.compile(r"[._\-/:]+")

STOPWORDS = frozenset(
    {
        *("a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "can", "could"),
        *("did", "do", "does", "for", "from", "had", "has", "have", "how", "i", "if", "in"),
        *("into", "is", "it", "its", "may", "me", "my", "no", "not", "of", "on", "or", "our"),
        *("should", "so", "than", "that", "the", "their", "them", "then", "there", "these"),
        *("they", "this", "those", "to", "us", "was", "we", "were", "what", "when", "where"),
        *("which", "who", "why", "will", "with", "would", "you", "your"),
    }
)


def tokenize(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN.findall(text.lower()):
        parts = [p for p in _PARTS.split(match) if p]
        if len(parts) > 1:
            tokens.append(match)
            tokens.extend(p for p in parts if p not in STOPWORDS)
        elif parts and parts[0] not in STOPWORDS:
            tokens.append(parts[0])
    return tokens
