"""Prompt templates for grounded answering and batched citation checking.

Both prompts number passages the same way (``[n]`` cites the n-th retrieved chunk), so the
generator's citations, the judge's checks and the response's ``contexts`` all line up.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from hyrag.llm import Message, system, user
from hyrag.models import Chunk, ScoredChunk

INSUFFICIENT = "INSUFFICIENT_CONTEXT"

ANSWER_SYSTEM = f"""\
You answer questions about Northwind Cloud's internal documentation. Use ONLY the context \
passages you are given; each one is wrapped in <passage id="n"> tags.

Rules:
1. If the passages contain information that answers the question, answer it, even when they \
use different words than the question.
2. End every sentence with the id of the passage that states it, in square brackets. Use two \
ids, like [1][3], when a sentence relies on two passages. Check the id on the passage you \
took the fact from. Every bullet point and numbered step needs its own citation too. The \
format (not the content) looks like this:
- <first step, taken from passage 2> [2].
- <second step, combining passages 2 and 4> [2][4].
3. Never state anything the passages do not say, and copy exact values (numbers, names, \
config keys, error codes, commands) as written.
4. Be concise: at most 6 sentences or bullet points, with no preamble.
5. If the passages answer only part of the question, answer that part, then add one final \
sentence starting "The context does not cover" naming the missing part.
6. If no passage is relevant to the question, reply with exactly these three lines and \
nothing else:
{INSUFFICIENT}
Found: <one sentence on what the passages do cover>
Missing: <one sentence on the information that would be needed>"""

JUDGE_SYSTEM = """\
You are a strict fact-checker for a documentation assistant. Each check pairs one claim from \
an answer with one passage the answer cited for it. Decide whether that passage supports \
that claim, using only the passage text.

Labels:
- SUPPORTED: the passage states the claim. Paraphrase is fine, but every number, name and \
condition must match.
- PARTIAL: the passage supports part of the claim, but a detail is missing or different.
- UNSUPPORTED: the passage does not state the claim, contradicts it, or is about something else.

Also give two overall ratings:
- completeness: does the answer address every part of the question? COMPLETE, PARTIAL or \
INCOMPLETE.
- answerable: do the passages themselves contain the information the question asks for? \
YES, PARTIAL or NO. Judge the passages, not the answer: a question about something the \
passages never mention is NO even if the answer sounds plausible.

Reply with only a JSON object in this shape, with one entry per check id:
{"checks": [{"id": 1, "label": "SUPPORTED", "reason": "<under 15 words>"}], \
"completeness": "COMPLETE", "answerable": "YES"}"""


@dataclass(frozen=True)
class JudgePair:
    id: int
    claim: str
    citation: int


def format_passage(number: int, chunk: Chunk) -> str:
    """``<passage id="n" source="...">``: small models cite tagged ids more reliably than ``[n]``
    headers, which they confuse with the ``[n]`` citations they are writing."""
    source = chunk.source + (f" p.{chunk.page}" if chunk.page else "")
    if chunk.section_heading:
        source += f" > {chunk.section_heading}"
    source = source.replace('"', "'")
    return f'<passage id="{number}" source="{source}">\n{chunk.text.strip()}\n</passage>'


def build_answer_messages(question: str, contexts: Sequence[ScoredChunk]) -> list[Message]:
    passages = "\n\n".join(format_passage(i, sc.chunk) for i, sc in enumerate(contexts, start=1))
    return [
        system(ANSWER_SYSTEM),
        user(f"{passages}\n\nQuestion: {question.strip()}"),
    ]


def build_judge_messages(
    question: str,
    answer: str,
    passages: dict[int, Chunk],
    pairs: Sequence[JudgePair],
) -> list[Message]:
    """One message holding every (claim, passage) pair, so verification is one model call."""
    passage_text = "\n\n".join(format_passage(n, passages[n]) for n in sorted(passages))
    # json.dumps quotes the claim, so a claim containing "|" or quotes cannot break the line.
    check_lines = "\n".join(
        f"{p.id}. Passage [{p.citation}] | Claim: {json.dumps(p.claim, ensure_ascii=False)}"
        for p in pairs
    )
    return [
        system(JUDGE_SYSTEM),
        user(
            f"Question: {question.strip()}\n\nAnswer:\n{answer.strip()}\n\n"
            f"Passages:\n\n{passage_text}\n\nChecks:\n{check_lines}"
        ),
    ]
