from __future__ import annotations

import json

import pytest

from hyrag.generation.claims import (
    extract_claims,
    find_citations,
    is_insufficient,
    parse_insufficient,
    split_sentences,
    strip_citations,
)
from hyrag.generation.prompts import (
    ANSWER_SYSTEM,
    INSUFFICIENT,
    JUDGE_SYSTEM,
    JudgePair,
    build_answer_messages,
    build_judge_messages,
    format_passage,
)
from hyrag.models import ScoredChunk
from tests.conftest import make_chunk


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Five days [2].", [2]),
        ("Uses [1][3] and later [1].", [1, 3]),
        ("Grouped [2, 4].", [2, 4]),
        ("A range [1-3].", [1, 2, 3]),
        ("An en dash range [2–3].", [2, 3]),
        ("No citations here.", []),
        ("A config key ingest.max_batch_size [1].", [1]),
    ],
)
def test_find_citations(text: str, expected: list[int]) -> None:
    assert find_citations(text) == expected


def test_strip_citations_tidies_punctuation() -> None:
    assert strip_citations("Up to 5 days [2]. ") == "Up to 5 days."
    assert strip_citations("First [1], then second [2][3].") == "First, then second."


def test_split_sentences_keeps_citations_on_both_sides_of_the_stop() -> None:
    answer = (
        "PTO accrues at 2.08 days per month [1]. Unused days expire. [2] Ask HR [3]!\n"
        "\n"
        "## Steps\n"
        "- **Find** the certificate [2].\n"
        "1. Renew it with `kubectl cert-manager renew` [2]"
    )
    assert split_sentences(answer) == [
        "PTO accrues at 2.08 days per month [1].",
        "Unused days expire. [2]",
        "Ask HR [3]!",
        "Steps",
        "Find the certificate [2].",
        "Renew it with `kubectl cert-manager renew` [2]",
    ]


def test_citations_after_the_full_stop_stay_with_their_sentence() -> None:
    """Seen from qwen2.5:7b: the citations get a full stop of their own."""
    answer = "- The allowance is $75 per day. [1][2]. The context does not cover alcohol."
    claims = extract_claims(answer)
    assert [(c.text, c.citations, c.kind) for c in claims] == [
        ("The allowance is $75 per day.", [1, 2], "fact"),
        ("The context does not cover alcohol.", [], "gap"),
    ]
    assert find_citations("Employees get 10 paid sick days per year. [2][3].") == [2, 3]


def test_extract_claims_numbers_facts_and_marks_gaps() -> None:
    answer = (
        "Steps:\n"
        "Up to 5 PTO days carry over [1].\n"
        "They expire on March 31 [1][2].\n"
        "OK.\n"
        "The context does not cover what happens to extra days."
    )
    claims = extract_claims(answer)
    assert [(c.index, c.kind, c.citations) for c in claims] == [
        (1, "fact", [1]),
        (2, "fact", [1, 2]),
        (3, "gap", []),
    ]
    assert claims[0].text == "Up to 5 PTO days carry over."
    assert claims[0].label is None


@pytest.mark.parametrize(
    "sentence",
    [
        "The context does not cover the Team plan.",
        "The provided passages do not mention a student discount [1].",
        "The Team plan is not mentioned in the provided passages. [1][2]",
        "- The Team plan pricing is not explicitly mentioned in the provided passages. [1][2]",
        "There is no information about dental insurance.",
    ],
)
def test_gap_sentences_are_not_facts_even_with_citations(sentence: str) -> None:
    (claim,) = extract_claims(sentence)
    assert claim.kind == "gap"


def test_a_closing_citation_group_is_inherited_by_its_block() -> None:
    answer = (
        "When it fires:\n"
        "- Find the failing certificate.\n"
        "- Describe it and read the events.\n"
        "- Force a renewal. [1][3]\n"
        "\n"
        "Root rotation is planned [2]. Next one is in 2027.\n"
        "The context does not cover manual rotation."
    )
    claims = extract_claims(answer)
    assert [(c.citations, c.inherited) for c in claims] == [
        ([1, 3], True),
        ([1, 3], True),
        ([1, 3], False),
        ([2], False),  # the block's citation is not on its last fact: nothing is inherited
        ([], False),
        ([], False),
    ]


def test_insufficient_replies_are_parsed() -> None:
    reply = f"{INSUFFICIENT}\nFound: Pricing for the Starter and Growth plans [1].\nMissing: none"
    assert is_insufficient(reply)
    assert is_insufficient("insufficient context")
    assert not is_insufficient("Up to 5 days [1].")
    assert parse_insufficient(reply) == (["Pricing for the Starter and Growth plans."], [])
    found, missing = parse_insufficient(f"{INSUFFICIENT}\nFound: -\nMissing: The Team plan.")
    assert (found, missing) == ([], ["The Team plan."])


def _contexts() -> list[ScoredChunk]:
    chunks = [
        make_chunk(
            "Up to 5 unused PTO days carry over.", source="hr/pto.md", heading="PTO > Carry"
        ),
        make_chunk(
            'Webhooks carry the "X-NW-Signature" header.',
            source="api/hooks.pdf",
            heading=None,
            page=3,
        ),
    ]
    return [ScoredChunk(chunk=c, rank=i, score=1.0) for i, c in enumerate(chunks, start=1)]


def test_answer_prompt_numbers_passages_and_states_the_rules() -> None:
    messages = build_answer_messages("  How many days carry over? ", _contexts())
    assert messages[0] == {"role": "system", "content": ANSWER_SYSTEM}
    prompt = messages[1]["content"]
    assert '<passage id="1" source="hr/pto.md > PTO > Carry">' in prompt
    assert '<passage id="2" source="api/hooks.pdf p.3">' in prompt
    assert prompt.endswith("Question: How many days carry over?")
    for rule in ("[1][3]", INSUFFICIENT, "The context does not cover"):
        assert rule in ANSWER_SYSTEM


def test_format_passage_escapes_quotes_in_the_source_attribute() -> None:
    chunk = make_chunk("text", source='odd "name".md', heading=None)
    assert format_passage(4, chunk) == '<passage id="4" source="odd \'name\'.md">\ntext\n</passage>'


def test_judge_prompt_batches_every_pair_in_one_message() -> None:
    contexts = _contexts()
    pairs = [
        JudgePair(1, 'Up to 5 days carry over, "per year" | maybe', 1),
        JudgePair(2, "The header is X-NW-Signature.", 2),
    ]
    messages = build_judge_messages(
        "Q?", "Answer [1][2].", {1: contexts[0].chunk, 2: contexts[1].chunk}, pairs
    )
    assert messages[0]["content"] == JUDGE_SYSTEM and len(messages) == 2
    prompt = messages[1]["content"]
    assert prompt.startswith("Question: Q?\n\nAnswer:\nAnswer [1][2].\n\nPassages:")
    line = next(x for x in prompt.splitlines() if x.startswith("1. "))
    assert json.loads(line.split("Claim: ", 1)[1]) == pairs[0].claim
    assert "2. Passage [2] | Claim:" in prompt
    assert '"completeness"' in JUDGE_SYSTEM
