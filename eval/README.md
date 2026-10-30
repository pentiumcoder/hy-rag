# Evaluation

`golden.jsonl` holds 55 questions about the Northwind Cloud corpus. `hyrag eval` scores retrieval and answers against them and writes the results to `results/`.

| type | count | what it tests |
|---|---|---|
| lookup | 20 | one fact from one document ("How much is the on-call stipend?") |
| multi-hop | 12 | facts from two documents ("What error does a Starter customer get at their rate limit, and what is that limit?") |
| exact-term | 10 | config keys, error codes and CLI commands that dense search alone tends to miss |
| unanswerable | 8 | near-misses the corpus almost covers ("How much does the Team plan cost?") and off-topic questions |
| ambiguous | 5 | questions with several valid readings ("What is the timeout?") |

## Question format

```json
{"id": "M01", "type": "multi_hop", "question": "...", "answerable": true,
 "reference": "The answer the documentation owners would give.",
 "must_include": ["NW-4290", "50"],
 "evidence": [{"sources": ["product/pricing-and-plans.md"], "quote": "50 requests/second"}]}
```

- **evidence** defines relevance. A retrieved chunk is relevant to an evidence item when it comes from one of the listed `sources` and contains the `quote` (case- and whitespace-insensitive). Quotes are short and verbatim, so one golden set works for every chunking strategy. The tests check that every quote exists in its source and survives chunking intact under all three strategies.
- **reference** is what the LLM judge compares answers with to grade correctness.
- **must_include** lists strings a correct answer contains. `a|b` accepts either one. They give a deterministic correctness check for offline runs and for when the judge fails.

## Metrics

| metric | computed from |
|---|---|
| recall@1, recall@5 | share of a question's evidence items found in the top 1 / 5 chunks |
| MRR | reciprocal rank of the first relevant chunk |
| correctness | `llama3.1:8b` grades each shown answer against the reference: correct 1, partial 0.5, incorrect 0 |
| faithfulness | mean citation verdict over an answer's sentences (Phase 3 verifier) |
| citation accuracy | share of (sentence, cited passage) pairs the judge labels supported (partial counts half) |
| abstention precision / recall | abstaining is the positive class; a question should be abstained on if it is unanswerable |
| utility | +1 per correct answer and −2 per wrong one (including any answer to an unanswerable question), per question |

## Running it

```bash
hyrag eval --offline                 # fake embedder + stub LLM in a scratch dir, ~20 s, no models
hyrag eval --retrieval-only          # all 18 retrieval configs with the real models, ~2 min
hyrag eval                           # plus answers on 4 configs with Ollama, ~1 h on an 8 GB GPU
```

The full run generates and verifies every answer with all abstention rules switched off and records every signal. It then replays abstention policies over the recorded runs to tune `abstain_threshold`, `answer_threshold` and the judge check without calling a model again (see [ADR-0004](../docs/adr/0004-tuned-abstention-policy.md)). LLM replies are cached, so a rerun after changing only scoring or policy code takes minutes.

CI runs the offline evaluation on every pull request and fails if the best recall@5 drops below 85%.
