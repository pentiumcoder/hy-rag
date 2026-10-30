# ADR-0004: A layered abstention policy, tuned on the evaluation set

- **Status:** Accepted
- **Date:** 2026-10-30
- **Supersedes:** the hand-picked `abstain_threshold = 0.2` in ADR-0003

## Context

ADR-0003 left one problem open: retrieval scores alone can't decide when to refuse. The cross-encoder rates how *related* a passage is to the question, not whether it *answers* it. The pricing page is closely related to "How much does the Team plan cost?", but there is no Team plan.

The Phase 4 golden set (`eval/golden.jsonl`) turns this into measurements. It has 8 unanswerable questions:
- 6 near-misses that the corpus almost covers (Team plan, parental leave, a January 2026 outage, contractors' vacation days, ...)
- 2 off-topic ones

Across the 4 generation configurations that gives 32 unanswerable cases next to 188 answerable ones. Retrieval confidence, `sigmoid((best rerank logit + 4) / 2)`, scores them like this:

| questions | retrieval confidence |
|---|---|
| off-topic | 0.03 |
| near-misses | 0.22 to 0.91 |
| answerable | 0.62 to 1.00 |

Near-misses overlap the answerable range almost completely. A gate high enough to catch them would also refuse real questions, so no retrieval threshold separates them.

## Decision

Refuse at the first of four layers that fires. Tune their thresholds by replaying recorded eval runs.

1. **Retrieval gate (`abstain_threshold`).** Nothing is generated below it. It only has to catch off-topic questions, which it does cheaply.
2. **Generator self-check.** The answer prompt tells the model to reply `INSUFFICIENT_CONTEXT` when the passages don't answer the question. An answer made only of "the context does not cover" sentences counts as the same refusal.
3. **Judge answerability check (`abstain_on_unanswerable`).** This is new. The batched verification call now also asks the judge whether the *passages* contain what the question asks for, and shows it every retrieved passage, not only the cited ones. If the judge says no and no sentence of the answer is supported, the answer is withdrawn and kept as `draft_answer`. It costs no extra model call.
4. **Confidence floor (`answer_threshold`).** A verified answer scoring below it is withdrawn.

`generate()` and `verify()` are now separate steps, so the eval can run them in phases. It generates every answer while `qwen2.5:7b` is loaded, then verifies and grades them all while `llama3.1:8b` is loaded. It runs with every layer switched off and records every signal per question. `hyrag eval` then scores the policies on a grid (gate 0 to 0.9, floor 0 to 0.8, judge check on or off) without calling a model again.

**Utility** counts +1 per correct answer (+0.5 per partial one) and −2 per wrong answer. Any answer to an unanswerable question counts as wrong, because a wrong answer shown as right costs more than an honest refusal.

Ties go to the lower confidence floor and to the judge check being on. Among equally good gates, the tuner takes the middle of the range rather than its edge.

## Results

These are pooled over the 4 configurations and 55 questions in `eval/results/`:

| policy | abstention precision / recall | answered correctly | answered wrongly | utility |
|---|---|---|---|---|
| no abstention at all | n/a | 161.5 | 41 | +0.361 |
| Day 3: gate 0.2 + generator | 88.2% / 93.8% | 161.5 | 7 | +0.670 |
| + judge answerability check | 88.6% / 96.9% | 161.5 | 6 | +0.680 |
| **tuned: gate 0.3 + generator + judge, no floor** | 88.6% / 96.9% | 161.5 | 6 | +0.680 |

- **The generator does most of the work.** Under the Day 3 policy every near-miss case reached it, and it declined 22 of the 24. The off-topic questions never reached it. It also declined 4 answerable cases (two exact-term questions with semantic chunks, and the ambiguous A04 twice). That is why abstention precision is 88.6% rather than 100%.
- **The judge check caught one more.** It was a refusal split over two lines ("The context does not cover / - which database ..."), which the claim parser read as a cited fact. The check is cheap insurance against parser and format misses, not a near-miss detector. On its own its AUROC is 0.74.
- **The gate is flat.** Any value from 0.05 to 0.6 gives the same utility: it blocks the off-topic questions and nothing answerable. 0.3 is the middle of that range. One near-miss (retrieval 0.22) now stops at the gate instead of reaching the generator, which saves a generation call.
- **No confidence floor helps.** Floors of 0.5 to 0.8 withdraw correct answers as fast as wrong ones. For example, a floor of 0.6 removes 1 wrong answer and 2 correct ones. `answer_threshold` stays 0.
- **What still gets through:**
  - With semantic chunking, the generator wrote "The Team plan, which is equivalent to the Growth plan, costs $499 per month". The judge accepted the Growth price as supported. It is the only unanswerable case answered under the tuned policy.
  - The other 5 "wrong" answers are answerable questions graded INCORRECT. Checked by hand:
    - 2 are real errors: M09 called a 30%-of-customers outage SEV2, and E04 gave NW-4012 as HTTP 400.
    - 3 are judge grading errors: both M03 answers are correct, and A01 deserves PARTIAL under the rubric.
- **One answerable question now abstains.** "What is the limit for Enterprise?" is deliberately ambiguous. With recursive chunks the generator replied only with "not explicitly stated" sentences, a reasonable refusal for an ambiguous question.

## Consequences

- The defaults are now `abstain_threshold = 0.3`, `abstain_on_unanswerable = true` and `answer_threshold = 0`. `hyrag eval` reprints the tuned values, so they can be re-derived whenever the corpus, models or prompts change. Because LLM replies are cached, a re-tune after changing only scoring code takes minutes.
- The tuning set is small: 32 unanswerable cases, of which 6 distinct near-misses. The gate's flat range is wide, so its exact value matters little. The judge check's value rests on a single case. More near-miss questions are the most useful addition to the golden set.
- Correctness is graded by a 7B judge, which got 3 of the 5 INCORRECT verdicts it gave wrong when checked by hand. Its correctness numbers are therefore a lower bound with noise of a few points. Comparisons between configurations are more reliable than the absolute values.

## Alternatives considered

- **A higher retrieval gate.** It cannot separate near-misses (0.22 to 0.91) from answerable questions (from 0.62) without refusing real questions.
- **A separate answerability classifier call before generation.** That is one more model call, and a model swap on 8 GB of VRAM, for every question. The generator already reads the passages and declines well.
- **Weighting answerability into the confidence score.** It's tempting, but its AUROC (0.74) is the weakest of the four signals. As a hard rule paired with "nothing is supported", it only fires when the answer has no support anyway.
