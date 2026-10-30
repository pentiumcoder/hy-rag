# ADR-0003: Batched cross-family citation checks, a composite confidence score, and abstaining before generation

- **Status:** Accepted; the abstention threshold is superseded by [ADR-0004](0004-tuned-abstention-policy.md)
- **Date:** 2026-10-29

## Context

A RAG answer with `[n]` markers looks trustworthy whether or not the passages say what the answer claims. On this corpus, with a local 7B generator, we saw two ways that goes wrong:

- **Wrong passage cited.** Asked how many PTO days carry over, `qwen2.5:7b` answered correctly ("Up to 5 unused PTO days ...") but cited `[2]`, a passage about taking PTO during an on-call week. The fact was in `[1]`. A reader who clicks `[2]` is misled, even though the answer is right.
- **Invented content.** With a worked example in the prompt, the model copied the example's fact ("Unused PTO expires on March 31") into an unrelated answer about certificate alerts, complete with a citation.

Questions the corpus cannot answer come in two kinds:

- **Off-topic:** "Who won the 2022 football world cup?" The cross-encoder's best logit is about −11.
- **Near-miss:** "How much does the Team plan cost?" The pricing page is retrieved (logit −2.7), but there is no Team plan.

The cross-encoder can't separate near-misses from real answers. "What is the per diem for travel meals?" scores −3.8, which is lower than the near-miss, yet the expense policy answers it.

The hardware constraint matters too. The generator (4.7 GB) and the judge (4.9 GB) don't both fit in 8 GB of VRAM, so Ollama swaps models between the generate and verify steps. Each judge call costs a model load plus generation time.

## Decision

1. **Verify every citation with a judge from a different model family.** `llama3.1:8b` judges `qwen2.5:7b`. A model grading its own output tends to accept its own phrasing and share its blind spots. Each (claim, cited passage) pair is labelled SUPPORTED, PARTIAL or UNSUPPORTED. A citation of a passage that doesn't exist is rejected in code, with no model call.
2. **Batch all pairs into one JSON call.** The judge sees the question, the answer and each cited passage once, plus a numbered list of checks. It returns `{"checks": [...], "completeness": ...}`. This costs one model swap and one prompt per answer instead of one per pair: a 5-bullet answer citing 2 passages each would otherwise take 10 calls. Malformed JSON gets one repair round. Missing verdicts become `UNVERIFIED`, and a judge failure becomes a warning, not an error.
3. **Score confidence as `0.4 × retrieval + 0.4 × citation coverage + 0.2 × completeness`.**
   - **Retrieval** is `sigmoid((best rerank logit + 4) / 2)`. It is centred on −4, where answerable and unanswerable questions overlap, and the scale of 2 spreads the −11 to +8 logit range over (0, 1).
   - **Citation coverage** is the mean verdict over the answer's factual sentences: supported counts 1, partial or unverified 0.5, unsupported or uncited 0.
   - **Completeness** is the judge's rating of whether the whole question was answered.
4. **Abstain before generating when retrieval confidence is below 0.2** (a logit of about −6.8). This only catches the clearly off-topic questions, which is all retrieval scores can do reliably. Near-misses go to the generator, which is told to reply `INSUFFICIENT_CONTEXT` with `Found:` and `Missing:` lines. If every sentence of an answer is a "the context does not cover" gap, the pipeline treats that as an abstention too. Every abstention returns what was found, what is missing and the closest documents, never a bare "I don't know".
5. **Make the prompt easy for a 7B model to follow.**
   - Passages sit in `<passage id="n">` tags. With `[n] Source:` headers, the model confused the headers with the citations it was writing.
   - The example format uses placeholders, never real-looking facts.
   - If a list ends with a single citation group and no other line in the block has one, that group is copied to the block's other sentences. Those inherited citations still go to the judge, so a wrong inheritance is flagged rather than hidden.
6. **Run the same pipeline offline.** `ExtractiveStubLLM` reads the real prompts. As a generator it copies the best-overlapping passage sentences; as a judge it labels each check by term overlap and requires every number in the claim to appear in the passage. Tests, CI and the Day 4 offline eval use it.

## Consequences

- On a first 12-question sample (8 answerable, 4 not):
  - all 8 answerable questions were answered, with confidence 0.81 to 1.00
  - the 2 off-topic questions abstained at the retrieval gate
  - the 2 near-misses abstained in the generator
  - stray citations were flagged in 2 answers

- Answers take about 15–30 s on an RTX 4070 laptop GPU, mostly from the model swap before verification. `--no-verify` and the `verify` override skip the judge when latency matters more than checking.
- A correct answer with a wrong citation now scores low: in the PTO example above, citation coverage was 0 and confidence 0.40. That is the intended behaviour. The UI shows the answer with its flagged source rather than hiding it.
- The judge is itself a 7B model and makes mistakes. It occasionally rates a partially relevant passage as SUPPORTED. Phase 4 measures citation accuracy against human labels and tunes `abstain_threshold`, `rerank_center` and `rerank_scale` from data instead of the hand-picked values above.
- The response cache makes repeated eval runs nearly free, but only for temperature-0 calls, so a sampled reply is never replayed as if it were fixed.

## Alternatives considered

- **Same-model self-check.** It's cheaper because there's no model swap, but self-verification is known to be lenient toward the model's own phrasing.
- **One judge call per citation.** Each call sees less context, so it is simpler, but on local hardware it multiplies latency by the number of pairs.
- **NLI cross-encoder for entailment.** It's fast and deterministic, but off-the-shelf NLI models struggle with long technical passages, config keys and numbers. It remains an option for a fast pre-filter.
- **Abstaining on a retrieval threshold alone.** No threshold separates "Team plan" (−2.7, unanswerable) from "per diem" (−3.8, answerable). The generator has to make that call.
- **Calibrating confidence with logistic regression.** This needs labelled outcomes, which Phase 4 produces. The fixed weights are the baseline that calibration will be compared against.
