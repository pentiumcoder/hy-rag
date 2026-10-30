# hy-rag

**Production-style Retrieval-Augmented Generation over internal documentation**: hybrid BM25 + dense retrieval, reciprocal rank fusion, cross-encoder reranking, grounded answers with *verified* citations, and an evaluation suite that measures all of it. Runs 100% locally, with no API keys required.

[![CI](https://github.com/pentiumcoder/hy-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/pentiumcoder/hy-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-purple)

> **Status: Day 4 of 5.** Retrieval, verified answers, a 55-question evaluation and the HTTP API are done. See the [roadmap](docs/ROADMAP.md) for what's next.

## Why this exists

Most RAG demos embed one PDF and call it done. Real internal docs are messier:

- **Mixed formats.** Markdown wikis, HTML exports, plain-text runbooks and PDF postmortems.
- **Exact identifiers.** Questions mention config keys (`ingest.max_batch_size`), error codes (`NW-4012`) and CLI flags that pure semantic search often misses.
- **Facts spread across documents.** "What error does an Enterprise customer see when they exceed their rate limit?" needs the pricing page *and* the error catalogue.
- **High cost of being wrong.** A confident, uncited answer is worse than "I don't know".

hy-rag treats those as first-class engineering problems and measures every design decision against a golden evaluation set.

## What works today (v0.4.0)

| Capability | Details |
|---|---|
| Multi-format loading | Markdown, plain text, HTML and PDF are normalized into sections with `source`, `section_heading`, `page` and a content hash |
| Raw + processed store | Originals are kept in `data/raw/` and parsed JSON in `data/processed/`. You can re-index or re-parse without re-uploading, and unchanged files are skipped. |
| 3 chunking strategies | **fixed** (size + overlap baseline), **recursive** (header-aware, packs whole sections), **semantic** (splits where embedding similarity between adjacent sentences drops) |
| Local embeddings | `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU), with a SQLite disk cache so re-indexing an unchanged corpus embeds nothing |
| Dense + sparse indexes | ChromaDB (one collection per chunking strategy) and BM25 with an identifier-aware tokenizer, both written only by `IndexManager`, which detects drift and repairs BM25 from Chroma |
| Near-duplicate filter | Chunks with cosine > 0.95 to an indexed chunk are skipped and logged to `dedup_report.json` |
| Hybrid retrieval | Dense and BM25 top 10 each, fused with Reciprocal Rank Fusion, then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks the top 20 down to 5 |
| Grounded answers | `qwen2.5:7b` on Ollama answers only from the retrieved passages, with a `[n]` citation after every sentence and bullet, or says the context is insufficient |
| Verified citations | A judge from a different model family (`llama3.1:8b`) labels every claim–citation pair SUPPORTED, PARTIAL or UNSUPPORTED in one batched call. Citations of passages that don't exist are rejected without asking it. |
| Confidence score | `0.4 × retrieval + 0.4 × citation coverage + 0.2 × completeness`, always shown with its breakdown |
| Layered abstention | Off-topic questions are refused before generation, the generator declines near-misses, and the judge withdraws answers the passages do not support. Every refusal lists what was found, what is missing and the closest documents. Thresholds are tuned on the eval set. |
| Evaluation suite | 55 golden questions (lookup, multi-hop, exact-term, unanswerable, ambiguous), 18 retrieval configs and 4 answer configs, with recall@k, MRR, LLM-graded correctness, faithfulness, citation accuracy and abstention precision/recall |
| HTTP API | FastAPI with `/v1/ask` (per-request overrides, such as hybrid vs dense), `/v1/search`, `/v1/documents`, `/v1/ingest` (upload) and `/v1/eval/summary`, plus OpenAPI docs at `/docs` |
| Sample corpus | 26 internal docs for the fictional company *Northwind Cloud*, covering runbooks, config and API references, policies and postmortems, across all 4 formats |

### Why hybrid: a real example from this corpus

Searching for the error code `NW-4290`:

| | where the `NW-4290` catalogue entry ranks |
|---|---|
| dense only (`bge-small`) | **#10**; the top 10 similarities all sit between 0.56 and 0.60, so the model is guessing |
| BM25 only | **#1** |
| hybrid (RRF) | #2 |
| hybrid + cross-encoder | **#1**, followed by the SDK's retry guidance for that error, which only BM25 found |

`hyrag search` prints every stage's score, so you can see this for any query:

```text
$ hyrag search "NW-4290" -k 3
| # | source                          | section                |     dense |    bm25 |    rrf | rerank |
|---+---------------------------------+------------------------+-----------+---------+--------+--------|
| 1 | engineering/error-codes.md      | NW-4290 RATE_LIMITED   | 0.559 #10 | 7.60 #1 | 0.0153 |   6.34 |
| 2 | api/sdk-python.md               | Retries                |         - | 7.14 #2 | 0.0081 |   5.87 |
| 3 | engineering/config-reference.md | gateway.rate_limit.rps |  0.563 #8 | 6.93 #3 | 0.0153 |   5.32 |
```

[ADR-0002](docs/adr/0002-hybrid-retrieval-and-rrf.md) explains the fusion design, including why the default weights are equal rather than 0.7 / 0.3.

## Evaluation results

All numbers come from `hyrag eval` on the 55-question [golden set](eval/golden.jsonl), with the local models. Full report: [eval/results/report.md](eval/results/report.md).

**Retrieval** (recall@5 = share of a question's gold evidence in the top 5 chunks, over the 47 answerable questions):

| | dense | sparse (BM25) | hybrid (RRF) |
|---|---|---|---|
| recursive chunks | 84.0% | 92.5% | 92.5% |
| recursive + cross-encoder rerank | 96.8% | 98.9% | **98.9%** |
| fixed + rerank | 93.6% | 97.9% | 97.9% |
| semantic + rerank | 92.5% | 94.7% | 93.6% |

- The reranker adds 3 to 13 points to every configuration.
- Dense search alone misses the most, especially on multi-hop questions: 67 to 71% before reranking.
- Header-aware recursive chunks beat embedding-based semantic chunks on this corpus.

**Answers** (`qwen2.5:7b` answering, `llama3.1:8b` checking citations and grading against the reference answers):

| config | correct (of answered) | faithfulness | citation accuracy | answered | abstention precision / recall |
|---|---|---|---|---|---|
| **recursive / hybrid** | **91.3%** | 97.5% | 83.9% | 97.9% | 88.9% / 100% |
| recursive / dense | 89.1% | 96.0% | 83.9% | 97.9% | 88.9% / 100% |
| fixed / hybrid | 87.2% | 96.3% | 86.8% | 100% | 100% / 100% |
| semantic / hybrid | 83.3% | 94.9% | 81.8% | 95.7% | 77.8% / 87.5% |

**Abstention:** 31 of the 32 unanswerable cases were refused, and 1 slipped through. Off-topic questions are refused before any generation. The Team-plan kind of near-miss is caught by the generator and the judge, because no retrieval score can separate it from a real answer. Every layer was tuned by replaying the recorded runs; see [ADR-0004](docs/adr/0004-tuned-abstention-policy.md).

| abstention policy | answered wrongly (of 220) | utility |
|---|---|---|
| none | 41 | +0.36 |
| Day 3: retrieval gate + generator | 7 | +0.67 |
| tuned: + judge answerability check, gate at the middle of its flat range | **6** | **+0.68** |

These are small-sample numbers: 55 questions, graded by a 7B judge that got 3 of its 5 INCORRECT verdicts wrong when checked by hand. Read them as comparisons between configurations rather than as absolute scores.

### Answers that show their work

When the model cites the wrong passage, the judge says so:

```text
$ hyrag ask "What should I do when CertExpiringSoon fires?"
 - Find the failing certificate using kubectl get certificates -A | grep -v True.
 - Describe the certificate and read the latest events with kubectl describe ...
 - Common causes include Let's Encrypt rate limits, DNS challenges, or an unhealthy internal CA issuer.
 - Once the cause is fixed, force a renewal with kubectl cert-manager renew <name> -n <namespace>. [1][3][4]

[1] runbooks/certificate-rotation.md > When CertExpiringSoon fires
[3] runbooks/database-failover.md > Automatic failover  flagged
[4] runbooks/ingest-backpressure.md > Runbook: Ingest Backpressure  flagged

confidence 0.85 = 0.4 x retrieval 0.99 (rerank) + 0.4 x citations 0.88 + 0.2 x completeness 0.50
! Passage [3] does not support sentence(s) 1, 2, 3, 4: Passage [3] is about a different alert ...
```

[ADR-0003](docs/adr/0003-verified-citations-confidence-abstention.md) covers the design: why the judge is a different model family, why all checks go in one call, and why retrieval scores alone can't decide when to abstain.

## Quickstart

```bash
git clone https://github.com/pentiumcoder/hy-rag.git && cd hy-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

hyrag ingest --all                                  # parse, chunk, dedup, embed and index (all strategies)
hyrag search "What error do I get when a batch is too large?"
hyrag search "NW-4290" --mode sparse                # dense | sparse | hybrid
hyrag chunk --compare                               # chunking statistics per strategy

ollama pull qwen2.5:7b && ollama pull llama3.1:8b   # generator and judge, about 9.6 GB
hyrag ask "How many PTO days carry over to next year?"
hyrag ask "What does a SEV1 incident require?" --json --no-verify

hyrag eval --offline                                # the eval with fake models, ~20 s
hyrag eval                                          # the full eval with the local models
hyrag serve                                         # API on http://127.0.0.1:8000 (docs at /docs)
```

```bash
curl -s localhost:8000/v1/ask -H 'Content-Type: application/json'   -d '{"query": "NW-4290", "overrides": {"mode": "dense"}}'
curl -s localhost:8000/v1/ingest -F files=@my-runbook.md
```

The first `ingest` downloads the two local retrieval models (about 150 MB) into `.cache/models/`. `hyrag ask` needs [Ollama](https://ollama.com) running; with 8 GB of VRAM the two LLMs take turns on the GPU, so an answer takes 15–30 s, and repeated questions come from the response cache in `data/cache/`. To use a hosted model instead, set `HYRAG_LLM_PROVIDER=openai` or `anthropic` and install the matching extra.

## Project layout

```
src/hyrag/
  ingestion/   loaders (md, txt, html, pdf), normalization, raw/processed store
  chunking/    fixed, recursive, semantic chunkers behind one Chunker protocol
  embeddings/  fastembed, OpenAI (optional) and fake embedders, plus the disk cache
  index/       Chroma vector store, BM25 store, tokenizer, dedup, IndexManager
  retrieval/   reciprocal rank fusion, cross-encoder reranker, HybridRetriever
  llm/         Ollama, OpenAI and Anthropic chat clients, response cache, test stub
  generation/  prompts, claim parsing, citation verifier, confidence, abstention, AnswerPipeline
  evaluation/  golden set, metrics, correctness judge, abstention policy tuning, reports
  api/         FastAPI service
  cli.py       `hyrag` command-line interface
data/corpus/   Northwind Cloud sample documentation
eval/          golden questions and committed results (report, charts, per-question runs)
scripts/       sample-PDF generator and its plain-text sources
docs/          roadmap and architecture decision records
```

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest --cov                                 # fast, offline: fake embedder, stub LLM, no reranker
HYRAG_TEST_MODELS=1 pytest -m models         # retrieval with the real local models
HYRAG_TEST_LLM=1 pytest -m llm               # answering end to end, needs Ollama running
hyrag eval --offline --min-recall 0.85       # what CI runs on every pull request
```

CI runs the same checks on Python 3.11, 3.12 and 3.13. The default test run uses a deterministic fake embedder and an extractive stub model that plays both generator and judge, so the whole pipeline runs with no network, models or API keys. A separate CI job runs the offline evaluation and fails if retrieval recall drops.

## Design decisions

- [ADR-0001: Three switchable chunking strategies](docs/adr/0001-chunking-strategies.md)
- [ADR-0002: Hybrid retrieval with equal-weight RRF and a cross-encoder rerank](docs/adr/0002-hybrid-retrieval-and-rrf.md)
- [ADR-0003: Batched cross-family citation checks, confidence and abstention](docs/adr/0003-verified-citations-confidence-abstention.md)
- [ADR-0004: A layered abstention policy, tuned on the evaluation set](docs/adr/0004-tuned-abstention-policy.md)

## License

[MIT](LICENSE)
