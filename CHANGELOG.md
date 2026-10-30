# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

## [0.4.0] - 2026-10-30

### Added

- Golden evaluation set `eval/golden.jsonl` with 55 questions: 20 lookup, 12 multi-hop, 10 exact-term, 8 unanswerable, 5 ambiguous. Each has a reference answer, required key facts and verbatim evidence quotes, and the tests check that every quote exists and survives chunking under all three strategies.
- `hyrag eval`:
  - recall@1, recall@5 and MRR on all 18 retrieval configurations (3 chunking strategies x dense/sparse/hybrid x with and without reranking)
  - answer metrics on 4 generation configurations: LLM-graded correctness, faithfulness, citation accuracy, key-fact recall, and abstention precision and recall
  - output: `summary.json`, `report.md`, PNG charts and per-question runs
- Abstention policy replay: generation runs once with every abstention rule off, then the retrieval gate, judge check and confidence floor are tuned offline by utility (+1 correct, -2 wrong).
- Judge answerability check. The batched verification call also rates whether the retrieved passages answer the question, and an answer the judge calls unanswerable with no supported sentence is withdrawn and kept as `draft_answer`.
- `AnswerPipeline.generate()` and `AnswerPipeline.verify()`, so batch jobs can run all generations, then all verifications, without swapping models on the GPU.
- FastAPI service (`hyrag serve`):
  - `POST /v1/ask`, with overrides
  - `POST /v1/search`
  - `GET /v1/documents`
  - `POST /v1/ingest`, for file uploads
  - `GET /v1/eval/summary`
  - `GET /health`
  - OpenAPI docs and CORS for the dashboard dev server
- CI job running the offline evaluation on every pull request, failing if the best recall@5 drops below 85%.
- Committed results in `eval/results/`, and ADR-0004 on the tuned abstention policy.

### Changed

- The default `abstain_threshold` is 0.3 (was 0.2), the middle of the range that is equally good on the eval set. `answer_threshold` stays 0, because no confidence floor improved utility.
- Sentences such as "X is not explicitly mentioned in the passages" are now recognized as gaps.

### Fixed

- Dense queries could fail with "Nothing found on disk" when Chroma 1.x lost a freshly built collection's vector segment, about once in 50 index builds in one process. Queries now fall back to exact search over the stored texts, re-embedded through the embedding cache. Every VectorStore on a directory now also shares one Chroma client.

## [0.3.0] - 2026-10-29

### Added

- `LLMClient` protocol with an Ollama adapter (the default: `qwen2.5:7b` generates, `llama3.1:8b` judges), optional OpenAI and Anthropic adapters, retries with backoff, tolerant JSON parsing with one repair round, and a SQLite response cache for deterministic calls.
- Grounded answer prompt: passages in `<passage id="n">` tags, a `[n]` citation after every sentence and bullet, exact values copied verbatim, a "The context does not cover" sentence for partial answers, and an `INSUFFICIENT_CONTEXT` reply with `Found:` and `Missing:` lines.
- Claim extraction that splits answers into sentences and bullets, expands `[1-3]` and `[2, 4]`, recognizes "not covered" sentences as gaps rather than facts, and lets a block's closing citation group cover the uncited lines before it.
- `CitationVerifier`: every (claim, passage) pair is labelled SUPPORTED, PARTIAL or UNSUPPORTED by the judge in one batched JSON call, along with a completeness rating. Citations of passages that do not exist are rejected without a model call.
- Confidence score `0.4 x retrieval + 0.4 x citation coverage + 0.2 x completeness`, where retrieval is a shifted sigmoid of the best cross-encoder logit.
- Structured abstention (`found`, `missing`, `suggested_documents`) when retrieval confidence is below `HYRAG_ABSTAIN_THRESHOLD` (nothing is generated), when the generator reports insufficient context, or when every sentence of the answer is a gap.
- `AnswerPipeline.ask(query, overrides) -> AnswerResponse`, with per-request overrides for mode, strategy, `top_k`, rerank, verification and the abstention threshold.
- `hyrag ask` with a claims table, flagged sources, the confidence breakdown, timings and `--json` output.
- `ExtractiveStubLLM`, a deterministic lexical generator and judge, so the whole pipeline runs offline in tests and CI.
- Opt-in end-to-end tests against Ollama (`HYRAG_TEST_LLM=1 pytest -m llm`), including a planted hallucination the judge must flag.
- ADR-0003 on batched, cross-family citation verification, confidence and abstention.

## [0.2.0] - 2026-10-28

### Added

- Local embeddings with `BAAI/bge-small-en-v1.5` through fastembed (the new default provider), an optional OpenAI embedder, and a SQLite embedding cache keyed by model and text.
- ChromaDB vector store (one cosine collection per chunking strategy) and a persistent BM25 index with an identifier-aware tokenizer that keeps `dotted.keys`, `snake_case` and `NW-4012`-style codes whole while also indexing their parts.
- `IndexManager`, the single writer for both indexes. It keeps a chunk manifest, applies incremental adds and removals, rebuilds when the embedder changes, and repairs BM25 and the manifest from Chroma if they drift.
- Near-duplicate filtering before insert (cosine > 0.95 against the index and within the batch), with skipped chunks recorded in `dedup_report.json` and re-evaluated if their original disappears.
- `HybridRetriever.retrieve(query, mode, k)` with dense, sparse and hybrid modes, weighted Reciprocal Rank Fusion and a cross-encoder reranker (`Xenova/ms-marco-MiniLM-L-6-v2`). Every result reports its dense, BM25, RRF and rerank scores.
- `hyrag ingest` and `hyrag search` CLI commands.
- Opt-in end-to-end tests with the real local models (`HYRAG_TEST_MODELS=1 pytest -m models`).
- ADR-0002 on hybrid retrieval and fusion weights.

### Changed

- The default embedding provider is now `fastembed`. Tests and CI pin the deterministic fake embedder.

## [0.1.0] - 2026-10-27

### Added

- Multi-format loaders for Markdown, plain text, HTML and PDF. They normalize text and record `source`, `section_heading` (full heading path), `page` and a content hash.
- `DocumentStore`, which keeps originals in `data/raw/` and parsed documents in `data/processed/`. It skips unchanged files and can re-parse from raw without re-uploading.
- Three switchable chunking strategies behind one `Chunker` interface:
  - `fixed`: size and overlap windows, the baseline
  - `recursive`: header-aware section packing with heading-path prefixes
  - `semantic`: topic-boundary detection using embedding similarity between windows of sentences
- `Embedder` protocol and a deterministic, dependency-free `FakeEmbedder` for tests and CI.
- `hyrag` CLI with `load` and `chunk` (including `--compare` and `--show`) commands.
- Northwind Cloud sample corpus: 26 internal documents across all four formats, plus the generator for the PDFs.
- CI on Python 3.11 to 3.13 (ruff, mypy strict, pytest with coverage), pre-commit hooks, Dependabot and community health files.
- ADR-0001 on the chunking strategies, and the project roadmap.

[Unreleased]: https://github.com/pentiumcoder/hy-rag/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/pentiumcoder/hy-rag/compare/v0.3.0...v0.4.0
[0.3.0]: https://github.com/pentiumcoder/hy-rag/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/pentiumcoder/hy-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pentiumcoder/hy-rag/releases/tag/v0.1.0
