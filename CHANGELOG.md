# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/pentiumcoder/hy-rag/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/pentiumcoder/hy-rag/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/pentiumcoder/hy-rag/releases/tag/v0.1.0
