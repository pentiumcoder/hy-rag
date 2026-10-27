# Changelog

All notable changes to this project are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

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

[Unreleased]: https://github.com/pentiumcoder/hy-rag/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/pentiumcoder/hy-rag/releases/tag/v0.1.0
