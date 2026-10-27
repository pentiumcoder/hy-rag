# hy-rag

**Production-style Retrieval-Augmented Generation over internal documentation**: hybrid BM25 + dense retrieval, reciprocal rank fusion, cross-encoder reranking, grounded answers with *verified* citations, and an evaluation suite that measures all of it. Runs 100% locally, with no API keys required.

[![CI](https://github.com/pentiumcoder/hy-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/pentiumcoder/hy-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-purple)

> **Status: Day 1 of 5.** Ingestion and chunking are done. See the [roadmap](docs/ROADMAP.md) for what's next.

## Why this exists

Most RAG demos embed one PDF and call it done. Real internal docs are messier:

- **Mixed formats.** Markdown wikis, HTML exports, plain-text runbooks and PDF postmortems.
- **Exact identifiers.** Questions mention config keys (`ingest.max_batch_size`), error codes (`NW-4012`) and CLI flags that pure semantic search often misses.
- **Facts spread across documents.** "What error does an Enterprise customer see when they exceed their rate limit?" needs the pricing page *and* the error catalogue.
- **High cost of being wrong.** A confident, uncited answer is worse than "I don't know".

hy-rag treats those as first-class engineering problems and measures every design decision against a golden evaluation set.

## What works today (v0.1.0)

| Capability | Details |
|---|---|
| Multi-format loading | Markdown, plain text, HTML and PDF are normalized into sections with `source`, `section_heading`, `page` and a content hash |
| Raw + processed store | Originals are kept in `data/raw/` and parsed JSON in `data/processed/`. You can re-index or re-parse without re-uploading, and unchanged files are skipped. |
| 3 chunking strategies | **fixed** (size + overlap baseline), **recursive** (header-aware, packs whole sections), **semantic** (splits where embedding similarity between adjacent sentences drops) |
| Sample corpus | 26 internal docs for the fictional company *Northwind Cloud*, covering runbooks, config and API references, policies and postmortems, across all 4 formats |

## Quickstart

```bash
git clone https://github.com/pentiumcoder/hy-rag.git && cd hy-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

hyrag load                                          # parse data/corpus into data/processed
hyrag chunk --strategy recursive --show 2           # chunk stats + sample chunks
hyrag chunk --compare                               # all three strategies side by side
```

## Project layout

```
src/hyrag/
  ingestion/   loaders (md, txt, html, pdf), normalization, raw/processed store
  chunking/    fixed, recursive, semantic chunkers behind one Chunker protocol
  embeddings/  embedder protocol + deterministic fake embedder for tests
  cli.py       `hyrag` command-line interface
data/corpus/   Northwind Cloud sample documentation
scripts/       sample-PDF generator and its plain-text sources
docs/          roadmap and architecture decision records
```

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest --cov
```

CI runs the same checks on Python 3.11, 3.12 and 3.13. Tests use a deterministic fake embedder, so they need no network, models or API keys.

## Design decisions

- [ADR-0001: Three switchable chunking strategies](docs/adr/0001-chunking-strategies.md)

## License

[MIT](LICENSE)
