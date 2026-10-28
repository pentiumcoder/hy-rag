# hy-rag

**Production-style Retrieval-Augmented Generation over internal documentation**: hybrid BM25 + dense retrieval, reciprocal rank fusion, cross-encoder reranking, grounded answers with *verified* citations, and an evaluation suite that measures all of it. Runs 100% locally, with no API keys required.

[![CI](https://github.com/pentiumcoder/hy-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/pentiumcoder/hy-rag/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Runs locally](https://img.shields.io/badge/runs-100%25%20locally-purple)

> **Status: Day 2 of 5.** Ingestion, indexing and hybrid retrieval are done. See the [roadmap](docs/ROADMAP.md) for what's next.

## Why this exists

Most RAG demos embed one PDF and call it done. Real internal docs are messier:

- **Mixed formats.** Markdown wikis, HTML exports, plain-text runbooks and PDF postmortems.
- **Exact identifiers.** Questions mention config keys (`ingest.max_batch_size`), error codes (`NW-4012`) and CLI flags that pure semantic search often misses.
- **Facts spread across documents.** "What error does an Enterprise customer see when they exceed their rate limit?" needs the pricing page *and* the error catalogue.
- **High cost of being wrong.** A confident, uncited answer is worse than "I don't know".

hy-rag treats those as first-class engineering problems and measures every design decision against a golden evaluation set.

## What works today (v0.2.0)

| Capability | Details |
|---|---|
| Multi-format loading | Markdown, plain text, HTML and PDF are normalized into sections with `source`, `section_heading`, `page` and a content hash |
| Raw + processed store | Originals are kept in `data/raw/` and parsed JSON in `data/processed/`. You can re-index or re-parse without re-uploading, and unchanged files are skipped. |
| 3 chunking strategies | **fixed** (size + overlap baseline), **recursive** (header-aware, packs whole sections), **semantic** (splits where embedding similarity between adjacent sentences drops) |
| Local embeddings | `BAAI/bge-small-en-v1.5` through fastembed (ONNX, CPU), with a SQLite disk cache so re-indexing an unchanged corpus embeds nothing |
| Dense + sparse indexes | ChromaDB (one collection per chunking strategy) and BM25 with an identifier-aware tokenizer, both written only by `IndexManager`, which detects drift and repairs BM25 from Chroma |
| Near-duplicate filter | Chunks with cosine > 0.95 to an indexed chunk are skipped and logged to `dedup_report.json` |
| Hybrid retrieval | Dense and BM25 top 10 each, fused with Reciprocal Rank Fusion, then a cross-encoder (`ms-marco-MiniLM-L-6-v2`) reranks the top 20 down to 5 |
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

## Quickstart

```bash
git clone https://github.com/pentiumcoder/hy-rag.git && cd hy-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

hyrag ingest --all                                  # parse, chunk, dedup, embed and index (all strategies)
hyrag search "What error do I get when a batch is too large?"
hyrag search "NW-4290" --mode sparse                # dense | sparse | hybrid
hyrag chunk --compare                               # chunking statistics per strategy
```

The first `ingest` downloads the two local models (about 150 MB) into `.cache/models/`. After that everything runs offline.

## Project layout

```
src/hyrag/
  ingestion/   loaders (md, txt, html, pdf), normalization, raw/processed store
  chunking/    fixed, recursive, semantic chunkers behind one Chunker protocol
  embeddings/  fastembed, OpenAI (optional) and fake embedders, plus the disk cache
  index/       Chroma vector store, BM25 store, tokenizer, dedup, IndexManager
  retrieval/   reciprocal rank fusion, cross-encoder reranker, HybridRetriever
  cli.py       `hyrag` command-line interface
data/corpus/   Northwind Cloud sample documentation
scripts/       sample-PDF generator and its plain-text sources
docs/          roadmap and architecture decision records
```

## Development

```bash
ruff check . && ruff format --check .
mypy
pytest --cov                                 # fast, offline: fake embedder, no reranker
HYRAG_TEST_MODELS=1 pytest -m models         # end-to-end with the real local models
```

CI runs the same checks on Python 3.11, 3.12 and 3.13. The default test run uses a deterministic fake embedder, so it needs no network, models or API keys.

## Design decisions

- [ADR-0001: Three switchable chunking strategies](docs/adr/0001-chunking-strategies.md)
- [ADR-0002: Hybrid retrieval with equal-weight RRF and a cross-encoder rerank](docs/adr/0002-hybrid-retrieval-and-rrf.md)

## License

[MIT](LICENSE)
