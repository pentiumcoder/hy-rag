# Roadmap

hy-rag is built in five phases. Each phase ends with a stable interface that the next phase builds on, and ships as a tagged release.

| Phase | Release | Focus | Handoff interface |
|---|---|---|---|
| 1 | v0.1.0 | Multi-format ingestion and chunking | `load_corpus()`, `get_chunker().chunk()` |
| 2 | v0.2.0 | Indexing, dedup and hybrid retrieval | `HybridRetriever.retrieve()` |
| 3 | v0.3.0 | Grounded generation, citation verification, confidence | `AnswerPipeline.ask()` |
| 4 | v0.4.0 | Evaluation framework and FastAPI service | `eval/results/summary.json`, `/v1/*` |
| 5 | v1.0.0 | React dashboard, Docker, case study | `docker compose up` |

## Phase 1: Ingestion and chunking ✅

- [x] Loaders for Markdown, text, HTML and PDF, with normalized metadata
- [x] Raw and processed document store; re-index without re-uploading
- [x] Fixed, recursive (header-aware) and semantic chunking strategies
- [x] Northwind Cloud sample corpus (26 docs, 4 formats)
- [x] CI, tests and community files

## Phase 2: Indexing and hybrid retrieval

- [ ] Local embeddings (`bge-small-en-v1.5` via fastembed) with a disk cache
- [ ] ChromaDB vector store and BM25 index kept in sync by a single index manager
- [ ] Near-duplicate filtering (cosine > 0.95) before insert
- [ ] Dense and sparse retrieval, weighted Reciprocal Rank Fusion
- [ ] Cross-encoder reranking (top 20 → top 5)

## Phase 3: Generation and citations

- [ ] Grounded prompt with numbered `[n]` citations
- [ ] Citation verification with an LLM judge from a different model family
- [ ] Composite confidence score (retrieval, citation coverage, completeness)
- [ ] Structured "I don't know" responses when retrieval is weak

## Phase 4: Evaluation and API

- [ ] 55-question golden dataset: lookup, multi-hop, exact-term, unanswerable, ambiguous
- [ ] Metrics: recall@k, MRR, correctness, faithfulness, citation accuracy, abstention precision
- [ ] Comparison matrix: 3 chunking strategies × hybrid / dense / sparse
- [ ] FastAPI service: `/v1/ask`, `/v1/documents`, `/v1/ingest`

## Phase 5: Dashboard and deployment

- [ ] React dashboard: answers with clickable citations, score breakdowns, hybrid vs dense comparison
- [ ] Docker Compose with API, ChromaDB, Ollama and frontend, plus a seed script
- [ ] Case study with evaluation results
