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

## Phase 2: Indexing and hybrid retrieval ✅

- [x] Local embeddings (`bge-small-en-v1.5` via fastembed) with a disk cache
- [x] ChromaDB vector store and BM25 index kept in sync by a single index manager
- [x] Near-duplicate filtering (cosine > 0.95) before insert
- [x] Dense and sparse retrieval, weighted Reciprocal Rank Fusion
- [x] Cross-encoder reranking (top 20 → top 5)

## Phase 3: Generation and citations ✅

- [x] LLM adapters: Ollama (default, local), optional OpenAI and Anthropic, a response cache and a deterministic stub for CI
- [x] Grounded prompt with numbered `[n]` citations
- [x] Citation verification with an LLM judge from a different model family, batched into one call
- [x] Composite confidence score (retrieval, citation coverage, completeness)
- [x] Structured "I don't know" responses when retrieval is weak

## Phase 4: Evaluation and API ✅

- [x] 55-question golden dataset: lookup, multi-hop, exact-term, unanswerable, ambiguous
- [x] Metrics: recall@k, MRR, correctness, faithfulness, citation accuracy, abstention precision
- [x] Comparison matrix: 3 chunking strategies × hybrid / dense / sparse, with and without reranking
- [x] Abstention thresholds tuned on the eval data, plus a judge answerability check
- [x] FastAPI service: `/v1/ask`, `/v1/search`, `/v1/documents`, `/v1/ingest`, `/v1/eval/summary`
- [x] Offline evaluation in CI on every pull request

## Phase 5: Dashboard and deployment ✅

- [x] React dashboard: answers with clickable citations, score breakdowns, hybrid vs dense comparison, eval charts, document upload
- [x] Docker Compose with the API (embedded Chroma, see ADR-0005), Ollama and the dashboard, plus a seed script and a GPU file
- [x] Case study with evaluation results, and a demo script

## Beyond 1.0

- More near-miss questions in the golden set, and human labels to calibrate the correctness judge
- A smaller judge, or room for both models on the GPU, to cut verification latency
- A Chroma server behind `VectorStore` if the index outgrows one machine
