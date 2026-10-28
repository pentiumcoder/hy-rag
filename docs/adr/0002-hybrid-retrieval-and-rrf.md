# ADR-0002: Hybrid retrieval with equal-weight RRF and a cross-encoder rerank

- **Status:** Accepted
- **Date:** 2026-10-28

## Context

Questions about internal docs come in two shapes:

- **Paraphrased:** "how many vacation days roll over?" when the policy says "carry over up to 5 unused PTO days". Dense embeddings handle this; keyword search does not.
- **Exact identifiers:** `NW-4290`, `keystone.token_ttl_minutes`, `X-NW-Signature`. Embedding models split these into sub-word pieces with little meaning. On our corpus, `bge-small-en-v1.5` ranks the `NW-4290` catalogue entry **10th** for the query `NW-4290`, and its top 10 similarities all sit between 0.56 and 0.60, so the model is effectively guessing. BM25 ranks the entry **1st**.

Neither retriever is enough on its own, and the two produce scores on incompatible scales: cosine similarity in [-1, 1] and BM25 in [0, ∞).

## Decision

1. **Run both retrievers.** Dense search uses Chroma with cosine similarity on `bge-small-en-v1.5`. Sparse search uses BM25 with an identifier-aware tokenizer that keeps `ingest.max_batch_size` whole *and* emits `ingest`, `max`, `batch` and `size`, so both exact-key and plain-English queries match.
2. **Fuse with Reciprocal Rank Fusion:** `score(d) = Σ wᵢ / (k + rankᵢ(d))`, with `k = 60`. RRF uses ranks only, so it needs no score normalization. That kind of normalization is fragile because BM25 scores shift with query length and corpus statistics.
3. **Use equal weights (0.5 / 0.5) by default.** The weights stay configurable, and Phase 4's evaluation sweeps them.
4. **Rerank the fused top 20 with a cross-encoder** (`ms-marco-MiniLM-L-6-v2`) and return the top 5. A cross-encoder reads the query and passage together, which is far more precise than comparing two independent embeddings. It is too slow for the whole corpus but fine for 20 pairs.
5. **Give every mode an equally deep candidate pool.** Dense-only and sparse-only modes each fetch `fusion_top_n` (20) candidates, so comparing modes in the evaluation compares retrievers, not pool sizes.

### Why not 0.7 / 0.3?

The original plan weighted dense retrieval 0.7 and BM25 0.3. The arithmetic rules that out. With `k = 60`, rank 1 scores `1/61` and rank 10 scores `1/70`, only about 15% apart. Any weight ratio above about 1.15 therefore dominates the rank signal entirely:

| dense / sparse weights | a BM25-only #1 hit outranks dense-only hits from rank… |
|---|---|
| 0.7 / 0.3 | 83 |
| 0.6 / 0.4 | 32 |
| 0.5 / 0.5 | 2 |

At 0.7 / 0.3, "hybrid" would quietly become "dense, with BM25 results appended after it". That is exactly the failure that hybrid search exists to prevent. `test_lopsided_weights_silence_the_weaker_retriever` pins this behaviour down.

## Consequences

- Every result carries its dense rank and score, its BM25 rank and score, its RRF score and its rerank score (`ScoredChunk`), so the dashboard and evaluation can show *why* a chunk ranked where it did.
- The BM25 index uses Lucene's always-positive idf, `ln(1 + (N − n + 0.5)/(n + 0.5))`, instead of rank_bm25's classic Okapi idf. The Okapi version is zero for a term that appears in exactly half the chunks and negative above that, which silently erases useful terms on a small corpus.
- Loading the embedder and the cross-encoder adds a few seconds to every cold CLI call. The API (Phase 4) loads both once at startup.
- Near-duplicate filtering keeps the first copy it sees, in source-path order. On this corpus that keeps `hr/leave-faq.txt` and skips the matching chunk of `hr/pto-policy.md` (cosine 0.96). Preferring authoritative sources is a possible refinement once the evaluation shows whether it matters.

## Alternatives considered

- **Linear combination of normalized scores** (min-max or z-score per query). It needs per-query normalization and is sensitive to outliers. RRF is the more robust default and is also what most production hybrid search engines offer first.
- **Dense only, with a larger model.** This doesn't fix identifiers: a larger model still sees `NW-4290` as sub-word pieces.
- **An LLM as reranker.** It is more flexible but 10–100× slower per query on local hardware. It remains a possible option for the evaluation.
