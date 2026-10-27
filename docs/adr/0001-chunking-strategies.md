# ADR-0001: Three switchable chunking strategies

- **Status:** Accepted
- **Date:** 2026-10-27

## Context

Chunking decides what the retriever can find. If chunks are too large, they dilute similarity scores and waste context-window space. If they are too small, they lose the context that makes them answerable; a line like "Default: 500" is useless unless it still says which setting it belongs to. Internal documentation makes this harder, because it mixes very different shapes:

- reference pages with many tiny sections (one config key or error code each);
- long narrative documents (postmortems, policies);
- tables, CLI listings and code blocks;
- PDFs, where structure has to be inferred from layout.

No single strategy is obviously best for all of these, and choosing by intuition is exactly what an evaluation suite exists to avoid.

## Decision

Implement three strategies behind one `Chunker` interface, record the strategy on every chunk, and choose the default from evaluation results (Phase 4) rather than up front.

1. **`fixed`: the baseline.** Character windows (default 800 characters, 120 overlap) over the flattened document, with cuts moved to the nearest word boundary. It ignores structure entirely. It exists so the other strategies have something to beat.
2. **`recursive`: structure-aware.** Each section produced by the loaders is the unit. Consecutive small sections are packed together up to the size limit. Oversized sections are split recursively: paragraphs, then lines, then sentences, then words, using LangChain's `RecursiveCharacterTextSplitter`. Every chunk is prefixed with its full heading path, for example `Configuration Reference > Ingestion > ingest.max_batch_size`, so tiny reference sections stay self-describing.
3. **`semantic`: topic-aware.** Sentences are embedded. Each gap between two sentences is scored by the cosine similarity between the mean embedding of the sentences before it and the sentences after it (a window of 2). Gaps below the document's 20th-percentile similarity that are also the local minimum in their neighbourhood become boundaries. Minimum and maximum sizes keep chunks usable, and a tiny trailing piece is folded into the previous chunk.

All chunk IDs are a hash of `(source, strategy, index, text)`, so re-chunking unchanged input yields identical IDs. Phase 2 relies on this to keep the dense and sparse indexes in sync.

## Consequences

- Each strategy gets its own vector collection in Phase 2, so the eval suite can compare them without re-ingesting.
- Semantic chunking costs one embedding call per sentence at ingest time. That is acceptable for an internal corpus, but it is the slowest strategy.
- The heading-path prefix gives `recursive` (and `semantic`) an advantage on short reference sections. That is intentional, because it is part of what "structure-aware" means. The baseline deliberately omits it.
- Two loader-level choices support all strategies: headings are tracked as full paths, and PDF sections are split per page, so every chunk can cite an exact page.

## Alternatives considered

- **Token-based sizing.** It is more precise for LLM context budgets, but it ties chunking to one tokenizer. Characters are close enough at this scale. This can be revisited if generation uses a model with a small context window.
- **Adjacent-sentence similarity for semantic chunking.** This was tried first. Single sentences are too noisy, and smoothing each sentence with its neighbours blurred the boundary by one sentence. Comparing windows before and after each gap, with local-minimum selection, fixed both problems; this is covered by `test_semantic_chunker_splits_on_topic_change`.
