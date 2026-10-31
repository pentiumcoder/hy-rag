# Demo script (under 4 minutes)

A shot list for a screen recording. Start with the stack running (`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`), the dashboard open at <http://localhost:8080>, and these questions asked once beforehand so the answers come from the cache and the video doesn't wait on the model.

| # | Time | Screen | Say |
|---|---|---|---|
| 1 | 0:00–0:20 | README, scrolled to the results table | "This is a RAG system over a company's internal docs: runbooks, configs, policies, postmortems. It runs fully local on a laptop GPU, and every number here comes from a 55-question evaluation." |
| 2 | 0:20–0:55 | Ask tab: "What error do I get when a batch exceeds ingest.max_batch_size?" | "Each sentence cites a passage. Click `[1]` and it scrolls to the passage and highlights it. Under that, a second model, from a different family, checked every sentence against the passage it cites." |
| 3 | 0:55–1:35 | Turn on **Compare hybrid vs dense**, turn off **Rerank**, ask `NW-4290` | "Error codes are where embeddings fail. Dense search alone ranks this error's own page tenth; it's guessing. Keyword search finds it first, and the hybrid combines both. Look at the score columns: dense, BM25, fusion, rerank, all visible." |
| 4 | 1:35–2:10 | Ask "What should I do when CertExpiringSoon fires?" | "Here the model cited two irrelevant passages along with the right one. The judge flagged them in red, and the confidence score drops. You see the mistake instead of trusting it." |
| 5 | 2:10–2:50 | Ask "How much does the Team plan cost?", then "Who won the 2022 World Cup?" | "There is no Team plan. The pricing page is retrieved and looks relevant, but the model says the documents don't answer it, and shows what it found, what's missing and where to look. The football question never reaches the model at all: retrieval confidence is near zero." |
| 6 | 2:50–3:25 | Evaluation tab | "Hybrid plus reranking finds 98.9% of the evidence. Answers are 91% correct with 97.5% of sentences supported. Wrong answers shown drop from 41 to 6 out of 220 with the abstention layers, and those thresholds were tuned on the data, not guessed." |
| 7 | 3:25–3:50 | Documents tab: drag in a Markdown file, then ask about it | "Upload a document and it's chunked, deduplicated, embedded and searchable in a few seconds." |
| 8 | 3:50–4:00 | Terminal: `docker compose up` | "One command, no API keys, zero cost per query. The case study in the repo walks through every decision and the numbers behind it." |

**Tips:**
- Record at 1440×900 or larger; the side-by-side view needs the width.
- For shot 7, prepare a small Markdown file with a distinctive term, for example a fictional on-call tool named "Q-77", so the answer obviously comes from the upload.
