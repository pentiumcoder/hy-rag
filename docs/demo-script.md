# Demo script (under 4 minutes)

A shot list for a screen recording. Start with the stack running (`docker compose -f docker-compose.yml -f docker-compose.gpu.yml up`), the dashboard open at <http://localhost:8080>, and these questions asked once beforehand so the answers come from the cache and the video doesn't wait on the model.

| # | Time | Screen | Say |
|---|---|---|---|
| 1 | 0:00–0:20 | README, scrolled to the results table | "This answers questions over a set of internal docs: runbooks, configs, policies, postmortems. It runs locally on a laptop GPU. The numbers come from 55 test questions I wrote alongside the sample docs, so they're a best case." |
| 2 | 0:20–0:55 | Ask tab: "What error do I get when a batch exceeds ingest.max_batch_size?" | "Each sentence cites a passage. Click `[1]` and it scrolls to the passage and highlights it. Under that, a second model, from a different family, checked every sentence against the passage it cites." |
| 3 | 0:55–1:35 | Turn on **Compare hybrid vs dense**, turn off **Rerank**, ask `NW-4290` | "Error codes are where embeddings struggle. Dense search alone ranks this error's own page tenth, so it declines to answer. Keyword search finds it first. The columns show each stage's score." |
| 4 | 1:35–2:10 | Ask "What should I do when CertExpiringSoon fires?" | "Here the model cited two irrelevant passages along with the right one. The checker crosses them out, and the citation part of the confidence score drops with them." |
| 5 | 2:10–2:50 | Ask "How much does the Team plan cost?", then "Who won the 2022 World Cup?" | "There is no Team plan. The pricing page is retrieved and looks relevant, but the model says the documents don't answer it, and shows what it found, what's missing and where to look. The football question never reaches the model at all: retrieval confidence is near zero." |
| 6 | 2:50–3:25 | Evaluation tab | "On this test set BM25 alone is as good as hybrid, which surprised me; the questions reuse the documents' wording. Dense on its own is clearly worse. With the declining rules, wrong answers shown went from 41 to 6 out of 220." |
| 7 | 3:25–3:50 | Documents tab: drag in a Markdown file, then ask about it | "Upload a document and it's chunked, deduplicated, embedded and searchable in a few seconds." |
| 8 | 3:50–4:00 | Terminal: `docker compose up` | "One command, no API keys. The case study in the repo has the details and the known gaps." |

Notes:

- Record at 1440×900 or larger; the side-by-side view needs the width.
- For shot 7, prepare a small Markdown file with a distinctive term, for example a fictional on-call tool named "Q-77", so the answer obviously comes from the upload.
