# ADR-0005: Deployment: one API process owns the indexes; small models on CPU by default

- **Status:** Accepted
- **Date:** 2026-10-31

## Context

The project has to run with one command for someone evaluating it, on hardware we don't control:

- a laptop with no GPU, or a workstation with an NVIDIA card
- Docker Desktop on Windows or macOS, or Docker on Linux

It also needs a dashboard and an API, and the models must not be baked into images. The Ollama image alone is 8.5 GB on disk, because it bundles the GPU libraries.

The original plan listed four services: `api`, `chroma`, `ollama` and `frontend`.

## Decision

1. **Three services, not four. Chroma runs embedded in the API container.**
   - `IndexManager` is the single writer for Chroma and BM25 together (ADR-0002). It keeps a manifest and repairs one index from the other when they drift. A separate Chroma server would split that consistency boundary across processes, with the BM25 index still living in the API.
   - The Chroma 1.x failure described in the 0.4.0 changelog showed how much depends on one process owning the index. That failure is a vector segment lost right after a write, now handled with an exact-search fallback.
   - A Chroma server can be added later behind `VectorStore` if the index outgrows one machine. At this corpus size it would only add a network hop.
2. **Small models by default; the full-size models in a GPU file.**
   - On CPU, `docker compose up` uses `qwen2.5:3b` to answer and `llama3.2:3b` to judge. That is about 4 GB in total and keeps the generator and the judge in different model families (ADR-0003).
   - `docker-compose.gpu.yml` reserves an NVIDIA GPU and switches to the evaluated `qwen2.5:7b` + `llama3.1:8b`. Model names stay overridable with `HYRAG_GENERATOR_MODEL` and `HYRAG_JUDGE_MODEL`.
3. **Models are pulled on first boot, not built into images.**
   - `scripts/seed.py` runs before the API starts. It indexes the corpus, then pulls any model Ollama is missing, with progress in the logs.
   - Both steps are idempotent: an unchanged corpus is not re-embedded and present models are not pulled again, so restarts take seconds.
   - Models, embedding caches, indexes and uploads live in named volumes.
4. **The dashboard talks to the API through its own origin.** nginx serves the built dashboard and proxies `/api/` to the API, and Vite does the same in development. The browser never makes cross-origin calls, and the API's CORS setting only matters for other tools.
5. **Health checks gate start-up order.**
   - The API reports healthy only once its index is non-empty and the embedder matches it, so the dashboard starts only after seeding has finished.
   - The API's start period is 10 minutes, because a first boot on a slow connection includes the model downloads.

## Consequences

- One `docker compose up --build` gives a working system on any machine with Docker. Measured on a laptop CPU: about 45 s per answer with the 3b models, and 15 s for a restart because nothing is re-embedded or re-pulled. That's fine for trying it out, but the latency figures in the README and the eval are for the GPU setup.
- The 3b models are not what the evaluation measured. Their answer quality is lower and has not been evaluated; `hyrag eval` can be run against them with the same golden set.
- Because the API holds the index in-process, it is one replica. Horizontal scaling would need the index split out (point 1) and the LLM calls queued. Neither matters for a local tool.

## Alternatives considered

- **A Chroma server container.** It was rejected for the consistency reason in point 1.
- **Baking models into the images.** That makes 5–10 GB images and slow CI builds, and the models would be downloaded even by people who point the API at an existing Ollama.
- **Using the host's Ollama from the container.** It's possible: set `HYRAG_OLLAMA_BASE_URL=http://host.docker.internal:11434`. But that depends on the host's setup, so it's documented as an option rather than made the default.
