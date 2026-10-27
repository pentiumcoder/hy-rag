# Contributing to hy-rag

Thanks for your interest! Issues and pull requests are welcome.

## Development setup

```bash
git clone https://github.com/pentiumcoder/hy-rag.git && cd hy-rag
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pre-commit install
```

## Before opening a pull request

```bash
ruff check . && ruff format --check .
mypy
pytest --cov
```

The test suite runs fully offline with a deterministic fake embedder, so it needs no models, network or API keys.

## Guidelines

- **Commits** follow [Conventional Commits](https://www.conventionalcommits.org/): `feat(retrieval): ...`, `fix(ingestion): ...`, `docs: ...`, `test: ...`, `ci: ...`, `chore: ...`.
- **Keep pull requests focused.** One logical change per PR, with tests.
- **Measure retrieval changes.** If a change touches chunking, retrieval or generation, include before/after numbers from the evaluation suite in the PR description.
- **Record design decisions.** Significant decisions get a short ADR in `docs/adr/`.

## Sample corpus

`data/corpus/` holds documentation for a fictional company, Northwind Cloud. The PDFs are generated from `scripts/pdf_sources/` with `python scripts/build_sample_pdfs.py`; edit the sources, not the PDFs.
