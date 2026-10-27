# Security Policy

## Supported versions

Only the latest release on `main` receives security fixes.

## Reporting a vulnerability

Please **do not** open a public issue for security problems. Report them privately through [GitHub security advisories](https://github.com/pentiumcoder/hy-rag/security/advisories/new). You should get a response within 7 days.

Include:

- a description of the issue and its impact;
- steps to reproduce, or a proof of concept;
- the affected version or commit.

## Scope notes

- hy-rag runs locally by default and sends no data to third parties. If you configure a hosted LLM or embedding provider, document text is sent to that provider.
- Never commit API keys. Use a local `.env` file (it is gitignored) based on `.env.example`.
- The sample corpus in `data/corpus/` is fictional. Any credential-like strings in it are placeholders.
