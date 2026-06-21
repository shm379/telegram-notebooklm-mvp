# Project guide for Claude Code

Telegram + NotebookLM-style ingestion/search bot. It ingests public Telegram
channels, transcribes media, stores transcripts, and lets users search and
summarize them. Written in Python 3.11.

## Layout

- `src/telegram_notebook/` — all application code (bot, pipeline, search,
  embeddings, transcription, db, etc.).
- `tests/` — pytest suite (`test_*.py`), with shared fixtures in `conftest.py`.
- `scripts/` — one-off helper scripts.
- `.github/workflows/` — CI, deploy, and the daily bug-triage automation.

## Build, lint, test

```bash
pip install -r requirements.txt
pip install -e ".[dev]"      # adds pytest + ruff

ruff check src/ tests/        # lint  (line-length 140; rules E, F, I, UP, B)
pytest -q                     # tests (testpaths=tests, pythonpath=src)
```

Both `ruff check` and `pytest -q` must pass before opening a PR. CI runs exactly
these two commands.

## Conventions

- Keep changes minimal and single-purpose; match the surrounding style.
- Tests live in `tests/` and should not require network or real Telegram/LLM
  credentials — mock external services (see existing tests for the patterns).
- Don't hand-edit the generated `CHANGELOG.*.md` files.
- Secrets/config come from environment variables (see `.env.example` and
  `src/telegram_notebook/config.py`); never hard-code credentials.

## Git

- Default branch is `main`; deploys happen on push to `main`, so never commit
  there directly — always work on a branch and open a PR.
