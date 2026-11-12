# Agent instructions for finsent-calib

This repository is public and is meant to be shown on a resume. Keep it clean and
reproducible.

## Plan of record

The plan and progress tracker is a Notion page (Project Roadmap: Calibrated Financial
Sentiment Signal with a Fine-tuned Small LLM). Code, configs and results live here; the plan
and the decision and hours logs live there.

## Rules

- Work one roadmap step at a time. If a step fails or an assumption turns out wrong, stop
  and ask before changing direction.
- Fix random seeds everywhere and log them. Never tune anything on the test set.
- Check MiniMind script names and paths against its current README before use.
- Never commit data, model weights or checkpoints. Dataset licenses (Financial PhraseBank is
  CC BY-NC-SA 3.0) forbid redistribution.
- Never commit secrets, API keys, tokens or personal paths.

## Environment

- uv only: `uv add <pkg>`, `uv run ...`. No pip, no conda, no `requirements.txt`.
- Python version is pinned in `.python-version`. Commit `pyproject.toml` and `uv.lock`.
- Colab notebooks install pinned versions so local and Colab runs match.

## Code style

- Package code goes in `src/finsent_calib/`, tests in `tests/`.
- Lint and format with ruff before committing: `uv run ruff check . && uv run ruff format .`
- Type hints and short docstrings on public functions.
- Commit messages: English, imperative mood, Conventional Commits style
  (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`).
