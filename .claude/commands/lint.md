Run ruff linter and formatter on the project.

1. Check for lint errors: `uv run ruff check .`
2. Auto-fix safe issues: `uv run ruff check --fix .`
3. Format code: `uv run ruff format .`

Ruff config is in `pyproject.toml` under `[tool.ruff]` (line length 119). Do not change it unless asked.

Review any remaining issues that need manual fixes.
