.PHONY: check lint format-check test

check: lint format-check test

lint:
	uv run ruff check .

format-check:
	uv run ruff format --check .

test:
	uv run pytest -q
