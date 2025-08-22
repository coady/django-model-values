check:
	uv run pytest -s --cov

lint:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy -p model_values

html:
	uv run mkdocs build
