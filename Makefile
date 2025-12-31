check:
	uv run pytest -s --cov

lint:
	uvx ruff check
	uvx ruff format --check
	uvx ty check model_values

html:
	uv run --group docs zensical build
