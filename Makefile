check:
	python -m pytest -s --cov

lint:
	ruff check .
	ruff format --check .
	mypy -p model_values

html:
	python -m mkdocs build
