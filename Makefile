check:
	python -m pytest -s --cov

lint:
	ruff .
	ruff format --check .
	mypy -p model_values

html:
	python -m mkdocs build
