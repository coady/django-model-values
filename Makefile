check:
	python -m pytest -s --cov

lint:
	black --check .
	ruff .
	mypy -p model_values

html:
	python -m mkdocs build
