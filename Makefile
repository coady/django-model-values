check:
	python3 -m pytest --cov

lint:
	black --check .
	flake8
	mypy -p model_values

html:
	python3 -m mkdocs build
