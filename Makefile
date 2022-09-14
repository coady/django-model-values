check:
	python3 -m pytest -s --cov

lint:
	black --check .
	flake8 --exclude .venv --ignore E501
	mypy -p model_values

html:
	python3 -m mkdocs build
