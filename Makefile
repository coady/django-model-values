check:
	python3 -m pytest -s --cov

lint:
	black --check .
	flake8 --ignore E501 model_values tests
	mypy -p model_values

html:
	python3 -m mkdocs build
