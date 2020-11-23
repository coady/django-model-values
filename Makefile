check:
	pytest --cov

lint:
	python3 setup.py check -ms
	black --check .
	flake8
	mypy -p model_values

html:
	PYTHONPATH=$(PWD):$(PYTHONPATH) mkdocs build
