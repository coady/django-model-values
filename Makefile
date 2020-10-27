all: check
	mkdocs build

check:
	python3 setup.py $@ -ms
	black -q --check .
	flake8
	mypy -p model_values
	pytest --cov --cov-fail-under=100
