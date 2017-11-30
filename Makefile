all: check html

clean:
	make -C docs $@
	hg st -in | xargs rm
	rm -rf build dist django_model_values.egg-info

html:
	make -C docs $@ SPHINXOPTS=-W
	rst2$@.py README.rst docs/_build/README.$@

dist: html
	python3 setup.py sdist bdist_wheel

check:
	python3 setup.py $@ -mrs
	flake8
	pytest-2.7 --cov
	pytest --cov --cov-append --cov-fail-under=100
