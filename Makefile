py_dirs := business_bayes
py_files = $(wildcard business_bayes/*.py)

.PHONY: fmt
fmt: env_ok
	env/bin/isort -sp .isort.cfg $(py_dirs)
	env/bin/black $(py_files)


.PHONY: test
test: check
	env/bin/python -m unittest discover $(py_dirs) -p "*.py" -v


.PHONY: check
check: env_ok
	env/bin/python -m mypy \
		--check-untyped-defs \
		--ignore-missing-imports \
		$(py_dirs)

	env/bin/python -m flake8 --select F $(py_dirs)
	env/bin/isort  -sp .isort.cfg  $(py_dirs) --check
	env/bin/black --check $(py_files)

env_ok: requirements.txt
	rm -rf env env_ok
	python3 -m venv env
	env/bin/pip install -r requirements.txt
	touch env_ok

run_notebook: env_ok
	env/bin/jupyter notebook

clean:
	rm -rf env_ok env
