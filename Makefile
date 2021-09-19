create-env:
	conda create -n pytorch python=3.8

requirements:
	pip install -r requirements.txt

neat:
	black wompth tests
	isort wompth/ tests/
	mypy wompth/ tests/

tests: neat
	python -m pytest -s -vv --cov=wompth --cov-report term-missing tests/