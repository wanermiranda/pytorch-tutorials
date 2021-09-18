create-env:
	conda create -n pytorch python=3.8

requirements:
	pip install -r requirements.txt

neat:
	black wompth
	mypy wompth/
