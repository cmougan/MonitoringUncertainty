.PHONY: install test
black:
	python -m black .
	python -m black stupid.py

gitall:
	git add .
	git commit -m $$m
	git push


try:
	@echo $$FOO

stupid:
	python -W ignore stupid.py

test:
	pytest testCN/testCN.py


export_requirements:
	conda list --export > requirements.txt

install_requirements:
	conda install --file requirements.txt

notebook_memory_usage:
	conda install -c conda-forge jupyter-resource-usage
	jupyter serverextension enable --py jupyter-resource-usage --sys-prefix
	jupyter nbextension install --py jupyter-resource-usage --sys-prefix
	jupyter nbextension enable --py jupyter-resource-usage --sys-prefix

install_some_packages:
	conda install pip
	pip install jedi==0.17.2



default: test



test:
    PYTHONPATH=. pytest
