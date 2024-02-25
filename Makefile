black:
	python -m black .
	python -m black stupid.py

gitall:
	git add .
	@read -p "Enter commit message: " message; 	git commit -m "$$message"
	git push


try:
	@echo $$FOO

stupid:
	python -W ignore stupid.py

test:
	pytest testCN/testCN.py


install_requirements:
	pip install -r requirements.txt


notebook_memory_usage:
	conda install -c conda-forge jupyter-resource-usage
	jupyter serverextension enable --py jupyter-resource-usage --sys-prefix
	jupyter nbextension install --py jupyter-resource-usage --sys-prefix
	jupyter nbextension enable --py jupyter-resource-usage --sys-prefix

install_some_packages:
	conda install pip
	pip install jedi==0.17.2

test:
	python -m pytest tests

computationalPerformance:
	python ComputationalPerformance.py
	python experiments/analyzeComputationalPerformance.py

q:
	squeue -u $(USER)


clean:
	rm logs/*
