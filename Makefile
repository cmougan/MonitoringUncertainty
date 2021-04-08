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


make export_requirements:
	conda list --export > requirements.txt

make install_requirements:
	conda install --file requirements.txt