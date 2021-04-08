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