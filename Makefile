install:
	pip install --upgrade pip &&\
		pip install -r requirements.txt

test:
	

format:
	isort *.py
	black *.py

run:
	

lint:
	

all: install format lint