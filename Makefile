.PHONY : test lint doc nb_to_md clean all

lint:
	black .

test:
	pytest --cov-report term-missing --color=yes --cov=motion tests

nb_to_md:
	jupyter nbconvert --to markdown notebooks/*.ipynb --output-dir='./docs'

doc:
	cp readme.md docs/index.md; \
	mkdocs gh-deploy

clean:
	rm -rf .pytest_cache .coverage site notebooks/.ipynb_checkpoints

all: lint test nb_to_md doc clean
