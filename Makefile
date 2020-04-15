.PHONY : test lint doc nb_to_md clean all

lint:
	black .

test:
	pytest --cov-report term-missing --color=yes --cov=pyomeca tests -rxXs

nb_to_md:
	jupyter nbconvert --to markdown notebooks/getting-started.ipynb --output-dir='./docs' --template=docs/nbconvert.tpl

doc:
	sed 's/docs\///g' README.md > docs/index.md; \
	mkdocs gh-deploy

clean:
	rm -rf .pytest_cache .coverage site notebooks/.ipynb_checkpoints

all: lint nb_to_md test doc clean
