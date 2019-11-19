test:
	python -m pytest --cov-report term-missing --color=yes --cov=motion tests

lint:
	black .

doc:
	cp readme.md docs/index.md; \
	mkdocs gh-deploy

clean:
	rm -rf .pytest_cache .coverage site