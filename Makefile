test:
	python -m pytest --color=yes --cov=motion tests

lint:
	black .

doc:
	mkdocs gh-deploy

clean:
	rm -rf .pytest_cache .coverage site