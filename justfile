version := `python3 -c "from src.sfem.__about__ import __version__; print(__version__)"`

# Cleans the repo.
clean:
	@find . | grep -E "(__pycache__|\.pyc|\.pyo|build|generated$)" | xargs rm -rf
	@rm -rf src/*.egg-info/ build/ dist/ .tox/

# Applies formatting to all files.
format:
	isort --profile black .
	black .
	blacken-docs

# Lints all files.
lint: format
	flake8 .

# Runs all tests.
test:
	pytest --cov=src/sfem --cov-report=term --cov-report=xml tests/
