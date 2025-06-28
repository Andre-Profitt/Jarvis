.PHONY: help install install-dev test lint format clean run setup-poetry check-deps update-deps security-scan

# Default target
help:
	@echo "JARVIS Ecosystem - Development Commands"
	@echo "======================================"
	@echo "install        Install production dependencies with Poetry"
	@echo "install-dev    Install all dependencies including dev tools"
	@echo "test           Run test suite with coverage"
	@echo "lint           Run all linters (black, isort, flake8, mypy)"
	@echo "format         Auto-format code with black and isort"
	@echo "clean          Remove build artifacts and cache files"
	@echo "run            Launch JARVIS system"
	@echo "setup-poetry   Install Poetry if not present"
	@echo "check-deps     Check for dependency conflicts and vulnerabilities"
	@echo "update-deps    Update all dependencies to latest compatible versions"
	@echo "security-scan  Run security vulnerability scan"

# Install Poetry if not present
setup-poetry:
	@if ! command -v poetry &> /dev/null; then \
		echo "Installing Poetry..."; \
		curl -sSL https://install.python-poetry.org | python3 -; \
	else \
		echo "Poetry is already installed"; \
	fi

# Install production dependencies
install: setup-poetry
	poetry install --no-dev

# Install all dependencies including dev
install-dev: setup-poetry
	poetry install
	poetry run pre-commit install

# Run tests with coverage
test:
	poetry run pytest -v --cov=core --cov-report=html --cov-report=term

# Run all linters
lint:
	poetry run black --check core/ plugins/ tests/
	poetry run isort --check-only core/ plugins/ tests/
	poetry run flake8 core/ plugins/ tests/
	poetry run mypy core/ plugins/
	poetry run pylint core/ plugins/
	poetry run bandit -r core/ plugins/

# Format code
format:
	poetry run black core/ plugins/ tests/
	poetry run isort core/ plugins/ tests/

# Clean build artifacts
clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	find . -type f -name "*.pyd" -delete
	find . -type f -name ".coverage" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name "htmlcov" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	find . -type d -name "dist" -exec rm -rf {} +
	find . -type d -name "build" -exec rm -rf {} +

# Run JARVIS
run:
	poetry run python LAUNCH-JARVIS-REAL.py

# Check for dependency conflicts
check-deps:
	poetry check
	poetry run pip-audit
	poetry run safety check

# Update dependencies
update-deps:
	poetry update
	poetry export -f requirements.txt --output requirements.txt --without-hashes

# Security vulnerability scan
security-scan:
	poetry run bandit -r core/ plugins/ -f json -o security-report.json
	poetry run safety check --json --output safety-report.json
	@echo "Security reports generated: security-report.json, safety-report.json"

# Development environment setup
dev-setup: install-dev
	@echo "Setting up development environment..."
	poetry run pre-commit install
	@echo "Creating .env file if not exists..."
	@if [ ! -f .env ]; then cp .env.example .env; fi
	@echo "Development environment ready!"

# Docker commands
docker-build:
	docker build -t jarvis-ecosystem:latest .

docker-run:
	docker run -it --rm -p 8765:8765 jarvis-ecosystem:latest

# Generate documentation
docs:
	poetry run sphinx-build -b html docs/ docs/_build/html

# Run specific JARVIS components
run-consciousness:
	poetry run python -m core.consciousness_jarvis

run-websocket:
	poetry run python -m core.websocket_server

# Database management
db-init:
	poetry run python -m core.database init

db-migrate:
	poetry run python -m core.database migrate

# Performance profiling
profile:
	poetry run python -m cProfile -o profile.stats LAUNCH-JARVIS-REAL.py
	poetry run python -m pstats profile.stats