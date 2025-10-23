.PHONY: help install install-dev test test-unit test-integration lint typecheck format clean build run docker-build docker-run all

# Default target
help:
	@echo "Mock ML Job Simulator - Development Commands"
	@echo ""
	@echo "Setup:"
	@echo "  make install          Install runtime dependencies"
	@echo "  make install-dev      Install development dependencies"
	@echo ""
	@echo "Testing:"
	@echo "  make test             Run all tests (unit + integration)"
	@echo "  make test-unit        Run unit tests with coverage"
	@echo "  make test-integration Run integration tests (requires Docker image)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint             Run ruff linter"
	@echo "  make typecheck        Run mypy type checker"
	@echo "  make format           Auto-fix linting issues"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build     Build Docker image"
	@echo "  make docker-run       Run Docker container"
	@echo ""
	@echo "Combined:"
	@echo "  make all              Run all checks (lint, typecheck, unit tests, build, integration tests)"
	@echo "  make ci               Run all checks as CI does"
	@echo ""
	@echo "Other:"
	@echo "  make clean            Remove generated files and caches"
	@echo "  make run              Run simulator locally (Python)"

# Setup targets
install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

# Testing targets
test-unit:
	pytest test_training_simulator.py --cov=training_simulator --cov-report=term --cov-report=html -v

test-integration:
	pytest test_integration.py -v

test: test-unit test-integration

# Code quality targets
lint:
	ruff check .

typecheck:
	mypy training_simulator.py test_training_simulator.py test_integration.py

format:
	ruff check --fix .

# Docker targets
docker-build:
	docker build -t mock-ml-job:test .

docker-run:
	docker run --rm \
		-e TOTAL_EPOCHS=2 \
		-e BATCHES_PER_EPOCH=5 \
		-e WRITE_INTERVAL=2 \
		-v $$(pwd)/metrics:/shared/metrics \
		mock-ml-job:test

# Combined targets
all: lint typecheck test-unit docker-build test-integration
	@echo ""
	@echo "✓ All checks passed!"

ci: lint typecheck test-unit docker-build test-integration
	@echo ""
	@echo "✓ CI checks passed!"

# Utility targets
clean:
	rm -rf __pycache__ .pytest_cache .mypy_cache .ruff_cache htmlcov .coverage coverage.xml
	rm -rf metrics
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

run:
	python training_simulator.py
