.PHONY: install install-dev install-train vendor train evaluate test test-cov lint format run dev docker-build docker-run deploy-check clean

PYTHON ?= python
PIP ?= pip
PROJECT = bank-statement-system

install:
	$(PIP) install -r requirements.txt

install-dev:
	$(PIP) install -r requirements-dev.txt

install-train:
	$(PIP) install -r requirements-train.txt

vendor:
	$(PYTHON) scripts/download_vendors.py

train:
	$(PYTHON) training/train.py --train-file samples/test_dataset/classification_data/classification_train_labeled.csv --holdout-file samples/test_dataset/classification_data/classification_holdout_test_noisy.csv --output-dir models/v1

evaluate:
	$(PYTHON) scripts/evaluate_dataset.py --dataset-dir samples/test_dataset --output-dir reports

test:
	pytest tests/ -v --tb=short

test-cov:
	pytest tests/ -v --tb=short --cov=core --cov=app --cov-report=html --cov-report=term-missing

lint:
	ruff check core/ app/ training/ scripts/ tests/

format:
	black core/ app/ training/ scripts/ tests/

run:
	uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

dev:
	LOG_LEVEL=DEBUG uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload

docker-build:
	docker build -t $(PROJECT) .

docker-run:
	docker run -p 8000:8000 --env-file .env.example $(PROJECT)

deploy-check:
	vercel dev

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; \
	find . -name "*.pyc" -delete 2>/dev/null; \
	rm -rf .pytest_cache htmlcov .coverage

