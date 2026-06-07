.PHONY: up down migrate migrate-gen migrate-up download load etl update test lint typecheck clean help

PYTHON := .venv/bin/python

help:
	@echo "GeoSearch — available targets:"
	@echo ""
	@echo "  Infrastructure"
	@echo "    up          Start PostgreSQL container"
	@echo "    down        Stop and remove PostgreSQL container"
	@echo ""
	@echo "  Database"
	@echo "    migrate     Generate + apply Alembic migrations"
	@echo "    migrate-gen Generate migration only (inspect before applying)"
	@echo "    migrate-up  Apply pending migrations"
	@echo ""
	@echo "  ETL pipeline"
	@echo "    download    Download raw GeoNames files (country zips + alternateNames)"
	@echo "    load        Load downloaded files into PostgreSQL"
	@echo "    etl         Full pipeline: download → load"
	@echo "    update      Apply today's GeoNames daily delta"
	@echo ""
	@echo "  Development"
	@echo "    test        Run test suite"
	@echo "    lint        Run ruff linter + formatter check"
	@echo "    typecheck   Run pyright type checker"
	@echo "    clean       Remove downloaded raw data files"

# ---------------------------------------------------------------------------
# Infrastructure
# ---------------------------------------------------------------------------

up:
	docker compose up -d
	@echo "Waiting for PostgreSQL to be ready..."
	@until docker compose exec postgres pg_isready -U geosearch > /dev/null 2>&1; do sleep 1; done
	@echo "PostgreSQL is ready."

down:
	docker compose down

# ---------------------------------------------------------------------------
# Database migrations
# ---------------------------------------------------------------------------

migrate: migrate-gen migrate-up

migrate-gen:
	$(PYTHON) -m alembic revision --autogenerate -m "$(or $(MSG),auto)"

migrate-up:
	$(PYTHON) -m alembic upgrade head

# ---------------------------------------------------------------------------
# ETL pipeline
# ---------------------------------------------------------------------------

download:
	$(PYTHON) -m src.etl.downloader

load:
	$(PYTHON) -m src.etl.loader

etl: download load

update:
	$(PYTHON) -m src.etl.updater

# ---------------------------------------------------------------------------
# Development
# ---------------------------------------------------------------------------

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check src tests
	$(PYTHON) -m ruff format --check src tests

typecheck:
	$(PYTHON) -m pyright src

clean:
	rm -rf data/raw/*.zip data/raw/*.txt data/raw/deltas/
