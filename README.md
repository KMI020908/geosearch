# GeoSearch

Multilingual toponym recognition and matching system.

Given free-text input, the system identifies city and populated-place mentions and returns a ranked list of matched GeoNames entries.

**Languages:** Russian, English, Turkish, Chinese  
**Countries:** Russia (RU), USA (US), Turkey (TR), China (CN)  
**Data source:** [GeoNames](https://www.geonames.org/) open database

---

## Architecture

```text
Input text → NER (GLiNER) → Hybrid Retrieval (BM25 + BiEncoder/FAISS) → Reranker → Ranked results
```

Current stage: **1 — Data Layer** (ETL pipeline into PostgreSQL).

---

## Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/)
- Docker + Docker Compose

---

## Quick start

```bash
# 1. Install dependencies
uv sync --dev
source .venv/bin/activate

# 2. Configure environment
cp .env.example .env          # edit DATABASE_URL if needed

# 3. Start PostgreSQL
make up

# 4. Apply schema migrations
make migrate

# 5. Download GeoNames data and load into DB (~1 GB download, ~20 min)
make etl
```

---

## Makefile targets

### Infrastructure

| Target | Description |
|--------|-------------|
| `make up` | Start PostgreSQL container and wait until ready |
| `make down` | Stop and remove the container |

### Database

| Target | Description |
|--------|-------------|
| `make migrate` | Generate a new migration from model changes and apply it |
| `make migrate-gen MSG="description"` | Generate migration only (review before applying) |
| `make migrate-up` | Apply all pending migrations |

### ETL pipeline

| Target | Description |
|--------|-------------|
| `make download` | Download raw GeoNames files for all configured countries |
| `make load` | Load downloaded files into PostgreSQL |
| `make etl` | Full pipeline: `download` then `load` |
| `make update` | Download and apply today's GeoNames daily delta files |

### Development

| Target | Description |
|--------|-------------|
| `make test` | Run the test suite |
| `make lint` | Run ruff linter and formatter check |
| `make typecheck` | Run pyright type checker |
| `make clean` | Remove downloaded raw data files from `data/raw/` |

---

## Project structure

```text
geosearch/
├── docker-compose.yml          # PostgreSQL 16 service
├── Makefile                    # developer workflow
├── pyproject.toml              # dependencies (uv)
├── alembic/                    # schema migrations
│   └── versions/
└── src/
    ├── config.py               # settings: countries, languages, DB URL
    ├── db/
    │   ├── models.py           # SQLAlchemy ORM: Geoname + AlternateName
    │   └── session.py          # async engine + session factory
    └── etl/
        ├── downloader.py       # download raw files from GeoNames
        ├── parser.py           # parse TSV files into Pydantic models
        ├── loader.py           # bulk-upsert into PostgreSQL
        └── updater.py          # incremental updates from daily delta files
tests/
└── etl/
    └── test_parser.py          # unit tests for parser (no DB required)
```

---

## Configuration

All settings live in `src/config.py` and are overridable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `DATABASE_URL` | `postgresql+asyncpg://geosearch:geosearch@localhost:5432/geosearch` | Async PostgreSQL connection string |
| `GEONAMES_DATA_DIR` | `data/raw` | Directory for downloaded GeoNames files |

### Adding a new country

1. Add the ISO-3166 two-letter code to `countries` in `src/config.py`.
2. Re-run `make etl` (existing data is upserted, not duplicated).

### Adding a new language

1. Add the BCP-47 language code to `languages` in `src/config.py`.
2. Re-run `make load` (the downloader skips already-present zip files).

---

## Running tests

```bash
make test
```

Tests in `tests/etl/test_parser.py` use only in-memory zip fixtures — no database or network connection required.
