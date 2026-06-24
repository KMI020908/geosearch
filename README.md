# GeoSearch

Multilingual toponym recognition and matching system.

Given free-text input, the system identifies city and populated-place mentions and returns a ranked list of matched GeoNames entries.

**Languages:** Russian, English, Turkish, Chinese  
**Countries:** Russia (RU), USA (US), Turkey (TR), China (CN)  
**Data source:** [GeoNames](https://www.geonames.org/) open database

---

## Architecture

```text
Input text → NER (GLiNER) → BM25 retrieval (char n-grams) → population rerank → Ranked results
```

The pipeline is served as a FastAPI app. On startup the search engine extracts a
corpus of `(geonameid, name_variant)` pairs from PostgreSQL, builds a character
n-gram BM25 index (`rank_bm25`, IDF pinned to 1, cached to disk), and loads the
GLiNER model. Per request: GLiNER extracts place spans from the text, the index
ranks candidate GeoNames entries by BM25 score over shared n-grams, and results
are reranked by population.

> BiEncoder/FAISS hybrid retrieval and a cross-encoder reranker are planned but
> not yet implemented — current reranking is a simple population sort.

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

# 6. Run the API (builds the BM25 index + loads GLiNER on first startup)
uv run uvicorn src.api.main:app --reload
```

Open http://localhost:8000/docs for the interactive Swagger UI.

### Run everything in Docker

```bash
docker compose up --build      # starts PostgreSQL + the API on :8000
```

The `app` service builds the BM25 index and downloads the GLiNER weights on first
startup; both are persisted in named volumes (`app_data`, `hf_cache`) across
restarts. The database must already be populated via `make etl`.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/search?text=...&top_k=50` | Extract place mentions from `text` and return ranked GeoNames matches |
| `GET` | `/health` | Liveness probe (unversioned) |

```bash
curl 'http://localhost:8000/v1/search?text=I%20flew%20from%20Moscow%20to%20Istanbul&top_k=10'
```

The response contains the original `query`, the `entities` GLiNER extracted, the
ranked `results` (geonameid, asciiname, country, population, coordinates), and a
`total` count.

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
├── docker-compose.yml          # PostgreSQL 16 + API service
├── Dockerfile                  # API image (uv + uvicorn)
├── Makefile                    # developer workflow
├── pyproject.toml              # dependencies (uv)
├── alembic/                    # schema migrations
│   └── versions/
└── src/
    ├── config.py               # settings: countries, languages, models, DB URL
    ├── db/
    │   ├── models.py           # SQLAlchemy ORM: Geoname + AlternateName
    │   └── session.py          # async engine + session factory
    ├── etl/
    │   ├── downloader.py       # download raw files from GeoNames
    │   ├── parser.py           # parse TSV files into Pydantic models
    │   ├── loader.py           # bulk-upsert into PostgreSQL
    │   └── updater.py          # incremental updates from daily delta files
    ├── search/
    │   ├── engine.py           # SearchEngine: GLiNER NER + count retrieval + rerank
    │   ├── bm25.py             # BM25 char-n-gram retrieval index (rank_bm25)
    │   └── tokenizer.py        # character n-gram tokenizer
    └── api/
        ├── main.py             # FastAPI app + lifespan (builds engine once)
        ├── routes.py           # /search and /health endpoints
        ├── deps.py             # DB session + engine dependencies
        └── schemas.py          # Pydantic request/response models
tests/
└── etl/
    └── test_parser.py          # unit tests for parser (no DB required)
```

---

## Configuration

All settings live in `src/config.py` and are overridable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `POSTGRES_USER` | `geosearch` | PostgreSQL user (required) |
| `POSTGRES_PASSWORD` | `geosearch` | PostgreSQL password (required) |
| `POSTGRES_DB` | `geosearch` | PostgreSQL database name (required) |
| `DATABASE_URL` | _(built from `POSTGRES_*` for localhost)_ | Optional full async DSN; overrides the default when set (e.g. in Docker) |
| `GEONAMES_DATA_DIR` | `data/raw` | Directory for downloaded GeoNames files |
| `COUNTRIES` | `["RU","US","TR","CN"]` | ISO-3166 country codes to load |
| `LANGUAGES` | `["ru","en","tr","zh"]` | Alternate-name language codes to keep |
| `GLINER_MODEL` | `urchade/gliner_multi-v2.1` | HuggingFace GLiNER model for NER |
| `NER_LABELS` | `["CITY","REGION","STATE","COUNTRY"]` | Entity labels GLiNER extracts |
| `INDEX_PATH` | `data/bm25_index.pkl` | Where the built BM25 index is cached (set to `/data/bm25_index.pkl` in Docker to persist on the volume) |
| `INDEX_WARM_START` | `false` | Load the cached index if present instead of rebuilding |
| `EXCLUDED_FEATURE_CODES` | `["PPLH","PPLQ","PPLW","PPLX"]` | GeoNames feature codes excluded from the corpus |

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
