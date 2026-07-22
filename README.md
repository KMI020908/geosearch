# GeoSearch

Multilingual toponym recognition and matching system.

Given free-text input, the system identifies city and populated-place mentions and returns a ranked list of matched GeoNames entries.

**Languages:** Russian, English, Turkish, Chinese  
**Countries:** Russia (RU), USA (US), Turkey (TR), China (CN)  
**Data source:** [GeoNames](https://www.geonames.org/) open database

---

## Architecture

```text
Input text → NER (GLiNER) → BM25 retrieval (char n-grams) → CatBoost reranker → Ranked results
```

The pipeline is served as a FastAPI app, built once in the `lifespan` and shared
read-only across requests. On startup the search engine extracts a corpus of
`(geonameid, name_variant)` pairs from PostgreSQL, builds a character n-gram BM25
index (`rank_bm25`, IDF pinned to 1 so it's pure term-frequency saturation —
deliberately tolerant of transliteration and spelling differences across
ru/en/tr/zh — cached to disk), loads the GLiNER model, and loads the trained
CatBoost reranker if one has been trained (`data/rerank_model.cbm`).

Per request: GLiNER extracts place spans from the text, BM25 retrieves the
best-matching *names* (retrieval is name-level, not place-level — homonyms
like "Moscow" share one BM25 document backed by every place with that name),
and the candidate geonameids are hydrated from PostgreSQL. **Retrieval then
ends** by ranking those candidates on their BM25 `retriever_score`, breaking
ties by population, and truncating to the top *k*. The reranker is applied
**strictly afterwards**, as a pure reordering of exactly those top-*k*
candidates — so retrieval, not the reranker, fixes *which* places reach the
top *k*; the reranker only changes their order, never the set (recall@k is set
by retrieval). If no reranker has been trained yet, or `use_rerank=false` on
`/v1/search`, results keep retriever order with the population tiebreak.

> BiEncoder/FAISS hybrid retrieval is planned but not yet implemented.

### Synthetic query dataset

`src/dataset/` generates synthetic queries about cities, plus the sample plan
they were drawn from. The dataset feeds several pipeline steps (reranker
fine-tuning, retriever evaluation, …), not just reranking. The flow is a Polars
pipeline feeding a DeepSeek call per query:

1. `sampling.py` loads every name spelling (canonical, ASCII, and in-scope
   alternate names) from PostgreSQL, groups them by `(name, language)`, and — for
   each `(language, country)` — takes a **population-stratified** sample (most-
   and least-populous names plus a random middle) so the dataset spans big cities
   and obscure villages. Each sampled name gets a random query *style* and *topic*.
2. `generate.py` sends one DeepSeek request per row and stores the returned query.
   Requests run **concurrently** (`DATASET__MAX_WORKERS` in flight), show a **tqdm**
   progress bar, and every completed row is appended to a JSONL **checkpoint**.
   Re-running **warm-starts** from that checkpoint, so an API outage mid-run only
   costs the unfinished rows. The final Parquet is assembled from the checkpoint,
   and the sample plan is saved next to it.

```bash
export DEEPSEEK_API_KEY=sk-...   # or add it to .env
make dataset                     # -> data/query_dataset.parquet + data/query_plan.parquet
```

### Reranker

`src/rerank/` turns the synthetic query dataset into a trained CatBoost ranker
that reorders BM25's candidates. It's a two-stage pipeline, gated by a golden-set
comparison:

1. **`dataset.py`** (`make rerank-data`) replays every query in
   `data/query_dataset.parquet` through the *live* search API with
   `use_rerank=false`, so the training data never depends on the reranker it's
   about to train. For each query the gold geonameid(s) are labelled positive and
   every other retrieved candidate negative. Each candidate is featurised by the
   shared `src/rerank/features.py::build_row` — the *same* builder the online
   reranker uses, so train- and serve-time features are identical by construction.
   The features are the NER `entities` and the candidate `document` text (both
   handed to CatBoost as text features) plus `log_population` (the dominant
   tiebreak signal, log-scaled), `retriever_score`, and `retriever_rank`. The
   `entities` — not the raw query — are used because that is exactly what the
   engine feeds the reranker online; the `document` is `build_descriptions()`
   (sorted name spellings + country + admin1 region). To change the feature set,
   edit `src/rerank/features.py`. The result is split **by query** — the ranking
   group — into `data/rerank_train.parquet` / `data/rerank_test.parquet`, so a
   query (and its gold) never leaks across the split. Splitting by geonameid
   instead would tear a query's candidates across both sets, leaking the query and
   leaving truncated (or positive-less) test groups.
2. **`train.py`** (`make rerank-train`) fits a CatBoost `YetiRank` listwise ranker
   over those features, grouped per query, with early stopping on
   NDCG@`RERANK__NDCG_TOP` over the held-out queries. The trained model is saved
   to `data/rerank_model.cbm`.

`src/rerank/model.py::Reranker` is the inference-side wrapper `SearchEngine`
loads at startup; it featurises candidates with the same `build_row`.

```bash
uv run uvicorn src.api.main:app --reload &   # rerank-data / rerank-eval need the API up
make rerank                                   # rerank-data -> rerank-train
make rerank-eval                              # golden-set: rerank vs baseline vs ideal
```

**Golden-set gate.** `evaluate.py` (`make rerank-eval`) replays the curated
`data/golden_set.parquet` through the API with the reranker off and on, and also
computes the *ideal* (oracle) reordering, printing `RR / P@1 / R@k` for all three.
The reranker only earns its place if it clears the retriever baseline — since it
merely reorders the retrieved top-k, `R@50` never changes; the win shows up in the
top-heavy `P@1` / `RR`. If it's below baseline, keep `use_rerank=false`.

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

The `app` service bind-mounts `./data` to `/app/data` — the BM25 index cache,
the reranker model, and the datasets the Makefile writes on the host are the
same files the container sees. GLiNER weights are cached separately in the
named `hf_cache` volume. `INDEX_WARM_START=true` in `docker-compose.yml` means
the container loads the cached index/reranker instead of rebuilding on every
restart; after retraining a new reranker on the host (`make rerank`), restart
the `app` service (`docker compose restart app`) to pick up the new
`data/rerank_model.cbm` — it's only loaded once, at startup. The database must
already be populated via `make etl`.

---

## API

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/v1/search?text=...&top_k=50&use_rerank=true` | Extract place mentions from `text` and return ranked GeoNames matches |
| `GET` | `/health` | Liveness probe (unversioned) |

`use_rerank` (default `true`) reorders the retrieved top-*k* candidates with the
trained reranker if one is loaded; `false` returns plain retriever order +
population tiebreak — used by `src/rerank/dataset.py` to mine negatives
independent of the reranker.

```bash
curl 'http://localhost:8000/v1/search?text=I%20flew%20from%20Moscow%20to%20Istanbul&top_k=10'
```

The response contains the original `query`, the `entities` GLiNER extracted, the
ranked `results` (geonameid, asciiname, country, population, coordinates,
`score`, `retriever_score`), and a `total` count. `score` is the reranker's
model score when `use_rerank=true`, otherwise it equals `retriever_score` — the
raw per-query character-n-gram BM25 term-frequency score for the matched name
(shared by every homonym under it), always exposed alongside `score` as a
reranker training feature.

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

### Dataset

| Target | Description |
|--------|-------------|
| `make dataset` | Generate the synthetic query dataset + sample plan (DeepSeek → Parquet); warm-starts from the checkpoint if re-run |

### Reranker

| Target | Description |
|--------|-------------|
| `make rerank-data` | Mine labelled `(query, document, label)` pairs by replaying the query dataset through the live search API (needs `uvicorn`/Docker `app` up) |
| `make rerank-train` | Fit the CatBoost `YetiRank` reranker on the mined pairs → `data/rerank_model.cbm` |
| `make rerank-eval` | Golden-set metrics: rerank vs retriever baseline vs ideal ceiling (needs the API up) |
| `make rerank` | Full pipeline: `rerank-data` → `rerank-train` |

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
    │   ├── engine.py           # SearchEngine: GLiNER NER + retrieval + rerank
    │   ├── bm25.py             # BM25 char-n-gram retrieval index (rank_bm25)
    │   └── tokenizer.py        # character n-gram tokenizer
    ├── dataset/
    │   ├── sampling.py         # Polars sample plan: names, homonyms, stratified pick
    │   └── generate.py         # DeepSeek generation: concurrency + warm-start checkpoint
    ├── rerank/
    │   ├── features.py         # shared feature builder (train/serve parity)
    │   ├── dataset.py          # mine labelled feature rows from the live API
    │   ├── train.py            # fit the CatBoost YetiRank ranker, NDCG@k eval
    │   ├── evaluate.py         # golden-set: rerank vs baseline vs ideal
    │   └── model.py            # Reranker: inference-side scoring wrapper
    └── api/
        ├── main.py             # FastAPI app + lifespan (builds engine once)
        ├── routes.py           # /search and /health endpoints
        ├── deps.py             # DB session + engine dependencies
        └── schemas.py          # Pydantic request/response models
tests/
├── etl/
│   └── test_parser.py          # unit tests for parser (no DB required)
└── rerank/
    └── test_dataset.py         # unit tests for split() / build_pairs() (no DB/API required)
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
| `DEEPSEEK_API_KEY` | _(none)_ | DeepSeek API key for `make dataset` (required only for that target) |
| `DATASET__DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model used to generate queries |
| `DATASET__MAX_WORKERS` | `8` | Concurrent in-flight requests during dataset generation |
| `DATASET__OUTPUT_PATH` | `data/query_dataset.parquet` | Where the generated dataset is written |
| `DATASET__PLAN_PATH` | `data/query_plan.parquet` | Where the sample plan is written |
| `RERANK__SEARCH_URL` | `http://localhost:8000/v1/search` | Live search API `rerank-data` replays queries against |
| `RERANK__TEST_SIZE` | `0.2` | Fraction of distinct queries held out for test (split by query — the ranking group — never by geonameid) |
| `RERANK__GOLDEN_SET_PATH` | `data/golden_set.parquet` | Curated `(query, gold geonameIds)` set for `make rerank-eval` (never used for training) |
| `RERANK__MODEL_PATH` | `data/rerank_model.cbm` | Where the trained reranker is saved/loaded from |
| `RERANK__NDCG_TOP` | `10` | `k` for the NDCG@k early-stopping/eval metric |

> Other dataset tunables (`DATASET__N_TOP`, `DATASET__N_MID`, `DATASET__N_LOW`, `DATASET__STYLE_WEIGHTS`,
> `DATASET__TOPICS`, `DATASET__TEMPERATURE`, `DATASET__REASONING_EFFORT`,
> `DATASET__MAX_RETRIES`, `DATASET__CHECKPOINT_PATH`, …) live in `DatasetConfig`, and other reranker
> tunables (`RERANK__TOP_K`, `RERANK__ITERATIONS`, `RERANK__LEARNING_RATE`, `RERANK__L2_LEAF_REG`,
> `RERANK__EARLY_STOPPING_ROUNDS`, …) live in `RerankConfig`, both in `src/config.py`.

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

Tests in `tests/etl/test_parser.py` and `tests/rerank/test_dataset.py` use only
in-memory fixtures — no database, live API, or network connection required.
