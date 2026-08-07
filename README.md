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
read-only across requests. **Serving needs no database.** On startup the engine
loads prebuilt artifacts — a character n-gram BM25 index (IDF pinned to 1, so it's
pure term-frequency saturation, deliberately tolerant of transliteration and
spelling differences across ru/en/tr/zh), a table of places to hydrate results
from, and the reranker's candidate documents — then the GLiNER model and the
CatBoost reranker if one has been trained.

**There is no database anywhere in the project.** The pipeline is three file
stages:

```text
raw GeoNames dumps  ->  staging corpus       ->  serving artifacts  ->  Hub
data/raw/*.zip          data/corpus/*.parquet    data/artifacts/
(make download)         (make etl, ~45s)         (make artifacts, ~45s)
```

PostgreSQL used to sit in the middle and was removed: one writer, one reader,
one upstream source, and no query more complex than a filter and a group-by. The
whole corpus is ~58 MB in memory. The artifacts rebuilt from Parquet are
*identical* to those the database produced — same 1,187,541 name groups, same
510,841-entry vocabulary, same place and description tables — so nothing about
retrieval or ranking changed.

The NER model is a **fine-tuned** GLiNER checkpoint (`data/gliner-geosearch/`,
produced by `make ner-train` from the `data/ner/` export of the synthetic query
dataset), not the zero-shot `urchade/gliner_multi-v2.1`. It must be served with the
word splitter it was trained on — `NER_CJK_SPLITTER=true`, see [NER](#ner) below.

Per request: GLiNER extracts place spans from the text, BM25 retrieves the
best-matching *names* (retrieval is name-level, not place-level — homonyms
like "Moscow" share one BM25 document backed by every place with that name),
and the candidate geonameids are hydrated through a `PlaceStore`
(`places.parquet`). **Retrieval then
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
   alternate names) from the staging corpus, groups them by `(name, language)`, and — for
   each `(language, country)` — takes a **population-stratified** sample (most-
   and least-populous names plus a random middle) so the dataset spans big cities
   and obscure villages. Each sampled name gets a random query *style* and *topic*.
2. On top of those plain `one_city` rows, `sampling.py` mines two more kinds:
   * **disambiguation** — `city_admin1` (name + region), `city_country`
     (name + country) and `city_admin1_country`, built from the homonym frames plus a
     `DATASET__FRAC_UNIQUE_DISAMBIGUATION` slice of the one_city names. Their gold
     `geonameid` is **narrowed** to the named region/country instead of covering every
     homonym, so the remaining homonyms become hard negatives. These are also the only
     rows that populate the reranker's `country_entities` / `admin1_entities` features.
   * **multi_city** — one query naming several cities of the same (language, country),
     gold = the union of their geonameids.

   A row that names a region skips any region named after the city itself — otherwise
   the generated query degenerates into "Shanghai, Shanghai", whose region adds no
   disambiguating information. The next most-populous region is used instead. The rule
   applies only to `city_admin1` / `city_admin1_country`; `city_country`, `one_city`
   and `multi_city` name no region and are untouched.
3. `generate.py` sends one DeepSeek request per row and stores the returned query
   together with its **gold NER spans**.

   * **One system prompt per query type** (`src/dataset/prompts.py`). Each
     `sample_source` gets its own prompt stating only its own case, its own input
     fields and its own expected entity set, instead of one shared prompt full of
     "some requests also include a region" conditionals for the model to resolve.
   * **Structured output.** Every response is json — `query`, `entities`
     (`{text, label}` per geographic span, copied out of the query in its inflected
     surface form, as if GLiNER had already run) and `confidence`, the model's own
     estimate that it followed every rule. DeepSeek's json mode has **no**
     server-side schema enforcement, so the contract is enforced on our side:
     Pydantic parsing (`src/dataset/schemas.py`), then `validate_generation`, a
     deterministic check driven by the same `PromptSpec` flags the prompt was built
     from — so instruction and check cannot drift apart. Span offsets come from
     `str.find`, never from the model. Failing rows are retried
     (`DATASET__MAX_GENERATION_ATTEMPTS`) and then persisted with `valid=false` and
     a reason, so recurring breakage is visible instead of silently re-paid for on
     every run; only valid rows reach the Parquet.
   * **Generated in query-type order.** Rows are grouped by `sample_source` and each
     group's first request is sent *alone* before the rest fan out — DeepSeek's
     prefix cache is populated by a *completed* request, so hitting a cold prefix
     with `DATASET__MAX_WORKERS` requests at once would miss on all of them. The
     run logs `prompt_cache_hit_tokens` / `prompt_cache_miss_tokens` per group, so
     the saving is measured rather than assumed.

   Requests run **concurrently** (`DATASET__MAX_WORKERS` in flight), show a **tqdm**
   progress bar, and every completed row is appended to a JSONL **checkpoint**.
   Re-running **warm-starts** from that checkpoint, so an API outage mid-run only
   costs the unfinished rows. Warm start is keyed on `PROMPT_VERSION` as well as
   `request_id`: bump it after changing a prompt and the affected rows regenerate
   instead of being mixed in with the new ones. The final Parquet is assembled from
   the checkpoint, and the sample plan is saved next to it.

   Alongside `entities`, each row carries `gold_city_entities` /
   `gold_country_entities` / `gold_admin1_entities` — the same spans bucketed by
   `src/rerank/features.py::bucket_entities`, i.e. directly comparable to the
   `entity_buckets` the live engine produces from GLiNER. They exist to **evaluate**
   NER and to filter dataset quality. The reranker does **not** train on them by
   default: it keeps mining its features from the live API (see below), which is
   what preserves train/serve parity. `RERANK__USE_GOLD_ENTITIES=true` points the
   mining at these columns instead, as an ablation (see below).

```bash
export DEEPSEEK_API_KEY=sk-...   # or add it to .env
make dataset                     # -> data/query_dataset.parquet + data/query_plan.parquet
```

### NER

Three commands, chained by `make ner`:

```bash
make ner-data    # query dataset -> data/ner/{train,val}.json
make ner-train   # fine-tune -> data/gliner-geosearch/ (GPU strongly advised)
make ner-eval    # tuned vs baseline + threshold sweep -> data/ner_metrics.json
```

**`ner-data`** (`src/ner/dataset.py`) exports the synthetic query dataset as the two
JSON files GLiNER's trainer reads, split **by place name** so a val query is never a
paraphrase of a train query about the same city. A row whose gold span does not land on
token boundaries is dropped whole rather than partially labelled — a query that mentions
a city with that city left unlabelled teaches the opposite of the intended lesson.

**`ner-train`** (`src/ner/train.py`) fine-tunes `urchade/gliner_multi-v2.1` and writes
the result to `NER__MODEL_DIR`, which defaults to the same path `GLINER_MODEL` serves
from — training publishes directly to what the engine loads. Runs on CPU, but expect
hours instead of minutes without a GPU.

The model is selected **by val span F1**, not by `eval_loss`. `eval_loss` is the
training objective on the holdout — a reasonable checkpoint signal, but not the quantity
anyone cares about. `BestSpanF1Callback` scores the val split after every epoch the way
the engine will call the model and saves whenever that improves. That also avoids a
failure that is silent by construction: `load_best_model_at_end=True` makes
`transformers.Trainer` reload the best checkpoint into the outer `GLiNER` wrapper, whose
parameter names carry a `model.` prefix, while `GLiNER.save_pretrained` writes the
*inner* module's keys unprefixed. Every key mismatches, `strict=False` swallows it, and
the model left in memory is the **last** epoch. Saving the live model at the moment it
is measured leaves no reload to fail.

**`ner-eval`** (`src/ner/evaluate.py`) is the gate. Micro span P/R/F1 on exact
`(start, end, label)` matches — downstream the reranker matches span text against
candidate names, so a span off by a word is not partial credit there either — bucketed
overall, per label and per language, with the tuned model compared against the zero-shot
baseline measured using the *same* splitter (otherwise the zh delta would be an artefact
of segmentation rather than of fine-tuning). Plus the threshold sweep. The full report,
including the per-query error list, is written to `data/ner_metrics.json`.

Averaging is micro over spans, not macro over queries: a query naming three cities feeds
three names into retrieval and should weigh three times as much. Every rate is printed
next to its support, because the val split is ~47 queries — a bucket like `label:STATE`
rests on 6 spans and swings by tenths of F1 on one of them. Read the numbers as
directional.

Two things about serving the result, both silent failure modes if got wrong:

* **The word splitter is not part of the checkpoint.** GLiNER classifies *token*
  spans, and its default splitter is whitespace-based with unicode `\w`, so a run
  of Han characters is one token: in `莫斯科新闻` ("Moscow news") the city `莫斯科`
  is unpredictable *in principle* — measured on the dataset, that is **all 56 zh
  spans** versus 0 for ru/en/tr. `src/ner/tokenizer.py::CjkAwareSplitter` splits
  each ideograph into its own token (alignment failures 0/410), the export built
  `train.json`'s token indices with it, and the fine-tune saw that tokenization —
  but `gliner_config.json` can only name GLiNER's *built-in* splitters, so nothing
  in the checkpoint records the choice. `SearchEngine.build` sets it from
  `NER_CJK_SPLITTER` (default `true`) and logs which splitter is live. Flip it to
  `false` in the same change that points `GLINER_MODEL` back at a zero-shot
  checkpoint, which never saw per-ideograph tokens.

  Because nothing downstream notices a mismatch, `make ner-train` also writes
  **`ner_meta.json`** next to the weights recording which splitter the checkpoint
  requires, and `SearchEngine._check_ner_meta` raises at startup when the configured
  flag disagrees with it. A checkpoint without that file is left alone — absence is
  not evidence either way.
* **`NER_THRESHOLD` is a parameter of serving, not of the model.** Retrieval is
  recall-hungry — a city span never extracted is a city that can never be
  retrieved, while a spurious span only adds a candidate the reranker can demote —
  so the served default is `0.4`, below GLiNER's own 0.5. `make ner-eval` runs the
  sweep that derives it; re-run it whenever the NER model changes.

Changing the NER model changes the reranker's training distribution: its text
features are mined from *live* NER output (see below), so re-run
`make rerank-data && make rerank-train && make rerank-eval` after swapping the
checkpoint.

### Reranker

`src/rerank/` turns the synthetic query dataset into a trained CatBoost ranker
that reorders BM25's candidates. It's a two-stage pipeline, gated by a golden-set
comparison:

1. **`dataset.py`** (`make rerank-data`) replays every query in
   `data/query_dataset.parquet` through the pipeline with `use_rerank=false`, so
   the training data never depends on the reranker it's about to train. In
   process by default (`src/search/batch.py`) — the same `SearchEngine.search`
   the endpoint calls, which makes it impossible to mine against a server left
   running on older code. For each query the gold geonameid(s) are labelled positive and
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

   **Entity source (`RERANK__USE_GOLD_ENTITIES`, default `false`).** By default the
   text features are the live API's `entity_buckets` — the NER spans, which is what
   keeps train/serve parity. Set it to `true` and the mining reads the dataset's
   `gold_*_entities` columns instead, answering "how good would the reranker be if
   NER were perfect?". That breaks parity deliberately, so the resulting model must
   not be served; and it only replaces the *text features* — candidates still come
   from retrieval driven by the NER spans, and queries NER missed entirely retrieve
   nothing and are skipped, so gold entities cannot recover NER's lost recall. The
   run's mode is only visible in the log (a WARNING in gold mode), so override the
   output paths to keep both datasets:

   ```bash
   RERANK__USE_GOLD_ENTITIES=true \
   RERANK__TRAIN_PATH=data/rerank_train_gold.parquet \
   RERANK__TEST_PATH=data/rerank_test_gold.parquet \
     make rerank-data
   ```
2. **`train.py`** (`make rerank-train`) fits a CatBoost `YetiRank` listwise ranker
   over those features, grouped per query, with early stopping on
   NDCG@`RERANK__NDCG_TOP` over the held-out queries. The trained model is saved
   to `data/rerank_model.cbm`.

`src/rerank/model.py::Reranker` is the inference-side wrapper `SearchEngine`
loads at startup; it featurises candidates with the same `build_row`.

```bash
make rerank        # rerank-data -> rerank-train
make rerank-eval   # golden-set: rerank vs baseline vs ideal
```

Both build the engine in process — no server to start. Set
`RERANK__SEARCH_URL=http://localhost:8000/v1/search` to replay against a running
instance instead, which is the one thing in-process cannot do: measure an actual
deployment, routes and response schema included. The two paths return identical
responses (verified over the golden set, reranker on and off).

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

That is the whole list. No database, no Docker, no external service — Docker is
optional and only packages the API as an image.

---

## Quick start

### Just run the search (no database, no ETL)

Everything the API needs is published on the HuggingFace Hub.

```bash
uv sync --dev && source .venv/bin/activate
cp .env.example .env

make hub-pull                              # ~60 MB of artifacts + the reranker
uv run uvicorn src.api.main:app --reload
```

Open http://localhost:8000/docs. The NER weights are the one thing not fetched
by `hub-pull` — point `GLINER_MODEL` at the Hub id and `GLiNER.from_pretrained`
caches them itself, or keep a local `data/gliner-geosearch/`.

### The full authoring pipeline (rebuild everything from source data)

```bash
uv sync --dev && source .venv/bin/activate
cp .env.example .env

make etl                # download the dumps + parse into data/corpus/ (~1 GB download)
make artifacts          # corpus -> the files serving reads
make dataset            # synthetic queries (needs DEEPSEEK_API_KEY)
make ner                # fine-tune GLiNER (GPU strongly advised)

uv run uvicorn src.api.main:app --reload   # serves from data/artifacts/
```

`data/corpus/` is intermediate and git-ignored: reproducible from the raw dumps
in ~45 s, and what gets published is the compiled artifact built from it.

`make artifacts` is the seam: re-run it after any change to the corpus
(`make etl`, `make update`, a new country or language), or serving keeps
answering from the previous build.

### Run it in Docker

```bash
docker build -t geosearch .
docker run -p 8000:8000 -v "$PWD/data:/app/data" geosearch
```

One container, nothing else — it reads `./data/artifacts`, so run
`make artifacts` or `make hub-pull` first.

---

## Artifacts and the HuggingFace Hub

`data/` is **not** in git. The model weights are ~1.1 GB, the raw GeoNames dumps
309 MB, and every re-train would add a full copy to history permanently. The
artifacts live on the Hub instead; git keeps only the small text records of what
was built (`ner_meta.json`, the metrics JSONs).

| Repo | Contents |
|---|---|
| [`mki0809/geosearch-index`](https://huggingface.co/datasets/mki0809/geosearch-index) | serving artifacts: BM25 index, places table, candidate descriptions, manifest |
| [`mki0809/geosearch-reranker`](https://huggingface.co/mki0809/geosearch-reranker) | the CatBoost model + its metrics |
| [`mki0809/geosearch-ner`](https://huggingface.co/mki0809/geosearch-ner) | the fine-tuned GLiNER checkpoint |
| [`mki0809/geosearch-queries`](https://huggingface.co/datasets/mki0809/geosearch-queries) | synthetic queries, rerank train/test, golden set, NER export |

Five separate repos because they version independently: re-training NER forces a
reranker re-train (it changes the spans the reranker is fed) but not the reverse,
and the index is rebuilt on every GeoNames refresh while the query dataset is not.

```bash
make hub-pull            # serving artifacts + reranker — enough to run the API
make hub-pull-all        # ...plus the training/eval corpora (needed for notebooks)

make hub-push-index      # publish (each needs a *write* HF_TOKEN in .env)
make hub-push-ner
make hub-push-rerank
make hub-push-data
make hub-push-space
```

**Pinning.** Every pull logs the commit it resolved and every push prints the
commit to pin. A number reported in a thesis should name the artifact it was
measured on, so set `HF__INDEX_REVISION=<40-char sha>` (and the `NER_`/`RERANKER_`
equivalents) to freeze a run; the startup log then records exactly what was served.

**The four serving files are one set.** `manifest.json` carries a `build_id`
stamped into each Parquet, plus the countries/languages the corpus was built
under, and both are checked at startup. Every failure here is otherwise silent: a
places table missing a geonameid the index can retrieve does not raise, it drops
that place from results it should have won; and an index built for two countries
served under a config naming four simply cannot return the other two.

The index is a pickle-free `.npz` loaded with `allow_pickle=False` — the previous
format was a pickled object graph, i.e. arbitrary code execution on a file fetched
from a public repo.

---

---

## Runbook: reproducing the metrics end to end

The order matters, and each step says what it needs and what it leaves behind.
Steps 1-3 are enough to run the notebooks; 4-6 rebuild the models themselves.

### 0. What state are you in?

```bash
ls data/artifacts/          # bm25_index.npz places.parquet descriptions.parquet manifest.json
ls data/gliner-geosearch/   # the NER weights (~1.1 GB, git-ignored)
ls data/rerank_model.cbm data/golden_set.parquet data/query_dataset.parquet
cat data/artifacts/manifest.json | head -20    # which build is on disk
```

Missing artifacts → `make hub-pull`. Missing datasets → `make hub-pull-all`.
Missing NER weights → set `GLINER_MODEL` to the Hub id, or re-run `make ner-train`.

### 1. Get the artifacts

Either fetch them:

```bash
make hub-pull-all       # artifacts + reranker + the query/golden datasets
```

Or rebuild from source data (~1 GB download, then ~90 s of parsing and building):

```bash
make etl
make artifacts
```

`make artifacts` prints a `build_id` and the row counts. It refuses to finish if
any geonameid the index can retrieve is missing from `places.parquet`, so a
successful run means the set is internally consistent.

### 2. Start the API (only for the notebooks)

`rerank-data` and `rerank-eval` no longer need it — they build the engine in
process. The four notebooks still call `localhost:8000`, so start it for those.
It needs no database.

```bash
uv run uvicorn src.api.main:app --reload
```

Check the startup log before trusting anything it serves:

```
building search engine (mode: artifacts)
Artifacts loaded: 1187541 names, 1300787 places, build 926cd38e-...
NER model loaded — CJK-aware splitter, threshold 0.40
Reranker loaded: 1300787 descriptions
```

Three things to read there. **`mode:`** — `artifacts` means no database is
involved. **The splitter** — `CJK-aware` is required by the fine-tuned
checkpoint; `GLiNER whitespace` with that checkpoint silently loses every
Chinese span. **`Reranker loaded`** — if it says `No reranker at ...` or `Stale
reranker`, `use_rerank=true` will quietly return retriever order, and your
"rerank" numbers will equal your baseline.

Smoke-test one query per language:

```bash
curl -sG http://localhost:8000/v1/search --data-urlencode 'text=Хочу в Москву' --data-urlencode 'top_k=3'
curl -sG http://localhost:8000/v1/search --data-urlencode 'text=莫斯科新闻'     --data-urlencode 'top_k=3'
```

`莫斯科新闻` must return Moscow. If it returns nothing, the splitter is wrong.

### 3. Run the notebooks

All four read `../data/golden_set.parquet` and call the API on
`localhost:8000`, so with step 2 running they need nothing else:

```bash
uv run jupyter lab notebooks/
```

| Notebook | What it answers |
|---|---|
| `pipeline_metrics.ipynb` | the full pipeline, reranker on |
| `pipeline_metrics_no_rerank.ipynb` | retrieval alone — the baseline to beat |
| `pipeline_metrics_ideal_rerank.ipynb` | the oracle ceiling: how much is left on the table |
| `pipeline_latency.ipynb` | the full pipeline, reranker on timing calculation |

Read them against each other, not in isolation. The reranker only reorders the
retrieved top-*k*, so **`Recall@50` is identical in all of them by construction**
— if it moves, something is wrong. The reranker's effect lives in `P@1` and `RR`,
and the gap between those and the ideal notebook is the headroom left.

The same comparison non-interactively, which also writes the file the model card
quotes — and needs no server at all:

```bash
make rerank-eval        # -> data/rerank_metrics.json
```

### 4. Re-derive the NER numbers

```bash
make ner-eval           # -> data/ner_metrics.json
```

Tuned vs zero-shot baseline, bucketed overall / per label / per language, plus
the threshold sweep. Read every rate next to its support: the val split is ~47
queries, so `label:STATE` rests on ~6 spans.

The sweep is what justifies `NER_THRESHOLD`. If its best threshold is not the
one in `.env`, change `.env` — the number is supposed to be derived, not asserted.

### 5. Rebuild the models (only if you changed something)

Retraining NER changes the spans the reranker is fed, so it invalidates the
reranker. The order is not optional:

```bash
make ner                # ner-data -> ner-train -> ner-eval   (GPU strongly advised)
make rerank             # rerank-data -> rerank-train
make rerank-eval
```

Check `RERANK__USE_GOLD_ENTITIES` is `false` before training anything you intend
to serve. `true` is an ablation that trains on the dataset's gold spans while
serving gets GLiNER's — a model trained that way is not servable, and nothing in
the `.cbm` records which it is.

### 6. Publish

```bash
make hub-push-index     # after `make artifacts`
make hub-push-ner       # after `make ner`  — needs ner_meta.json to exist
make hub-push-rerank    # after `make rerank-eval`, so the card has numbers
make hub-push-data
```

Each prints the commit sha. Put those in the thesis, and set the matching
`HF__*_REVISION` when you want a run reproduced exactly.

### Sanity checks worth keeping

```bash
make test lint typecheck

# a rebuild must reproduce the artifacts exactly — compare row counts and
# frames against the previous build before publishing
make artifacts && cat data/artifacts/manifest.json
```

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

### ETL pipeline

| Target | Description |
|--------|-------------|
| `make download` | Download raw GeoNames files for all configured countries |
| `make load` | Parse the downloaded dumps into the staging corpus (`data/corpus/*.parquet`) |
| `make etl` | Full pipeline: `download` then `load` |
| `make update` | Download and apply today's GeoNames daily delta files |

### Serving artifacts

| Target | Description |
|--------|-------------|
| `make artifacts` | Compile the staging corpus into the files serving reads: BM25 index, places table, candidate descriptions, manifest. Re-run after any corpus change |

### HuggingFace Hub

| Target | Description |
|--------|-------------|
| `make hub-pull` | Fetch the serving artifacts + reranker — the first command on a fresh clone |
| `make hub-pull-all` | ...plus the training and evaluation corpora |
| `make hub-push-index` | Publish the serving artifacts (needs a write `HF_TOKEN`) |
| `make hub-push-ner` | Publish the fine-tuned NER model |
| `make hub-push-rerank` | Publish the CatBoost reranker |
| `make hub-push-data` | Publish the query dataset and derived training sets |
| `make hub-push-space` | Deploy the Gradio Space |

### Dataset

| Target | Description |
|--------|-------------|
| `make dataset` | Generate the synthetic query dataset + sample plan (DeepSeek → Parquet); warm-starts from the checkpoint if re-run |
| `make dataset-plan` | Build and print the sample plan only — no API calls, no key |

### NER

| Target | Description |
|--------|-------------|
| `make ner-data` | Export the query dataset as GLiNER train/val JSON (split by place name) |
| `make ner-train` | Fine-tune GLiNER, saving the best-val-span-F1 epoch to `NER__MODEL_DIR` |
| `make ner-eval` | Span P/R/F1 vs the zero-shot baseline + threshold sweep → `data/ner_metrics.json` |
| `make ner` | Full pipeline: `ner-data` → `ner-train` → `ner-eval` |

### Reranker

| Target | Description |
|--------|-------------|
| `make rerank-data` | Mine labelled `(query, document, label)` pairs by replaying the query dataset through the pipeline (in process; `RERANK__SEARCH_URL` replays against a live server instead) |
| `make rerank-train` | Fit the CatBoost `YetiRank` reranker on the mined pairs → `data/rerank_model.cbm` |
| `make rerank-eval` | Golden-set metrics: rerank vs retriever baseline vs ideal ceiling → `data/rerank_metrics.json` |
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
├── Dockerfile                  # API image (uv + uvicorn); optional, nothing depends on it
├── space/                      # the Gradio demo pushed to HuggingFace Spaces
│   ├── app.py                  #   runs a copy of src/, so it cannot diverge
│   └── requirements.txt        #   a deliberate subset: no sqlalchemy, no fastapi
├── Makefile                    # developer workflow
├── pyproject.toml              # dependencies (uv)
└── src/
    ├── config.py               # settings: countries, languages, models, paths
    ├── corpus.py               # the staging corpus: two Parquet tables + delta ops
    ├── etl/
    │   ├── downloader.py       # download raw files from GeoNames
    │   ├── parser.py           # parse TSV files into Pydantic models
    │   ├── loader.py           # parse the dumps into data/corpus/
    │   └── updater.py          # incremental updates from daily delta files
    ├── search/
    │   ├── engine.py           # SearchEngine: GLiNER NER + retrieval + rerank
    │   ├── artifacts.py        # build/load/validate the DB-free serving set
    │   ├── places.py           # PlaceStore protocol + the Parquet-backed store
    │   ├── corpus_queries.py   # the three projections that used to be SQL
    │   ├── bm25.py             # BM25 char-n-gram index, pickle-free .npz format
    │   └── tokenizer.py        # character n-gram tokenizer
    ├── hub/
    │   ├── client.py           # the only module that calls huggingface_hub
    │   ├── publish.py          # push each artifact to its own repo, with a card
    │   ├── pull.py             # fetch what a fresh clone needs
    │   ├── push.py             # `python -m src.hub.push --what ...`
    │   ├── cards.py            # render READMEs from the metrics files
    │   └── arrow.py            # normalise large_* Arrow types for the Viewer
    ├── dataset/
    │   ├── sampling.py         # Polars sample plan: names, homonyms, stratified pick
    │   ├── prompts.py          # one system prompt + entity contract per query kind
    │   ├── schemas.py          # Pydantic models for the LLM's JSON response
    │   └── generate.py         # DeepSeek generation: concurrency + warm-start checkpoint
    ├── ner/
    │   ├── tokenizer.py        # CjkAwareSplitter + span→token alignment
    │   ├── dataset.py          # query dataset -> GLiNER train/val JSON (split by name)
    │   ├── train.py            # fine-tune GLiNER, select the best epoch by val span F1
    │   └── evaluate.py         # span P/R/F1 vs baseline + decision-threshold sweep
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
tests/                          # no DB, network, GPU or trained model required
├── etl/
│   └── test_parser.py          # TSV parsing, from in-memory zip fixtures
├── dataset/
│   ├── test_sampling.py        # the sample plan
│   └── test_generate.py        # response validation + warm-start checkpointing
├── ner/
│   ├── test_tokenizer.py       # CJK splitting + span alignment
│   ├── test_dataset.py         # export drops unusable rows; split leaks no name
│   └── test_evaluate.py        # span metrics, scored from supplied span sets
├── rerank/
│   ├── test_dataset.py         # split() / build_pairs()
│   └── test_features.py        # the shared match features
├── search/
│   ├── test_bm25_serialization.py  # scores bit-identical to rank_bm25
│   ├── test_places.py          # the place store's contract
│   └── test_artifacts.py       # manifest, build_id and coverage checks
├── hub/
│   └── test_arrow.py           # nested large-type normalisation
```

---

## Configuration

All settings live in `src/config.py` and are overridable via `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GEONAMES_DATA_DIR` | `data/raw` | Where `make download` puts the raw dumps and `make etl` reads them |
| `CORPUS_DIR` | `data/corpus` | The staging corpus `make etl` writes and `make artifacts` reads — intermediate, git-ignored |
| `COUNTRIES` | `["RU","US","TR","CN"]` | ISO-3166 country codes to load |
| `LANGUAGES` | `["ru","en","tr","zh"]` | Alternate-name language codes to keep |
| `GLINER_MODEL` | `data/gliner-geosearch` | GLiNER model for NER — a local directory (the fine-tuned checkpoint) or a HuggingFace id (e.g. `urchade/gliner_multi-v2.1`) |
| `NER_LABELS` | `["CITY","REGION","STATE","COUNTRY"]` | Entity labels GLiNER extracts |
| `NER_CJK_SPLITTER` | `true` | Serve with `CjkAwareSplitter` (per-ideograph zh tokens). Required by the fine-tuned checkpoint, must be `false` for a zero-shot one |
| `NER_THRESHOLD` | `0.4` | GLiNER decision threshold — a serving parameter, set below GLiNER's own 0.5 default: lower trades precision for the recall retrieval depends on |
| `ARTIFACTS_DIR` | `data/artifacts` | What serving reads: BM25 index, places table, candidate descriptions, manifest |
| `EXCLUDED_FEATURE_CODES` | `["PPLH","PPLQ","PPLW","PPLX"]` | GeoNames feature codes excluded from the corpus |
| `DEEPSEEK_API_KEY` | _(none)_ | DeepSeek API key for `make dataset` (required only for that target) |
| `DATASET__DEEPSEEK_MODEL` | `deepseek-chat` | DeepSeek model used to generate queries |
| `DATASET__MAX_WORKERS` | `8` | Concurrent in-flight requests during dataset generation |
| `DATASET__MAX_GENERATION_ATTEMPTS` | `3` | In-process retries when a response fails to parse or fails the entity checks |
| `DATASET__MAX_TOKENS` | `1024` | Response cap; too low truncates the json mid-object and costs the whole row |
| `DATASET__OUTPUT_PATH` | `data/query_dataset.parquet` | Where the generated dataset is written |
| `DATASET__PLAN_PATH` | `data/query_plan.parquet` | Where the sample plan is written |
| `NER__MODEL_DIR` | `data/gliner-geosearch` | Where `make ner-train` writes the tuned model. Keep equal to `GLINER_MODEL`, or training publishes somewhere the engine never reads |
| `NER__BASE_MODEL` | `urchade/gliner_multi-v2.1` | Checkpoint to fine-tune from, and the baseline `make ner-eval --baseline` compares against |
| `NER__VAL_SIZE` | `0.2` | Fraction of each language's rows held out. Split by place *name*, so the realised fraction lands near this, not on it |
| `NER__EPOCHS` | `15` | Fine-tuning epochs (`--epochs` overrides per run) |
| `NER__BATCH_SIZE` | `8` | Per-device batch size |
| `NER__LEARNING_RATE` | `5e-6` | Rate for the pretrained encoder |
| `NER__OTHERS_LR` | `1e-5` | Rate for the span head + label embeddings — the parts that must move to fit a new label set |
| `NER__METRICS_PATH` | `data/ner_metrics.json` | Where `make ner-eval` writes the full report |
| `NER__THRESHOLD_SWEEP` | `[0.1 … 0.7]` | Decision thresholds `make ner-eval --sweep` reports, so `NER_THRESHOLD` is derived rather than asserted |
| `RERANK__SEARCH_URL` | _(empty)_ | Empty replays queries **in process**; set a URL to replay against a running server instead |
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

The whole suite runs without a network, a GPU or a trained model — ETL tests use
in-memory zip fixtures, and the NER tests score pre-supplied span sets rather
than loading GLiNER. There is nothing to stub out for a database, because there
is no database.
