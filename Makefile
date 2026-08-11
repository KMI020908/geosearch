.PHONY: download load etl update artifacts hub-pull hub-pull-all hub-push-index hub-push-ner hub-push-rerank hub-push-data hub-push-space dataset dataset-plan ner-data ner-train ner-eval ner rerank-data rerank-train rerank-eval rerank test lint typecheck clean help

PYTHON := .venv/bin/python

help:
	@echo "GeoSearch — available targets:"
	@echo ""
	@echo "  ETL pipeline"
	@echo "    download    Download raw GeoNames files (country zips + alternateNames)"
	@echo "    load        Parse them into the staging corpus (data/corpus/*.parquet)"
	@echo "    etl         Full pipeline: download → load"
	@echo "    update      Apply today's GeoNames daily delta"
	@echo ""
	@echo "  Serving artifacts"
	@echo "    artifacts   Compile the corpus into what serving reads (index + places + descriptions)"
	@echo ""
	@echo "  HuggingFace Hub"
	@echo "    hub-pull        Fetch serving artifacts + reranker (start here on a fresh clone)"
	@echo "    hub-pull-all    ...plus the training/eval datasets"
	@echo "    hub-push-index  Publish the serving artifacts   (needs a write HF_TOKEN)"
	@echo "    hub-push-ner    Publish the fine-tuned NER model"
	@echo "    hub-push-rerank Publish the fine-tuned cross-encoder reranker"
	@echo "    hub-push-data   Publish the query dataset + derived training sets"
	@echo "    hub-push-space  Deploy the Gradio Space"
	@echo ""
	@echo "  Dataset"
	@echo "    dataset      Generate synthetic query dataset + name pool (DeepSeek -> Parquet)"
	@echo "    dataset-plan Build + inspect the sample plan only (no API calls, no key)"
	@echo ""
	@echo "  NER"
	@echo "    ner-data     Export GLiNER fine-tuning JSON (train/val) from the query dataset"
	@echo "    ner-train    Fine-tune GLiNER, saving the best-span-F1 epoch (GPU strongly advised)"
	@echo "    ner-eval     Span P/R/F1 vs the zero-shot baseline + threshold sweep -> JSON"
	@echo "    ner          Full NER pipeline: ner-data -> ner-train -> ner-eval"
	@echo ""
	@echo "  Reranker"
	@echo "    rerank-data  Build labelled reranker pairs (runs the pipeline in process)"
	@echo "    rerank-train Fine-tune the cross-encoder reranker on the pairs"
	@echo "    rerank-eval  Golden-set rerank-vs-baseline-vs-ideal metrics -> JSON"
	@echo "    rerank       Full reranker pipeline: rerank-data -> rerank-train"
	@echo ""
	@echo "  Development"
	@echo "    test        Run test suite"
	@echo "    lint        Run ruff linter + formatter check"
	@echo "    typecheck   Run pyright type checker"
	@echo "    clean       Remove downloaded raw data files"

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
# Serving artifacts
# ---------------------------------------------------------------------------

# Compile the staging corpus into the files serving reads: BM25 index, places
# table, candidate descriptions, manifest. Needs `make etl` to have run.
# Re-run after any corpus change — serving keeps answering from the last build.
artifacts:
	$(PYTHON) -m src.search.artifacts

# ---------------------------------------------------------------------------
# HuggingFace Hub
# ---------------------------------------------------------------------------

# First command after a fresh clone: data/ is no longer in git. Pulls the
# serving artifacts + reranker; add --what all for the training corpora too.
# Needs no token for public repos.
hub-pull:
	$(PYTHON) -m src.hub.pull --what serving

hub-pull-all:
	$(PYTHON) -m src.hub.pull --what all

# Publishing. Each needs a *write* HF_TOKEN in .env. Separate targets because
# the artifacts are rebuilt on different schedules — pushing an unchanged 1.1GB
# checkpoint next to a 12KB Parquet is not something to do by accident.
hub-push-index:
	$(PYTHON) -m src.hub.push --what index

hub-push-ner:
	$(PYTHON) -m src.hub.push --what ner

hub-push-rerank:
	$(PYTHON) -m src.hub.push --what rerank

hub-push-data:
	$(PYTHON) -m src.hub.push --what data

hub-push-space:
	$(PYTHON) -m src.hub.push --what space

# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

dataset:
	$(PYTHON) -m src.dataset.generate

# Build and inspect the sample plan without spending a single API token. Needs
# the staging corpus (`make etl`), but no DEEPSEEK_API_KEY.
dataset-plan:
	$(PYTHON) -m src.dataset.generate --plan-only

# ---------------------------------------------------------------------------
# NER
# ---------------------------------------------------------------------------

# Export the query dataset as GLiNER train/val JSON.
ner-data:
	$(PYTHON) -m src.ner.dataset

# Fine-tune GLiNER on that export and publish the best epoch to NER__MODEL_DIR
# (the same path the engine serves from). Runs on CPU, but expect hours rather
# than minutes without a GPU.
ner-train:
	$(PYTHON) -m src.ner.train

# The NER gate: tuned vs zero-shot baseline, per label and per language, plus the
# decision-threshold sweep that justifies NER_THRESHOLD. Writes NER__METRICS_PATH.
ner-eval:
	$(PYTHON) -m src.ner.evaluate --baseline --sweep

ner: ner-data ner-train ner-eval

# ---------------------------------------------------------------------------
# Reranker
# ---------------------------------------------------------------------------

rerank-data:
	$(PYTHON) -m src.rerank.dataset

rerank-train:
	$(PYTHON) -m src.rerank.train

rerank-eval:
	$(PYTHON) -m src.rerank.evaluate

rerank: rerank-data rerank-train

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
