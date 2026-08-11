from typing import Self

from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# The canonical ``"<sample_source>:<pool>"`` buckets of the sample plan — the key
# set of `DatasetConfig.quotas`. It lives here rather than in
# `src/dataset/sampling.py` because that module imports this one; sampling asserts
# its own `KINDS` table matches this tuple, so the two cannot drift.
#
# `pool` says where a bucket's names come from: "homonym" for names that share
# their spelling with another place (the hard negatives the reranker has to learn
# to resolve), "unique" for names that do not, "all" for the kinds that do not
# split. Both pools exist for every disambiguation kind so the "region/country
# named -> prefer the matching candidate" pattern is not only ever seen on
# homonyms.
PLAN_KINDS: tuple[str, ...] = (
    "one_city:all",
    "multi_city:all",
    "city_admin1:homonym",
    "city_admin1:unique",
    "city_country:homonym",
    "city_country:unique",
    "city_admin1_country:homonym",
    "city_admin1_country:unique",
)


class GroupQuota(BaseModel):
    """How many *names* to draw from one group's pool: head, middle, tail.

    Sorting a group's pool by population descending and taking the head, a random
    slice of the middle and the tail is what keeps capitals and ``population=0``
    villages both represented instead of drowning in the long tail.

    Counts names, not rows: a homonym name fans out into up to
    ``DatasetConfig.n_targets_per_name`` concrete rows after the pick, so a
    homonym bucket lands in ``[total, total * n_targets_per_name]`` rows per
    group. A unique name has exactly one target, so those buckets are exact.
    """

    n_top: int = Field(default=0, ge=0)
    n_mid: int = Field(default=0, ge=0)
    n_low: int = Field(default=0, ge=0)

    @property
    def total(self) -> int:
        """Names drawn per group, before the target fan-out."""
        return self.n_top + self.n_mid + self.n_low


class DatasetConfig(BaseModel):
    """Tunables for the synthetic query-dataset generator.

    The dataset (city queries + the candidate name pool) feeds several pipeline
    steps, not just the reranker. Everything the generation script needs lives
    here so nothing is hardcoded in the script itself. Override individual
    fields via env with the ``DATASET__`` prefix (e.g.
    ``DATASET__DEEPSEEK_MODEL=deepseek-chat``).
    """

    seed: int = 42

    # Per-group row-count control, uniform across every query kind: each bucket of
    # `PLAN_KINDS` draws `n_top` + `n_mid` + `n_low` names from *its own group's*
    # pool, so no kind can starve a (language, country) the way a single global
    # budget does.
    #
    # Which group a quota applies to depends on the kind, and that asymmetry is
    # deliberate. `one_city`, `multi_city`, `city_admin1` and every `:unique`
    # bucket are grouped by (language, country). `city_country:homonym` and
    # `city_admin1_country:homonym` are grouped by **language alone**: their names
    # are homonyms *across* countries, so iterating countries would draw the same
    # name once per country it lives in and emit duplicate rows.
    #
    # Those two language-keyed buckets get a quota of 2 rather than 1 because they
    # have 4 groups against the `:unique` buckets' 16 — at an equal quota the
    # homonyms, which are the whole point of the disambiguation kinds, would end
    # up the minority of the disambiguation block.
    quotas: dict[str, GroupQuota] = Field(
        default_factory=lambda: {
            "one_city:all": GroupQuota(n_top=2, n_mid=2, n_low=2),
            "multi_city:all": GroupQuota(n_top=1, n_low=1),
            "city_admin1:homonym": GroupQuota(n_top=1),
            "city_admin1:unique": GroupQuota(n_top=1),
            "city_country:homonym": GroupQuota(n_top=1, n_low=1),
            "city_country:unique": GroupQuota(n_top=1),
            "city_admin1_country:homonym": GroupQuota(n_top=1, n_low=1),
            "city_admin1_country:unique": GroupQuota(n_top=1),
        }
    )

    # Caps how many concrete region/country targets one picked name expands into
    # (population-ranked), which is what bounds a homonym bucket's row count at
    # `quota.total * n_targets_per_name`. A country-level homonym named "Moscow"
    # spending both its slots on "Moscow, Russia" and "Moscow, United States" is
    # the intended shape: same query text, different gold.
    n_targets_per_name: int = 2

    # Multi-city queries: one query naming several cities ("flights Moscow Kazan")
    # so the reranker sees 2-3 CITY spans in one query. Each name drawn by the
    # `multi_city:all` quota is an *anchor*, joined with this many additional
    # cities from the same group; gold = the union of their geonameids (same
    # simple labelling as one_city, no disambiguation). Expressed as "extra
    # cities" rather than "total cities" so the row count equals the quota exactly
    # and a group too small to pair is skipped instead of raising.
    multi_city_extra_min: int = 1
    multi_city_extra_max: int = 2

    # Query-style mix (skewed toward terse browser-search input).
    style_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "search": 0.60,
            "casual": 0.25,
            "formal": 0.12,
            "rich": 0.03,
        }
    )

    # Topics the LLM anchors each query to, so queries aren't bare "<city>".
    topics: list[str] = Field(
        default_factory=lambda: [
            "flights",
            "hotels",
            "weather",
            "transport",
            "food",
            "tourism",
            "history",
            "population",
            "relocation",
            "news",
            "education",
        ]
    )

    # DeepSeek API (key comes from Settings.deepseek_api_key, not here).
    deepseek_model: str = "deepseek-v4-flash"
    deepseek_base_url: str = "https://api.deepseek.com"
    temperature: float = 1.3
    # `deepseek-v4-flash` reasons on every call and offers no way to switch that
    # off — the accepted values are low/medium/high/max/xhigh, with no "none".
    # Leaving the field unset is not the same as sending "low": measured over the
    # same 5 `one_city` rows, the model's own default spent a median 158 reasoning
    # tokens (3.1s) against "low"'s 372 (4.9s), so the explicit floor was costing
    # roughly double for nothing. `None` means "omit the parameter".
    reasoning_effort: str | None = None
    max_retries: int = 5
    # Split per phase rather than one float: a bare `timeout=` applies the same
    # budget to connect and read alike, and those want very different numbers here.
    # Connecting is ~0.5s when the API is healthy, while a single reasoning
    # response has been observed taking 35s under load, so the read budget carries
    # all the headroom.
    connect_timeout: float = 10.0
    read_timeout: float = 180.0
    # Concurrent in-flight requests. DeepSeek is rate-limited server-side, so
    # the SDK's 429 backoff (max_retries) does the throttling for us.
    max_workers: int = 8
    # Response cap — and it covers the model's *reasoning* tokens, not just the
    # answer, which is what makes it easy to under-size on a reasoning model. Set
    # it too low and the whole budget goes to the hidden chain of thought:
    # `max_tokens=100` returns `finish_reason="length"`, 100 reasoning tokens and
    # an **empty** content string, so the row fails as unparsable rather than as a
    # truncated query. Even a plain `one_city` row has been seen spending 776
    # reasoning tokens, so the answer's own ~60 tokens are the small part of this.
    max_tokens: int = 4096
    # In-process retries when a response fails to parse or fails the
    # deterministic entity checks (`validate_generation`). Distinct from
    # `max_retries`, which is the SDK's transport/429 budget.
    max_generation_attempts: int = 3

    output_path: str = "data/query_dataset.parquet"
    # Append-only JSONL of completed rows; lets a crashed/interrupted run
    # warm-start and skip everything already generated.
    checkpoint_path: str = "data/query_dataset.checkpoint.jsonl"
    # The sample plan (one row per intended query, before generation), saved so
    # downstream steps can reuse the same sampled set of names.
    plan_path: str = "data/query_plan.parquet"

    @model_validator(mode="after")
    def _quota_keys_are_known(self) -> Self:
        """Require `quotas` to cover exactly `PLAN_KINDS`.

        A `DATASET__QUOTAS` env override *replaces* the whole dict —
        pydantic-settings does not deep-merge nested models — so an override has
        to restate every bucket. Checking both directions turns a typo into a
        startup error instead of a silently zeroed quota that drops a whole query
        kind from the dataset.
        """
        unknown = sorted(set(self.quotas) - set(PLAN_KINDS))
        missing = sorted(set(PLAN_KINDS) - set(self.quotas))
        if unknown or missing:
            raise ValueError(
                f"dataset.quotas must cover exactly PLAN_KINDS "
                f"(unknown keys: {unknown}, missing keys: {missing})"
            )
        return self


class NerConfig(BaseModel):
    """Tunables for the GLiNER data export, fine-tune and evaluation.

    Covers all three NER commands: ``make ner-data`` exports the synthetic query
    dataset as GLiNER training JSON (:mod:`src.ner.dataset`), ``make ner-train``
    fine-tunes on it (:mod:`src.ner.train`) and ``make ner-eval`` measures the
    result (:mod:`src.ner.evaluate`). Override via env with the ``NER__`` prefix.
    """

    seed: int = 42

    # Fraction of each language's rows held out for validation. The split is by
    # place *name*, so the realised fraction lands near this, not exactly on it.
    val_size: float = 0.2

    query_dataset_path: str = "data/query_dataset.parquet"
    train_path: str = "data/ner/train.json"
    val_path: str = "data/ner/val.json"

    # --- Fine-tuning ---------------------------------------------------------

    # Zero-shot checkpoint to fine-tune from, and the baseline `make ner-eval`
    # compares the tuned model against.
    base_model: str = "urchade/gliner_multi-v2.1"

    # Where training writes the tuned model — the same path `Settings.gliner_model`
    # serves from, so `make ner-train` publishes straight to what the engine loads.
    model_dir: str = "data/gliner-geosearch"
    # HF Trainer's own bookkeeping directory (logs, trainer state). The model is
    # NOT selected from here — see `src.ner.train.BestSpanF1Callback`.
    checkpoint_dir: str = "data/ner/checkpoints"
    metrics_path: str = "data/ner_metrics.json"

    epochs: int = 15
    batch_size: int = 8
    # GLiNER's own fine-tuning recipe splits the rate in two: a small one for the
    # pretrained transformer encoder, a larger one for the span head and label
    # embeddings — the parts that actually have to move to fit a new label set.
    learning_rate: float = 5e-6
    others_lr: float = 1e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1

    # --- Publishing ----------------------------------------------------------

    # Push the trained model to the HuggingFace Hub after training. Weights are
    # ~1.1GB in a single file, which git cannot hold: GitHub hard-rejects blobs
    # over 100MB, and even under that limit every re-train would add a full copy
    # to history permanently. The Hub is where the artifact lives; git keeps only
    # `ner_meta.json` and `ner_metrics.json`, the record of what was trained.
    #
    # Off by default: pushing publishes outside this machine, so it is opt-in
    # rather than a side effect of running `make ner-train`. *Where* it goes and
    # whether that repo is private are not training decisions — they live on
    # `HfConfig` below, shared with every other artifact this project publishes.
    push_to_hub: bool = False

    # --- Evaluation ----------------------------------------------------------

    # Decision thresholds `make ner-eval --sweep` reports, so `ner_threshold`
    # (below, on Settings) is chosen from a measurement rather than asserted.
    threshold_sweep: list[float] = Field(
        default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    )


class RerankConfig(BaseModel):
    """Tunables for building and fine-tuning the reranker.

    ``rerank-data`` turns the query dataset + live search API into labelled
    ``(query_text, document, label)`` pairs; ``rerank-train`` fine-tunes a
    sentence-transformers cross-encoder on them. Override via env with the
    ``RERANK__`` prefix.
    """

    seed: int = 42

    # Where to run the replayed queries. Empty (the default) means *in process*:
    # `src/search/batch.py` builds the engine and calls the same `search()` the
    # route calls, so no server has to be started and — more to the point — a
    # server left running on older code cannot silently mine features from a
    # different pipeline than the one that will serve them.
    #
    # Set it to a URL to replay against a running instance instead. That is the
    # one thing in-process cannot do: measure an actual deployment, its routes
    # and its response schema included.
    search_url: str = ""
    top_k: int = 50
    request_timeout: float = 30.0

    # Fraction of distinct queries held out for test (split by query — the
    # ranking group — so no query text leaks across train and test).
    test_size: float = 0.2

    # Where the mined text features come from. False (default) — the NER spans in
    # the API response's ``entity_buckets``, exactly what the engine feeds the
    # reranker in production. True — the query dataset's gold ``gold_*_entities``
    # columns: an ablation measuring the reranker under perfect entity tagging.
    # A model trained with True is NOT servable — online the reranker always gets
    # NER spans, so gold-trained features break train/serve parity by design.
    use_gold_entities: bool = False

    query_dataset_path: str = "data/query_dataset.parquet"
    # Hand-curated (query, gold geonameIds) set for the final rerank-vs-baseline
    # comparison (make rerank-eval). Never used for training or model selection.
    golden_set_path: str = "data/golden_set.parquet"
    train_path: str = "data/rerank_train.parquet"
    test_path: str = "data/rerank_test.parquet"
    # A directory (sentence-transformers `CrossEncoder.save_pretrained()` output —
    # config.json, weights, tokenizer files), the same shape as `NerConfig.model_dir`.
    model_path: str = "data/rerank_model"
    # Candidate documents (`build_descriptions` output) as a table, so serving can
    # load them without the database that produced them. Written by
    # `make artifacts`; the reranker cannot score anything without it.
    descriptions_path: str = "data/descriptions.parquet"
    # `make rerank-eval` results. The reranker's counterpart to
    # `NerConfig.metrics_path`: small, git-tracked, and the only place the
    # published model card's numbers are allowed to come from.
    metrics_path: str = "data/rerank_metrics.json"

    # --- Cross-encoder fine-tuning --------------------------------------------

    # Zero-shot checkpoint to fine-tune from.
    base_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # HF Trainer's own bookkeeping directory (logs, trainer state). The model is
    # NOT selected from here — see `src.rerank.train.BestRankCallback`.
    checkpoint_dir: str = "data/rerank/checkpoints"
    epochs: int = 4
    batch_size: int = 32
    learning_rate: float = 2e-5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    # Max token length for a (query_text, document) pair. Sized for the short
    # entity-span queries and 2-3 line documents this model actually sees.
    max_length: int = 128
    # BinaryCrossEntropyLoss `pos_weight`. top_k=50 retrieval with ~1 gold
    # positive per query means the mined pairs are heavily imbalanced — None
    # (default) computes it from the train split's own neg/pos row ratio so
    # the loss does not collapse toward predicting everything negative.
    pos_weight: float | None = None

    # --- Publishing ------------------------------------------------------------

    # Same reasoning as `NerConfig.push_to_hub`: publishing sends the checkpoint
    # outside this machine, so it stays opt-in rather than a side effect of
    # `make rerank-train`.
    push_to_hub: bool = False


class HfConfig(BaseModel):
    """Where this project's artifacts live on the HuggingFace Hub.

    Five repos rather than one, because they version independently. Re-training
    the NER model forces a reranker re-train (it changes the spans the reranker
    is fed) but not the reverse, and the serving index is rebuilt on every
    GeoNames refresh while the query dataset is not — so pinning any one of them
    must not pin the others. Override via env with the ``HF__`` prefix.

    Each ``*_repo`` holds a full ``owner/name`` rather than being composed from a
    shared namespace, so a fork can redirect one repo without redirecting the
    rest.
    """

    ner_repo: str = "mki0809/geosearch-ner"
    reranker_repo: str = "mki0809/geosearch-reranker"
    # Serving artifacts: BM25 index, places table, candidate descriptions.
    index_repo: str = "mki0809/geosearch-index"
    # Training/eval corpora: synthetic queries, rerank pairs, golden set.
    datasets_repo: str = "mki0809/geosearch-queries"
    space_repo: str = "mki0809/geosearch"

    # Which commit to pull. "main" is right for development, but a number
    # reported in the thesis has to name the artifact it was measured on, so
    # every pull logs the sha it resolved to and every push prints the sha to
    # pin. Set these to a 40-char commit to freeze a run.
    ner_revision: str = "main"
    reranker_revision: str = "main"
    index_revision: str = "main"

    # Applies to repos this project *creates*. Existing repos keep whatever
    # visibility they already have — `create_repo(exist_ok=True)` does not
    # change it.
    private: bool = False

    # Local source of the Space: this directory's contents become the Space
    # root, so it cannot be the repository root (that would upload .git, data/
    # and the notebooks).
    space_dir: str = "space"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    # Where `make etl` writes the parsed GeoNames corpus, and where
    # `make artifacts` reads it back. Intermediate and git-ignored: it is
    # reproducible from the raw dumps, and what gets published is the compiled
    # artifact built from it, not this.
    corpus_dir: str = Field(default="data/corpus")

    geonames_data_dir: str = Field(default="data/raw")

    # Countries to load from GeoNames
    countries: list[str] = Field(default=["RU", "US", "TR", "CN"])

    # Languages to keep from alternateNamesV2
    languages: list[str] = Field(default=["ru", "en", "tr", "zh"])

    # NER model. Defaults to the fine-tuned checkpoint `make ner-train` writes —
    # this path and `NerConfig.model_dir` are deliberately the same, so training
    # publishes straight to what the engine loads. A hub id like
    # "urchade/gliner_multi-v2.1" works too: the value goes straight to
    # GLiNER.from_pretrained, which reads an existing local directory and only
    # falls back to the hub when the path is absent.
    gliner_model: str = Field(default="data/gliner-geosearch")
    ner_labels: list[str] = Field(default=["CITY", "REGION", "STATE", "COUNTRY"])

    # Serve with src.ner.tokenizer.CjkAwareSplitter instead of GLiNER's own
    # whitespace splitter. Required by the fine-tuned checkpoint — it was trained
    # on that tokenization and the splitter is not part of a saved model
    # (`words_splitter_type` in gliner_config.json only names the built-in kinds) —
    # and harmful to a zero-shot checkpoint, which never saw per-ideograph zh
    # tokens. Flip to false in the same change that points `gliner_model` back at
    # a hub checkpoint.
    ner_cjk_splitter: bool = Field(default=True)

    # Decision threshold for predict_entities. A parameter of serving, not of the
    # model: retrieval is recall-hungry — a city span never extracted is a city
    # that can never be retrieved, while a spurious span only adds a candidate the
    # reranker can demote — so the served operating point sits below GLiNER's own
    # 0.5 default. `make ner-eval --sweep` (src/ner/evaluate.py) is where this
    # number is derived; re-run it whenever the NER model changes.
    ner_threshold: float = Field(default=0.4)

    # What serving reads. One directory rather than a path per file: the four
    # files are only valid as a set (see src/search/artifacts.py), so naming
    # them separately would invite pointing half at one build and half at
    # another.
    artifacts_dir: str = Field(default="data/artifacts")
    excluded_feature_codes: list[str] = Field(default=["PPLH", "PPLQ", "PPLW", "PPLX"])

    # HuggingFace token, for pushing the trained NER model and for pulling it back
    # when `gliner_model` names a *private* hub repo. `huggingface_hub` reads this
    # same variable itself, so leaving it in the environment is enough for
    # `from_pretrained`; it is declared here so a missing token fails with a
    # message naming it instead of a 401.
    hf_token: str | None = Field(default=None)

    # Which Hub repos this project publishes to and pulls from (HF__ prefix).
    hf: HfConfig = Field(default_factory=HfConfig)

    # Synthetic query-dataset generation (secret from env, tunables nested).
    deepseek_api_key: str | None = Field(default=None)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # GLiNER fine-tuning data export (tunables nested under NER__).
    ner: NerConfig = Field(default_factory=NerConfig)

    # Reranker dataset build + training (tunables nested under RERANK__).
    rerank: RerankConfig = Field(default_factory=RerankConfig)


settings = Settings()
