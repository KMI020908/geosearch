from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class DatasetConfig(BaseModel):
    """Tunables for the synthetic query-dataset generator.

    The dataset (city queries + the candidate name pool) feeds several pipeline
    steps, not just the reranker. Everything the generation script needs lives
    here so nothing is hardcoded in the script itself. Override individual
    fields via env with the ``DATASET__`` prefix (e.g.
    ``DATASET__DEEPSEEK_MODEL=deepseek-chat``).
    """

    seed: int = 42

    # Population-stratified sampling per (language, country): always take the
    # `n_top` most-populous and `n_low` least-populous names, plus `n_mid`
    # random from the middle, so the dataset spans big cities and obscure
    # villages alike.
    n_top: int = 2
    n_mid: int = 2
    n_low: int = 2

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
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"
    temperature: float = 1.3
    reasoning_effort: str = "low"
    max_retries: int = 5
    request_timeout: float = 60.0
    # Concurrent in-flight requests. DeepSeek is rate-limited server-side, so
    # the SDK's 429 backoff (max_retries) does the throttling for us.
    max_workers: int = 8

    output_path: str = "data/query_dataset.parquet"
    # Append-only JSONL of completed rows; lets a crashed/interrupted run
    # warm-start and skip everything already generated.
    checkpoint_path: str = "data/query_dataset.checkpoint.jsonl"
    # The sample plan (one row per intended query, before generation), saved so
    # downstream steps can reuse the same sampled set of names.
    plan_path: str = "data/query_plan.parquet"


class RerankConfig(BaseModel):
    """Tunables for building and training the reranker.

    ``rerank-data`` turns the query dataset + live search API into labelled
    ``(query, document, label)`` pairs; ``rerank-train`` fits a CatBoost ranker
    on them. Override via env with the ``RERANK__`` prefix.
    """

    seed: int = 42

    # Live search API to mine retrieval candidates from (the app must be up).
    search_url: str = "http://localhost:8000/v1/search"
    top_k: int = 50
    request_timeout: float = 30.0

    # Fraction of distinct geonameids held out for test (split by geonameid so
    # a place never appears in both train and test).
    test_size: float = 0.2

    query_dataset_path: str = "data/query_dataset.parquet"
    train_path: str = "data/rerank_train.parquet"
    test_path: str = "data/rerank_test.parquet"
    model_path: str = "data/rerank_model.cbm"

    # CatBoost ranker (YetiRank loss, NDCG eval).
    iterations: int = 1000
    learning_rate: float = 0.05
    l2_leaf_reg: float = 3.0
    early_stopping_rounds: int = 100
    ndcg_top: int = 10


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_nested_delimiter="__",
    )

    postgres_user: str
    postgres_password: str
    postgres_db: str
    # Full async DSN; when set (e.g. in Docker) it overrides the localhost
    # default built from the fields above.
    database_url: str | None = Field(default=None)

    geonames_data_dir: str = Field(default="data/raw")

    # Countries to load from GeoNames
    countries: list[str] = Field(default=["RU", "US", "TR", "CN"])

    # Languages to keep from alternateNamesV2
    languages: list[str] = Field(default=["ru", "en", "tr", "zh"])

    # NER model
    gliner_model: str = Field(default="urchade/gliner_multi-v2.1")
    ner_labels: list[str] = Field(default=["CITY", "REGION", "STATE", "COUNTRY"])

    # Retrieval BM25 index
    index_path: str = Field(default="data/bm25_index.pkl")
    index_warm_start: bool = Field(default=False)
    excluded_feature_codes: list[str] = Field(
        default=["PPLH", "PPLQ", "PPLW", "PPLX"]
    )

    # Synthetic query-dataset generation (secret from env, tunables nested).
    deepseek_api_key: str | None = Field(default=None)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)

    # Reranker dataset build + training (tunables nested under RERANK__).
    rerank: RerankConfig = Field(default_factory=RerankConfig)


settings = Settings()
