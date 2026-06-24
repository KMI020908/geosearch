from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

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


settings = Settings()
