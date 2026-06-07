from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = Field(
        default="postgresql+asyncpg://geosearch:geosearch@localhost:5432/geosearch"
    )
    geonames_data_dir: str = Field(default="data/raw")

    # Countries to load from GeoNames
    countries: list[str] = Field(default=["RU", "US", "TR", "CN"])

    # Languages to keep from alternateNamesV2
    languages: list[str] = Field(default=["ru", "en", "tr", "zh"])


settings = Settings()
