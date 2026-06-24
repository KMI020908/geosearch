from pydantic import BaseModel, Field


class GeonameResult(BaseModel):
    geonameid: int
    asciiname: str
    country_code: str
    population: int
    feature_code: str | None
    latitude: float | None
    longitude: float | None


class SearchResponse(BaseModel):
    query: str
    entities: list[str] = Field(
        description="Geographic entities extracted by NER (empty when use_ner=false)"
    )
    results: list[GeonameResult]
    total: int
