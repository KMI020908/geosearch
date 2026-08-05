from pydantic import BaseModel, Field


class GeonameResult(BaseModel):
    geonameid: int
    asciiname: str
    country_code: str
    population: int
    feature_code: str | None
    latitude: float | None
    longitude: float | None
    score: float = Field(
        description=(
            "Final ranking score: the reranker model score when use_rerank=true, "
            "otherwise equal to retriever_score."
        )
    )
    retriever_score: float = Field(
        description=(
            "Raw BM25 retriever score for the matched name, independent of "
            "use_rerank. A reranker training feature (see src/rerank)."
        )
    )


class SearchResponse(BaseModel):
    query: str
    entities: list[str] = Field(
        description="Geographic entities extracted by NER (empty when use_ner=false)"
    )
    entity_buckets: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The NER spans grouped by type (city_entities / country_entities / "
            "admin1_entities), each a space-joined string. This is what the "
            "reranker scores, and what the reranker dataset builder mines."
        ),
    )
    results: list[GeonameResult]
    total: int
