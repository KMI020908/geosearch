from fastapi import APIRouter, Query

from src.api.deps import Engine
from src.api.schemas import EntitySpanResult, GeonameResult, SearchResponse
from src.search.engine import SearchResult

router = APIRouter()
# Liveness probe stays unversioned — it's about the process, not the API contract.
health_router = APIRouter()


def to_response(query: str, result: SearchResult) -> SearchResponse:
    """Render a pipeline result as the endpoint's response body.

    A named function rather than an inline literal in the handler, because
    :mod:`src.search.batch` renders the same shape when replaying queries in
    process. Sharing the builder is what makes "in-process and HTTP return the
    same dicts" true by construction instead of by inspection — and what keeps
    a field added here from being invisible to the reranker's mining.
    """
    return SearchResponse(
        query=query,
        entities=result.entities,
        spans=[
            EntitySpanResult(text=s.text, label=s.label, start=s.start, end=s.end)
            for s in result.spans
        ],
        entity_buckets=result.entity_buckets,
        results=[
            GeonameResult(
                geonameid=m.geonameid,
                asciiname=m.asciiname,
                country_code=m.country_code,
                population=m.population,
                feature_code=m.feature_code,
                latitude=m.latitude,
                longitude=m.longitude,
                score=score,
                retriever_score=m.retriever_score,
            )
            for m, score in result.ranked
        ],
        total=len(result.ranked),
    )


@router.get("/search", response_model=SearchResponse)
async def search(
    engine: Engine,
    text: str = Query(..., min_length=1, description="Free-text query"),
    top_k: int = Query(default=50, ge=1, le=100, description="Number of results"),
    use_rerank: bool = Query(
        default=True, description="Reorder with the trained reranker (if loaded)"
    ),
) -> SearchResponse:
    """Search for populated places matching the geographic entities in *text*.

    The text is run through GLiNER to extract city/region/country spans,
    then the BM25 index retrieves the best matching GeoNames entries, which
    are reordered by the trained reranker (or population sort as a fallback).
    """
    result = await engine.search(text, top_k=top_k, use_rerank=use_rerank)
    return to_response(text, result)


@health_router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 once the app has started."""
    return {"status": "ok"}
