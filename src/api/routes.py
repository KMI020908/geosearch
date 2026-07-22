from fastapi import APIRouter, Query

from src.api.deps import DBSession, Engine
from src.api.schemas import GeonameResult, SearchResponse

router = APIRouter()
# Liveness probe stays unversioned — it's about the process, not the API contract.
health_router = APIRouter()


@router.get("/search", response_model=SearchResponse)
async def search(
    engine: Engine,
    session: DBSession,
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
    entities, matches = await engine.search(
        text, top_k=top_k, session=session, use_rerank=use_rerank
    )
    return SearchResponse(
        query=text,
        entities=entities,
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
            for m, score in matches
        ],
        total=len(matches),
    )


@health_router.get("/health")
async def health() -> dict[str, str]:
    """Liveness probe — returns 200 once the app has started."""
    return {"status": "ok"}
