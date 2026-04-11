"""Search router: GET /v1/search"""
from typing import Annotated

from fastapi import APIRouter, Query, Request

from ...schemas import CityResult, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


@router.get("", response_model=SearchResponse)
def search(
    request: Request,
    query: Annotated[str, Query(min_length=0, description="Search query")],
    top_k: Annotated[int, Query(ge=1, le=100, description="Max results")] = 20,
) -> SearchResponse:
    detector = request.app.state.detector
    hits = detector.detect(query, top_k=top_k)
    return SearchResponse(
        query=query,
        results=[CityResult(**h) for h in hits],
        total=len(hits),
    )
