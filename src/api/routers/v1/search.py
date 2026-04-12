"""Search router: GET /v1/search"""
import uuid
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request

from ....models.city_detector import CityDetector
from ...schemas import CityResult, SearchResponse

router = APIRouter(prefix="/search", tags=["search"])


def _get_detector(request: Request) -> CityDetector:
    return request.app.state.detector


@router.get("", response_model=SearchResponse)
def search(
    query: Annotated[str, Query(min_length=0, description="Search query text")],
    top_k: Annotated[int, Query(ge=1, le=100, description="Max results")] = 20,
    context_id: Annotated[
        str | None,
        Query(description="Optional trace ID for comparing runs; auto-generated if omitted"),
    ] = None,
    detector: CityDetector = Depends(_get_detector),
) -> SearchResponse:
    if context_id is None:
        context_id = str(uuid.uuid4())

    hits = detector.detect(query, top_k=top_k, context_id=context_id)

    return SearchResponse(
        context_id=context_id,
        query=query,
        results=[CityResult(**h) for h in hits],
        total=len(hits),
    )
