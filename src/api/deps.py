from typing import Annotated

from fastapi import Depends, Request

from src.search.engine import SearchEngine


def get_engine(request: Request) -> SearchEngine:
    """Return the SearchEngine stored in application state."""
    return request.app.state.engine


# No DBSession dependency: `SearchEngine.search` hydrates through its own
# PlaceStore (src/search/places.py), so a request needs no database session —
# and in `artifacts` serving mode there is no database to open one against.
Engine = Annotated[SearchEngine, Depends(get_engine)]
