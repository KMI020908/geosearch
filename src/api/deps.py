from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.db.session import get_session
from src.search.engine import SearchEngine


def get_engine(request: Request) -> SearchEngine:
    """Return the SearchEngine stored in application state."""
    return request.app.state.engine


DBSession = Annotated[AsyncSession, Depends(get_session)]
Engine = Annotated[SearchEngine, Depends(get_engine)]
