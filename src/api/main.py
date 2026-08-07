import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from src.api.routes import health_router, router
from src.config import settings
from src.search.engine import SearchEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Build the search engine once at startup and tear down cleanly."""
    logger.info("Starting up — building search engine…")
    app.state.engine = await SearchEngine.build(settings)
    logger.info("Search engine ready — serving requests")
    yield
    logger.info("Shutting down")


app = FastAPI(
    title="GeoSearch",
    description=(
        "Multilingual toponym recognition and matching. "
        "Given free-text input, identifies city/place mentions and returns "
        "ranked GeoNames entries."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router, prefix="/v1")
app.include_router(health_router)
