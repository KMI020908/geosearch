"""FastAPI application factory."""
import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response

from ..core.logging import setup_logging
from ..data.config import GLINER_MODEL, BM25_INDEX_PATH, BIENCODER_INDEX_PATH
from ..models.bm25_index import BM25Index
from ..models.city_detector import CityDetector
from ..models.biencoder_index import BiEncoderIndex
from .routers.v1 import search as search_v1

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    if not BM25_INDEX_PATH.exists():
        raise RuntimeError(
            f"Index not found at {BM25_INDEX_PATH}. "
            "Build it first: python -m src.data.build_index"
        )
    logger.info("Loading BM25 index", extra={"path": str(BM25_INDEX_PATH)})
    index_bm25 = BM25Index.load(BM25_INDEX_PATH)

    index_biencoder = None
    if not BIENCODER_INDEX_PATH.exists():
        raise RuntimeError(
            f"Index not found at {BIENCODER_INDEX_PATH}. "
            "Build it first: python -m src.data.build_index"
        )
    logger.info("Loading retriever index", extra={"path": str(BIENCODER_INDEX_PATH)})
    index_biencoder = BiEncoderIndex.load(BIENCODER_INDEX_PATH)

    logger.info("Loading GLiNER model", extra={"model": GLINER_MODEL})
    app.state.detector = CityDetector(
        gliner_model=GLINER_MODEL, index_bm25=index_bm25, index_biencoder=index_biencoder
    )
    logger.info("Application ready")
    yield
    del app.state.detector
    logger.info("Application shutdown")


def create_app() -> FastAPI:
    app = FastAPI(
        title="GeoSearch",
        description="GLiNER + BM25 multilingual city detection.",
        version="0.2.0",
        lifespan=lifespan,
    )

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        elapsed_ms = round((time.perf_counter() - start) * 1000, 2)
        logger.info(
            "HTTP %s %s → %d (%.1f ms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            extra={
                "method": request.method,
                "path": request.url.path,
                "status_code": response.status_code,
                "duration_ms": elapsed_ms,
            },
        )
        return response

    @app.get("/health", tags=["ops"], summary="Liveness check")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(search_v1.router, prefix="/v1")
    return app
