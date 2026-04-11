"""FastAPI application factory."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..data.config import GLINER_MODEL, INDEX_PATH
from ..models.bm25_index import GeoSearchIndex
from ..models.city_detector import CityDetector
from .routers.v1 import search as search_v1


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"Index not found at {INDEX_PATH}. "
            "Build it first: python -m src.data.build_index"
        )
    print(f"Loading BM25 index from {INDEX_PATH} ...")
    index = GeoSearchIndex.load(INDEX_PATH)
    print(f"Loading GLiNER model '{GLINER_MODEL}' ...")
    app.state.detector = CityDetector(gliner_model=GLINER_MODEL, index=index)
    print("Ready.")
    yield
    del app.state.detector


def create_app() -> FastAPI:
    app = FastAPI(
        title="GeoSearch",
        description="GLiNER + BM25 multilingual city detection.",
        version="0.2.0",
        lifespan=lifespan,
    )
    app.include_router(search_v1.router, prefix="/v1")
    return app
