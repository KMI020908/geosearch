"""FastAPI application factory."""
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ..data.config import INDEX_PATH
from ..models.bm25_index import GeoSearchIndex
from .routers.v1 import search as search_v1


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not INDEX_PATH.exists():
        raise RuntimeError(
            f"Index not found at {INDEX_PATH}. "
            "Build it first: python src/data/build_index.py"
        )
    print(f"Loading BM25 index from {INDEX_PATH} ...")
    app.state.index = GeoSearchIndex.load(INDEX_PATH)
    print("Index ready.")
    yield
    del app.state.index


def create_app() -> FastAPI:
    app = FastAPI(
        title="GeoSearch",
        description="BM25-powered multilingual city search.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.include_router(search_v1.router, prefix="/v1")
    return app
