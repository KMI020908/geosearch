"""Run a batch of queries through the pipeline, in process or over HTTP.

Both callers — ``make rerank-data`` (mining training features) and
``make rerank-eval`` (the golden-set gate) — used to require a separately
started server. That was inherited from the notebook prototype rather than
chosen, and it buys nothing that matters here: the engine is importable and
database-free, so calling it directly runs the *same* ``SearchEngine.search``
the route calls, and the route adds only serialisation.

What it does cost is a failure mode. Replaying against a server left running on
older code silently mines features from a different pipeline than the one that
will serve them — a hazard real enough that
:func:`src.rerank.dataset._entity_buckets` carries a hand-written check for one
symptom of it. In process, the skew cannot exist: it is the same code object.

HTTP is kept, opt-in via ``RERANK__SEARCH_URL``, for the one case where it says
something in-process cannot: measuring an actual deployment, including its
routes and response schema.

Both paths return the same dicts. The in-process path builds them from
:class:`~src.api.schemas.SearchResponse`, the model the endpoint itself
serialises, so "same shape" is by construction rather than by inspection.
"""

from __future__ import annotations

import asyncio
import logging

from tqdm import tqdm

from src.config import RerankConfig, settings

logger = logging.getLogger(__name__)


# Built once per process. `rerank-eval` runs the same query set twice (reranker
# off, then on) and construction is ~15s, nearly all of it loading GLiNER, so
# rebuilding per call would double the cost of the command for nothing.
_cached_engine = None


def _engine():
    """Return the process-wide engine, building it on first use."""
    global _cached_engine
    if _cached_engine is None:
        from src.search.engine import SearchEngine

        logger.info("Building the search engine in process…")
        _cached_engine = asyncio.run(SearchEngine.build(settings))
    return _cached_engine


def _response_dict(query: str, result) -> dict:
    """Render a :class:`SearchResult` exactly as the endpoint would."""
    from src.api.routes import to_response

    return to_response(query, result).model_dump()


def _in_process(
    queries: list[str], cfg: RerankConfig, *, use_rerank: bool
) -> list[dict]:
    engine = _engine()
    results = []
    for query in tqdm(queries, unit="q"):
        result = asyncio.run(
            engine.search(query, top_k=cfg.top_k, use_rerank=use_rerank)
        )
        results.append(_response_dict(query, result))
    return results


def _over_http(
    queries: list[str], cfg: RerankConfig, *, use_rerank: bool
) -> list[dict]:
    import httpx

    assert cfg.search_url
    results = []
    with httpx.Client(timeout=cfg.request_timeout) as client:
        for query in tqdm(queries, unit="q"):
            response = client.get(
                cfg.search_url,
                params={"text": query, "top_k": cfg.top_k, "use_rerank": use_rerank},
            )
            response.raise_for_status()
            results.append(response.json())
    return results


def search_batch(
    queries: list[str], cfg: RerankConfig, *, use_rerank: bool = False
) -> list[dict]:
    """Run *queries*, returning one response dict each, in order.

    Order is preserved and matters: callers line the results up against their
    input positionally.
    """
    if cfg.search_url:
        logger.info("Replaying %d queries against %s", len(queries), cfg.search_url)
        return _over_http(queries, cfg, use_rerank=use_rerank)
    logger.info("Replaying %d queries in process", len(queries))
    return _in_process(queries, cfg, use_rerank=use_rerank)
