"""Build the labelled reranker training set from the query dataset.

The pipeline mirrors the notebook prototype, cleaned up:

1. Load one text *description* per in-scope geonameid from the database —
   sorted name spellings + country + region (:func:`build_descriptions`). This
   reuses the same name loaders as the query-dataset sampler, so the reranker
   sees exactly the documents retrieval indexes.
2. Replay every generated query through the live search API
   (:func:`search_batch`) to mine retrieval candidates.
3. Featurise every retrieved candidate (:func:`build_pairs` via
   :mod:`src.rerank.features`), labelling gold geonameid(s) positive and every
   other retrieved candidate negative. The text features are the query's entity
   buckets — by default the NER spans the API returns, or the dataset's gold
   spans when ``RERANK__USE_GOLD_ENTITIES=true`` (see :func:`_entity_buckets`).
4. Split by query — the ranking group — so no query (and its gold) leaks
   across train/test (:func:`split`) and write both Parquets.

The API server must be running (``RERANK__SEARCH_URL``). Run as::

    python -m src.rerank.dataset
"""

from __future__ import annotations

import asyncio
import logging

import httpx
import polars as pl
from country_list import countries_for_language
from tqdm import tqdm

from src.config import RerankConfig, settings
from src.dataset.sampling import load_admin1_names, load_name_rows
from src.db.session import AsyncSessionFactory
from src.rerank.features import (
    FEATURE_COLUMNS,
    GOLD_BUCKET_COLS,
    NAME_SEPARATOR,
    build_row,
)

logger = logging.getLogger(__name__)

# One mined row: the shared feature columns, the raw ``query`` (kept only as the
# ranking group id at train time and the split key — never fed to the model),
# and the ``label`` target.
OUTPUT_COLS = [*FEATURE_COLUMNS, "query", "label"]


def build_descriptions(
    name_rows: pl.DataFrame, admin1_names: pl.DataFrame
) -> dict[int, str]:
    """Map each geonameid to a document string: names, country, region.

    All spellings of a place are sorted and joined on the first line, the English
    country name on the second, the admin1 region on the third — the same
    document a retriever would index for that place.

    The spellings are separated by :data:`~src.rerank.features.NAME_SEPARATOR`
    rather than a plain space so the boundary between them survives: the match
    features compare the query's spans against each name individually, which a
    space-joined line makes impossible — and which would dilute the comparison by
    however many spellings a place happens to have.
    """
    cc_to_name = dict(countries_for_language("en"))
    grouped = (
        name_rows.join(admin1_names, on=["country_code", "admin1_code"], how="left")
        # Both lines of the description are nullable in the join, and a null
        # reaching `str.join` below is a TypeError rather than a blank line.
        .with_columns(
            admin1_name=pl.col("admin1_name").fill_null(""),
            country_code=pl.col("country_code").fill_null(""),
        )
        .group_by("geonameid", "country_code", "admin1_code")
        .agg(
            pl.col("name").alias("names"), pl.first("admin1_name").alias("admin1_name")
        )
    )
    descriptions: dict[int, str] = {}
    for row in grouped.iter_rows(named=True):
        names = NAME_SEPARATOR.join(sorted(set(row["names"])))
        country = cc_to_name.get(row["country_code"], row["country_code"]) or ""
        descriptions[row["geonameid"]] = "\n".join(
            [names, country, row["admin1_name"]]
        ).strip()
    return descriptions


def search_batch(queries: list[str], cfg: RerankConfig) -> list[dict]:
    """Query the live search API once per query, returning the JSON responses.

    Synchronous on purpose: the endpoint is the bottleneck and ordering matters
    (responses line up with ``queries`` positionally).
    """
    results: list[dict] = []
    with httpx.Client(timeout=cfg.request_timeout) as client:
        for query in tqdm(queries, unit="q"):
            response = client.get(
                cfg.search_url,
                # use_rerank=False: mine hard negatives from raw retrieval, so
                # the training set never depends on the reranker it will train.
                params={"text": query, "top_k": cfg.top_k, "use_rerank": False},
            )
            response.raise_for_status()
            results.append(response.json())
    return results


def _entity_buckets(res: dict, qrow: dict, use_gold: bool) -> dict[str, str]:
    """The query's typed entity strings: gold dataset spans or live NER spans.

    ``use_gold=False`` (default) reads ``entity_buckets`` off the API response —
    the GLiNER spans the engine itself feeds the reranker, which is what keeps
    train/serve features identical. ``use_gold=True`` reads the dataset's
    reference spans instead (:data:`~src.rerank.features.GOLD_BUCKET_COLS`),
    which the generator wrote with the very same ``bucket_entities``, so the two
    sources are drop-in comparable.
    """
    if use_gold:
        missing = [c for c in GOLD_BUCKET_COLS.values() if c not in qrow]
        if missing:
            raise RuntimeError(
                f"use_gold_entities=True but the query dataset has no {missing} "
                "column(s) — it predates the gold entity buckets. Regenerate it "
                "(make dataset) or unset RERANK__USE_GOLD_ENTITIES."
            )
        # ``or ""``: build_row needs a str, and a Parquet column may hold nulls.
        return {name: qrow[col] or "" for name, col in GOLD_BUCKET_COLS.items()}

    # A missing bucket means the API is on an older build that predates the
    # typed-entity split — replaying against it would silently mislabel the
    # features. Fail loudly with a fix instead of a bare KeyError.
    if "entity_buckets" not in res:
        raise RuntimeError(
            "Search API response has no 'entity_buckets' field — the server "
            "is running an outdated build. Restart it on the current code "
            "(uv run uvicorn src.api.main:app) before running rerank-data."
        )
    return res["entity_buckets"]


def build_pairs(
    results: list[dict],
    query_dataset: pl.DataFrame,
    descriptions: dict[int, str],
    *,
    use_gold_entities: bool = False,
) -> pl.DataFrame:
    """Turn search results into labelled feature rows, one per candidate.

    Every retrieved candidate is featurised via :func:`src.rerank.features.build_row`
    — the same builder the online :class:`~src.rerank.model.Reranker` uses, so
    train- and serve-time features are identical by construction. The text
    features are the query's entity spans split by type, not the raw query:
    ``entity_buckets`` from the API response (the NER spans the engine actually
    feeds the reranker) by default, or the dataset's gold spans when
    ``use_gold_entities`` is set (:func:`_entity_buckets`). A candidate is
    labelled 1 if it is one of the query's gold geonameids, else 0, and
    ``retriever_rank`` is its 0-based position in the retrieved list.

    ``use_gold_entities`` is an ablation — "how would the reranker do if NER were
    perfect?" — and only replaces the *text features*. The candidate set still
    comes from retrieval driven by the NER spans (the API takes text, not spans),
    and queries where NER found nothing retrieve nothing and are skipped below.
    So it does not recover recall lost to NER misses, and the resulting model is
    not servable: online the reranker always sees NER spans.

    Every retrieved candidate becomes a row (no negative subsampling). Queries
    where NER found no entity are skipped (nothing to rank), as are candidates
    without a loaded description (a gold place retrieval missed contributes no
    positive — the reranker only ever ranks retrieved candidates anyway).
    """
    # Responses line up with dataset rows positionally (see :func:`search_batch`),
    # and that alignment carries the labels — plus the features themselves under
    # ``use_gold_entities``. A length mismatch would silently pair the wrong rows.
    if len(results) != query_dataset.height:
        raise ValueError(
            f"got {len(results)} search results for {query_dataset.height} query "
            "dataset rows — they must line up positionally"
        )

    rows: list[dict] = []
    for res, qrow in zip(results, query_dataset.iter_rows(named=True), strict=True):
        if not res["entities"]:
            continue
        pos_gids = set(qrow["geonameid"])
        entity_buckets = _entity_buckets(res, qrow, use_gold_entities)
        for rank, candidate in enumerate(res["results"]):
            gid = candidate["geonameid"]
            if gid not in descriptions:
                continue
            row = build_row(
                **entity_buckets,
                document=descriptions[gid],
                population=candidate["population"],
                retriever_score=candidate["retriever_score"],
                retriever_rank=rank,
            )
            row["query"] = res["query"]
            row["label"] = 1 if gid in pos_gids else 0
            rows.append(row)

    if not rows:
        return pl.DataFrame(schema={c: pl.Float64 for c in OUTPUT_COLS})
    return pl.DataFrame(rows).select(OUTPUT_COLS)


def split(pairs: pl.DataFrame, cfg: RerankConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out ``test_size`` of distinct queries for test, rest for train.

    The split unit is the query — the ranking *group* the model learns and is
    scored within. Holding out whole queries keeps each query's full candidate
    set on one side of the split, so no query text leaks train↔test and every
    test group keeps its gold positive. Splitting by geonameid instead would
    tear a query's candidates across both sets, leaking the query and leaving
    truncated (or positive-less) test groups that make NDCG uninformative.
    """
    queries = pairs.select("query").unique(maintain_order=True)
    test_queries = queries.sample(fraction=cfg.test_size, seed=cfg.seed)
    is_test = pairs["query"].is_in(test_queries["query"].to_list())
    train = pairs.filter(~is_test).select(OUTPUT_COLS)
    test = pairs.filter(is_test).select(OUTPUT_COLS)
    return train, test


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = settings.rerank

    logger.info("Loading city descriptions from database…")
    name_rows = asyncio.run(load_name_rows(AsyncSessionFactory, settings))
    admin1_names = load_admin1_names(settings.geonames_data_dir, settings.countries)
    descriptions = build_descriptions(name_rows, admin1_names)
    logger.info("Built %d descriptions", len(descriptions))

    query_dataset = pl.read_parquet(cfg.query_dataset_path)

    # The mined Parquet carries no marker of which entity source produced it and
    # the output paths do not differ, so the log line is the only record of the
    # run's mode — make the ablation impossible to mistake for a real training set.
    if cfg.use_gold_entities:
        logger.warning(
            "Entity source: GOLD spans from %s (RERANK__USE_GOLD_ENTITIES=true). "
            "This is an ablation — a model trained on it is NOT servable, since "
            "the online reranker always gets NER spans.",
            cfg.query_dataset_path,
        )
    else:
        logger.info("Entity source: NER spans (entity_buckets from the API)")

    logger.info("Searching %d queries via %s…", query_dataset.height, cfg.search_url)
    results = search_batch(query_dataset["query"].to_list(), cfg)

    pairs = build_pairs(
        results, query_dataset, descriptions, use_gold_entities=cfg.use_gold_entities
    )
    train, test = split(pairs, cfg)
    train.write_parquet(cfg.train_path)
    test.write_parquet(cfg.test_path)
    logger.info(
        "Wrote %d train / %d test pairs to %s, %s",
        train.height,
        test.height,
        cfg.train_path,
        cfg.test_path,
    )


if __name__ == "__main__":
    main()
