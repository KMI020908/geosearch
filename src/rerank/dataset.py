"""Build the labelled reranker training set from the query dataset.

The pipeline mirrors the notebook prototype, cleaned up:

1. Load one text *description* per in-scope geonameid from the database —
   sorted name spellings + country + region (:func:`build_descriptions`). This
   reuses the same name loaders as the query-dataset sampler, so the reranker
   sees exactly the documents retrieval indexes.
2. Replay every generated query through the live search API
   (:func:`search_batch`) to mine retrieval candidates.
3. For each query, emit its gold geonameid(s) as positives, the top retrieved
   non-gold candidates as hard negatives, and a few uniformly random
   documents from the whole corpus as easy negatives (:func:`build_pairs`).
4. Split by geonameid so a place never leaks across train/test
   (:func:`split`) and write both Parquets.

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

logger = logging.getLogger(__name__)

PAIR_SCHEMA = ["query", "document", "geonameid", "population", "retriever_score", "label"]
OUTPUT_COLS = ["query", "document", "population", "retriever_score", "label"]


def build_descriptions(
    name_rows: pl.DataFrame, admin1_names: pl.DataFrame
) -> dict[int, str]:
    """Map each geonameid to a document string: names, country, region.

    All spellings of a place are sorted and space-joined on the first line, the
    English country name on the second, the admin1 region on the third — the
    same document a retriever would index for that place.
    """
    cc_to_name = dict(countries_for_language("en"))
    grouped = (
        name_rows.join(admin1_names, on=["country_code", "admin1_code"], how="left")
        .with_columns(admin1_name=pl.col("admin1_name").fill_null(""))
        .group_by("geonameid", "country_code", "admin1_code")
        .agg(pl.col("name").alias("names"), pl.first("admin1_name").alias("admin1_name"))
    )
    descriptions: dict[int, str] = {}
    for row in grouped.iter_rows(named=True):
        names = " ".join(sorted(set(row["names"])))
        country = cc_to_name.get(row["country_code"], row["country_code"])
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


def build_pairs(
    results: list[dict],
    query_dataset: pl.DataFrame,
    descriptions: dict[int, str],
) -> pl.DataFrame:
    """Turn search results into labelled ``(query, document, ...)`` rows.

    Every retrieved candidate becomes one row, labelled 1 if it is one of the
    query's gold geonameids and 0 otherwise. Each row also carries the
    candidate's ``population`` and raw BM25 ``retriever_score`` — the same two
    numeric features the reranker reads off ``GeonameMatch`` at serve time, so
    online and offline features are identical. Because every row is a retrieved
    candidate, a gold place that retrieval missed contributes no positive (the
    reranker only ever ranks retrieved candidates anyway). Queries where NER
    found no entity are skipped (nothing to rank), as are candidates without a
    loaded description.
    """
    rows: list[tuple] = []
    for res, qrow in zip(results, query_dataset.iter_rows(named=True)):
        if not res["entities"]:
            continue
        pos_gids = set(qrow["geonameid"])
        for candidate in res["results"]:
            gid = candidate["geonameid"]
            if gid not in descriptions:
                continue
            label = 1 if gid in pos_gids else 0
            rows.append((
                res["query"],
                descriptions[gid],
                gid,
                candidate["population"],
                candidate["retriever_score"],
                label,
            ))

    return pl.DataFrame(rows, schema=PAIR_SCHEMA, orient="row")


def split(
    pairs: pl.DataFrame, cfg: RerankConfig
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Hold out ``test_size`` of distinct geonameids for test, rest for train.
    """
    gids = pairs.select("geonameid").unique(maintain_order=True)
    test_gids = gids.sample(fraction=cfg.test_size, seed=cfg.seed)
    is_test = pairs["geonameid"].is_in(test_gids["geonameid"].to_list())
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
    logger.info("Searching %d queries via %s…", query_dataset.height, cfg.search_url)
    results = search_batch(query_dataset["query"].to_list(), cfg)

    pairs = build_pairs(results, query_dataset, descriptions)
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
