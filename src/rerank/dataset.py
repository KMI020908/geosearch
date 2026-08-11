"""Build the labelled reranker training set from the query dataset.

The pipeline mirrors the notebook prototype, cleaned up:

1. Load one text *description* per in-scope geonameid — sorted name spellings +
   country + region (:func:`build_descriptions`) — through
   :func:`src.search.artifacts.read_descriptions`, the same call the engine
   makes. So the documents a model is *mined against* are the documents it is
   later *served* with.
2. Replay every generated query through the live search API
   (:func:`search_batch`) to mine retrieval candidates.
3. Turn every retrieved candidate into a labelled pair (:func:`build_pairs`),
   labelling gold geonameid(s) positive and every other retrieved candidate
   negative. ``query_text`` is by default the NER spans the API returns
   (``" ".join(entities)`` — the same string BM25 retrieval scores against), or
   the dataset's gold spans when ``RERANK__USE_GOLD_ENTITIES=true`` (see
   :func:`_query_text`).
4. Split by query — the ranking group — so no query (and its gold) leaks
   across train/test (:func:`split`) and write both Parquets.

Runs in process by default; set ``RERANK__SEARCH_URL`` to replay against a
live server instead (:mod:`src.search.batch`). Run as::

    python -m src.rerank.dataset
"""

from __future__ import annotations

import asyncio
import logging

import polars as pl
from country_list import countries_for_language

from src.config import RerankConfig, settings
from src.rerank.features import GOLD_BUCKET_COLS, NAME_SEPARATOR
from src.search.artifacts import read_descriptions
from src.search.batch import search_batch

logger = logging.getLogger(__name__)

# One mined row: the (query_text, document) pair the cross-encoder scores, the
# raw ``query`` (kept only as the ranking group id at train time and the split
# key — never fed to the model), and the ``label`` target.
OUTPUT_COLS = ["query_text", "document", "query", "label"]


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


def _query_text(res: dict, qrow: dict, use_gold: bool) -> str:
    """The query's text side: gold dataset spans or live NER spans, joined flat.

    ``use_gold=False`` (default) joins ``entities`` off the API response — the
    exact ``" ".join(entities)`` string the engine feeds both BM25 retrieval and
    the reranker, which is what keeps train/serve features identical.
    ``use_gold=True`` joins the dataset's reference spans instead
    (:data:`~src.rerank.features.GOLD_BUCKET_COLS`, city/country/admin1 order),
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
        parts = [qrow[col] or "" for col in GOLD_BUCKET_COLS.values()]
        return " ".join(p for p in parts if p)

    # A missing field means the API is on an older build that predates it —
    # replaying against it would silently mislabel the pairs. Fail loudly with a
    # fix instead of a bare KeyError.
    if "entities" not in res:
        raise RuntimeError(
            "Search API response has no 'entities' field — the server is "
            "running an outdated build. Restart it on the current code "
            "(uv run uvicorn src.api.main:app) before running rerank-data."
        )
    return " ".join(res["entities"])


def build_pairs(
    results: list[dict],
    query_dataset: pl.DataFrame,
    descriptions: dict[int, str],
    *,
    use_gold_entities: bool = False,
) -> pl.DataFrame:
    """Turn search results into labelled ``(query_text, document, label)`` rows.

    ``query_text`` is the raw text side the online cross-encoder
    :class:`~src.rerank.model.Reranker` scores against: by default the same
    ``" ".join(entities)`` string BM25 retrieval used, or the dataset's gold
    spans when ``use_gold_entities`` is set (:func:`_query_text`). A candidate is
    labelled 1 if it is one of the query's gold geonameids, else 0.

    ``use_gold_entities`` is an ablation — "how would the reranker do if NER were
    perfect?" — and only replaces ``query_text``. The candidate set still comes
    from retrieval driven by the NER spans (the API takes text, not spans), and
    queries where NER found nothing retrieve nothing and are skipped below. So it
    does not recover recall lost to NER misses, and the resulting model is not
    servable: online the reranker always sees NER spans.

    Every retrieved candidate becomes a row (no negative subsampling). Queries
    where NER found no entity are skipped (nothing to rank), as are candidates
    without a loaded description (a gold place retrieval missed contributes no
    positive — the reranker only ever ranks retrieved candidates anyway).
    """
    # Responses line up with dataset rows positionally (see :func:`search_batch`),
    # and that alignment carries the labels — plus ``query_text`` itself under
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
        query_text = _query_text(res, qrow, use_gold_entities)
        for candidate in res["results"]:
            gid = candidate["geonameid"]
            if gid not in descriptions:
                continue
            rows.append(
                {
                    "query_text": query_text,
                    "document": descriptions[gid],
                    "query": res["query"],
                    "label": 1 if gid in pos_gids else 0,
                }
            )

    if not rows:
        return pl.DataFrame(
            schema=dict.fromkeys(OUTPUT_COLS, pl.Utf8) | {"label": pl.Int64}
        )
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

    # The same file the engine serves from. `descriptions.parquet` is written by
    # this very `build_descriptions`, so these are not an approximation of the
    # documents the model will be scored against — they are those documents.
    descriptions = asyncio.run(read_descriptions(settings))
    logger.info("Loaded %d descriptions", len(descriptions))

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
        logger.info("Entity source: NER spans (entities from the API)")

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
