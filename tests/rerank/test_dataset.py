"""Unit tests for reranker dataset building — no DB, no live API needed."""

import math

import polars as pl
import pytest

from src.config import RerankConfig
from src.rerank.dataset import OUTPUT_COLS, build_pairs, split
from src.rerank.features import build_row

# Positives are unique per query; negatives are drawn from a small shared pool,
# which mirrors real data where a handful of popular places show up as negatives
# across many unrelated queries. The split unit is the *query* (the ranking
# group), so each held-out query keeps its full candidate set intact on one side.
_N_QUERIES = 20
_NEG_POOL = [90001, 90002, 90003, 90004, 90005]


def _row(
    query: str, document: str, population: int, score: float, rank: int, label: int
):
    # The entity content is irrelevant to the split tests — stash the query in
    # one bucket so the row is well-formed; the other buckets stay empty.
    row = build_row(
        city_entities=query,
        country_entities="",
        admin1_entities="",
        document=document,
        population=population,
        retriever_score=score,
        retriever_rank=rank,
    )
    row["query"] = query
    row["label"] = label
    return row


def _synthetic_pairs() -> pl.DataFrame:
    rows: list[dict] = []
    for i in range(_N_QUERIES):
        query = f"query-{i}"
        rows.append(_row(query, f"City-{i}", 100 + i, 1.0, 0, 1))
        for rank, neg in enumerate(_NEG_POOL, start=1):
            rows.append(_row(query, f"Neg-{neg}", 50, 0.5, rank, 0))
    return pl.DataFrame(rows).select(OUTPUT_COLS)


def test_split_keeps_each_querys_full_candidate_set_not_just_the_gold_row() -> None:
    pairs = _synthetic_pairs()
    cfg = RerankConfig(test_size=0.5, seed=42)

    train, test = split(pairs, cfg)

    assert test.height > 0
    # A held-out query carries its full group — one positive plus the shared
    # negatives — so the test label mean is the natural 1-in-6, never the
    # degenerate all-positives a gold-row-only split would produce.
    assert test["label"].mean() < 0.9


def test_split_never_leaks_a_query_across_train_and_test() -> None:
    pairs = _synthetic_pairs()
    cfg = RerankConfig(test_size=0.5, seed=42)

    train, test = split(pairs, cfg)

    # The query is the ranking group_id, so it must land wholly on one side of
    # the split — nothing the model trains on may reappear at eval.
    train_queries = set(train["query"].to_list())
    test_queries = set(test["query"].to_list())
    assert train_queries.isdisjoint(test_queries)


def _make_result(
    query: str,
    entities: list[str],
    candidate_gids: list[int],
    entity_buckets: dict[str, str] | None = None,
) -> dict:
    """A search response where each candidate carries population + retriever_score.

    ``entity_buckets`` mirrors the live API response: the NER spans split by type.
    Defaults to putting every span in ``city_entities`` (enough for tests that
    don't care about the split).
    """
    if entity_buckets is None:
        entity_buckets = {
            "city_entities": " ".join(entities),
            "country_entities": "",
            "admin1_entities": "",
        }
    return {
        "query": query,
        "entities": entities,
        "entity_buckets": entity_buckets,
        "results": [
            {"geonameid": gid, "population": gid * 10, "retriever_score": 1.0 / gid}
            for gid in candidate_gids
        ],
    }


def test_build_pairs_labels_gold_positive_and_keeps_every_candidate() -> None:
    descriptions = {gid: f"doc-{gid}" for gid in [1, 2, 3, 4, 5, 6]}
    results = [_make_result("q1", ["q1"], candidate_gids=[1, 2, 3, 4, 5, 6])]
    query_dataset = pl.DataFrame({"query": ["q1"], "geonameid": [[1]]})

    pairs = build_pairs(results, query_dataset, descriptions)

    # One gold positive (gid 1); every retrieved candidate becomes a row.
    assert pairs["label"].sum() == 1
    assert pairs.height == 6
    positive = pairs.filter(pl.col("label") == 1).row(0, named=True)
    # gid 1 -> population 10 -> log1p(10); rank 0 (top of the retrieved list).
    assert positive["log_population"] == math.log1p(10)
    assert positive["retriever_rank"] == 0.0


def test_build_pairs_text_features_are_the_typed_entity_buckets() -> None:
    descriptions = {1: "doc-1"}
    results = [
        _make_result(
            "поездка в Казань, Россия зимой",
            ["Казань", "Россия"],
            candidate_gids=[1],
            entity_buckets={
                "city_entities": "Казань",
                "country_entities": "Россия",
                "admin1_entities": "",
            },
        )
    ]
    query_dataset = pl.DataFrame(
        {"query": ["поездка в Казань, Россия зимой"], "geonameid": [[1]]}
    )

    pairs = build_pairs(results, query_dataset, descriptions)

    # The text features are the NER spans split by type (not the raw query) —
    # matching exactly what the engine feeds the reranker online.
    row = pairs.row(0, named=True)
    assert row["city_entities"] == "Казань"
    assert row["country_entities"] == "Россия"
    assert row["admin1_entities"] == ""


def test_build_pairs_uses_gold_entities_only_when_flagged() -> None:
    descriptions = {1: "doc-1"}
    # NER misread the city ("Казан") and missed the country entirely; the dataset's
    # gold spans have both right.
    results = [
        _make_result(
            "поездка в Казань, Россия зимой",
            ["Казан"],
            candidate_gids=[1],
            entity_buckets={
                "city_entities": "Казан",
                "country_entities": "",
                "admin1_entities": "",
            },
        )
    ]
    query_dataset = pl.DataFrame(
        {
            "query": ["поездка в Казань, Россия зимой"],
            "geonameid": [[1]],
            "gold_city_entities": ["Казань"],
            "gold_country_entities": ["Россия"],
            "gold_admin1_entities": [""],
        }
    )

    gold_row = build_pairs(
        results, query_dataset, descriptions, use_gold_entities=True
    ).row(0, named=True)
    assert gold_row["city_entities"] == "Казань"
    assert gold_row["country_entities"] == "Россия"

    # The default must stay NER even when the gold columns are right there —
    # train/serve parity is the production path.
    ner_row = build_pairs(results, query_dataset, descriptions).row(0, named=True)
    assert ner_row["city_entities"] == "Казан"
    assert ner_row["country_entities"] == ""


def test_build_pairs_raises_when_gold_columns_are_missing() -> None:
    descriptions = {1: "doc-1"}
    results = [_make_result("q1", ["q1"], candidate_gids=[1])]
    # A query dataset generated before the gold buckets existed.
    query_dataset = pl.DataFrame({"query": ["q1"], "geonameid": [[1]]})

    with pytest.raises(RuntimeError, match="gold_city_entities"):
        build_pairs(results, query_dataset, descriptions, use_gold_entities=True)


def test_build_pairs_skips_candidates_without_a_description() -> None:
    descriptions = {1: "doc-1", 2: "doc-2"}
    results = [_make_result("q1", ["q1"], candidate_gids=[1, 2, 999])]
    query_dataset = pl.DataFrame({"query": ["q1"], "geonameid": [[1]]})

    pairs = build_pairs(results, query_dataset, descriptions)

    # 999 has no description, so only gids 1 and 2 survive.
    assert pairs.height == 2
