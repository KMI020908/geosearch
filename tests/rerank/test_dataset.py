"""Unit tests for reranker dataset building — no DB, no live API needed."""

import polars as pl

from src.config import RerankConfig
from src.rerank.dataset import PAIR_SCHEMA, build_pairs, split

# Positives are unique per query; negatives are drawn from a small shared pool,
# which mirrors real data where a handful of popular places show up as hard
# negatives across many unrelated queries. This shape is what triggered the
# original split() bug: dedup on "query" kept the (always-first) positive row,
# so the sampled test set only ever contained positive-labelled geonameids.
_N_QUERIES = 20
_NEG_POOL = [90001, 90002, 90003, 90004, 90005]


def _synthetic_pairs() -> pl.DataFrame:
    rows: list[tuple] = []
    for i in range(_N_QUERIES):
        query = f"query-{i}"
        pos_gid = 1000 + i
        rows.append((query, f"doc-{pos_gid}", pos_gid, 100, 1.0, 1))
        for neg_gid in _NEG_POOL:
            rows.append((query, f"doc-{neg_gid}", neg_gid, 50, 0.5, 0))
    return pl.DataFrame(rows, schema=PAIR_SCHEMA, orient="row")


def test_split_holds_out_geonameids_not_just_query_gold_rows() -> None:
    pairs = _synthetic_pairs()
    cfg = RerankConfig(test_size=0.5, seed=42)

    train, test = split(pairs, cfg)

    assert test.height > 0
    # The bug produced a test set that was ~all positives (query-deduped rows
    # are always the query's gold row). A correct geonameid-level split pulls
    # in shared negatives too, so the label mean should not be degenerate.
    assert test["label"].mean() < 0.9


def test_split_never_leaks_a_geonameid_across_train_and_test() -> None:
    pairs = _synthetic_pairs()
    cfg = RerankConfig(test_size=0.5, seed=42)

    train, test = split(pairs, cfg)

    # Re-derive geonameid membership from the original frame by rejoining on
    # (query, document, label) since split() drops "geonameid" from its output.
    train_with_gid = pairs.join(train, on=["query", "document", "label"], how="inner")
    test_with_gid = pairs.join(test, on=["query", "document", "label"], how="inner")

    train_gids = set(train_with_gid["geonameid"].to_list())
    test_gids = set(test_with_gid["geonameid"].to_list())
    assert train_gids.isdisjoint(test_gids)


def _make_result(query: str, candidate_gids: list[int]) -> dict:
    """A search response where each candidate carries population + retriever_score."""
    return {
        "query": query,
        "entities": [query],
        "results": [
            {"geonameid": gid, "population": gid * 10, "retriever_score": 1.0 / gid}
            for gid in candidate_gids
        ],
    }


def test_build_pairs_labels_retrieved_gold_positive_and_carries_features() -> None:
    descriptions = {gid: f"doc-{gid}" for gid in [1, 2, 3, 4, 5, 6]}
    results = [_make_result("q1", candidate_gids=[1, 2, 3, 4, 5, 6])]
    query_dataset = pl.DataFrame({"query": ["q1"], "geonameid": [[1]]})

    pairs = build_pairs(results, query_dataset, descriptions)

    positives = pairs.filter(pl.col("label") == 1)
    negatives = pairs.filter(pl.col("label") == 0)
    assert positives["geonameid"].to_list() == [1]
    assert sorted(negatives["geonameid"].to_list()) == [2, 3, 4, 5, 6]
    # Numeric features are mined straight off the candidate.
    pos = positives.row(0, named=True)
    assert pos["population"] == 10
    assert pos["retriever_score"] == 1.0


def test_build_pairs_skips_candidates_without_a_description() -> None:
    descriptions = {1: "doc-1", 2: "doc-2"}
    results = [_make_result("q1", candidate_gids=[1, 2, 999])]
    query_dataset = pl.DataFrame({"query": ["q1"], "geonameid": [[1]]})

    pairs = build_pairs(results, query_dataset, descriptions)

    assert 999 not in pairs["geonameid"].to_list()
