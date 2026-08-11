"""Unit tests for entity bucketing — no DB, no live API, no trained model.

The reranker itself is a cross-encoder scoring raw text pairs, so there is no
hand-computed matching engine left to test here (see :mod:`src.rerank.model`);
what remains is the plumbing that groups NER spans by type for display.
"""

from src.rerank.features import LABEL_BUCKETS, bucket_entities


def test_bucket_entities_groups_by_label():
    spans = [("Казань", "CITY"), ("Россия", "COUNTRY"), ("Татарстан", "REGION")]
    buckets = bucket_entities(spans)
    assert buckets["city_entities"] == "Казань"
    assert buckets["country_entities"] == "Россия"
    assert buckets["admin1_entities"] == "Татарстан"


def test_bucket_entities_joins_multiple_spans_in_occurrence_order():
    spans = [("Moscow", "CITY"), ("Kazan", "CITY")]
    assert bucket_entities(spans)["city_entities"] == "Moscow Kazan"


def test_bucket_entities_region_and_state_share_the_admin1_bucket():
    assert bucket_entities([("Texas", "STATE")])["admin1_entities"] == "Texas"
    assert bucket_entities([("Bavaria", "REGION")])["admin1_entities"] == "Bavaria"


def test_bucket_entities_every_bucket_is_always_present():
    buckets = bucket_entities([])
    assert set(buckets) == set(LABEL_BUCKETS)
    assert all(v == "" for v in buckets.values())


def test_bucket_entities_ignores_unknown_labels():
    buckets = bucket_entities([("something", "PERSON")])
    assert all(v == "" for v in buckets.values())
