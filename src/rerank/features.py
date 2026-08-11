"""Entity-bucketing shared between the search engine and reranker mining.

The reranker itself (:mod:`src.rerank.model`) is a cross-encoder scoring raw
``(query_text, document)`` text pairs — it attends jointly over both texts, so
unlike the CatBoost model this project used before, there is no bag-of-words
column independence problem to work around with hand-computed overlap
features. What is left here is purely the plumbing that turns GLiNER spans
into the strings both training and serving use: the typed entity buckets
(shown on the API response for display) and the document-spelling separator.
"""

from __future__ import annotations

# Map GLiNER labels (`settings.ner_labels`) onto the entity feature columns.
# GeoNames admin1 is the state/province/region level, so REGION and STATE both
# feed the single `admin1_entities` bucket — matching the candidate document,
# which also carries one admin1 line.
LABEL_BUCKETS: dict[str, set[str]] = {
    "city_entities": {"CITY"},
    "country_entities": {"COUNTRY"},
    "admin1_entities": {"REGION", "STATE"},
}

# The query dataset stores the *reference* spans in the same buckets under a
# `gold_` prefix (:mod:`src.dataset.generate`), so they are byte-comparable to
# the live `entity_buckets`. Derived from `LABEL_BUCKETS` rather than spelled
# out, so adding a bucket does not need a second edit here.
GOLD_BUCKET_COLS: dict[str, str] = {name: f"gold_{name}" for name in LABEL_BUCKETS}

# Separator between a candidate's name spellings on the first line of its
# `document` (:func:`src.rerank.dataset.build_descriptions`). A plain space
# loses the boundary between names — "Little Rock Литл-Рок" is indistinguishable
# from one four-token name.
NAME_SEPARATOR = " | "


def bucket_entities(spans: list[tuple[str, str]]) -> dict[str, str]:
    """Group ``(text, label)`` NER spans into the three entity display strings.

    Spans sharing a bucket are space-joined in occurrence order. Every bucket key
    is always present (empty string when no span of that type was found). This is
    a display field on the API response (`entity_buckets`) — the reranker itself
    is fed the flat, untyped `query_text` (`" ".join(entities)`, the same string
    BM25 retrieval scores against), not these buckets. Labels not in
    `LABEL_BUCKETS` are ignored.
    """
    buckets: dict[str, list[str]] = {name: [] for name in LABEL_BUCKETS}
    for text, label in spans:
        for name, labels in LABEL_BUCKETS.items():
            if label in labels:
                buckets[name].append(text)
                break
    return {name: " ".join(texts) for name, texts in buckets.items()}
