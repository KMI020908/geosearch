"""Lookup builders and pairing utilities for query generation.

Each lookup builder aggregates city geoname_ids from an exploded DataFrame at a
particular grouping level, returning a dictionary for O(1) lookups.

The four levels mirror the disambiguation granularity:

- **bare**          — (language, name) → all homonyms worldwide
- **country**       — (language, name, country_code) → homonyms within a country
- **state**         — (language, name, admin1_code) → homonyms within a region
- **state_country** — (language, name, admin1_code, country_code) → full precision
"""
import random

import polars as pl

BareLookup = dict[tuple[str, str], list[int]]
CountryLookup = dict[tuple[str, str, str], list[int]]
StateLookup = dict[tuple[str, str, str], list[int]]
StateCountryLookup = dict[tuple[str, str, str, str], list[int]]


def build_bare_lookup(exploded: pl.DataFrame) -> BareLookup:
    """Map (language, name) → all geoname_ids (no country/state filter)."""
    lookup: BareLookup = {}
    for lang, name, ids in (
        exploded
        .group_by("language", "name")
        .agg(pl.col("geoname_id").alias("geoname_ids"))
    ).iter_rows():
        lookup[(lang, name)] = ids
    return lookup


def build_country_lookup(exploded: pl.DataFrame) -> CountryLookup:
    """Map (language, name, country_code) → geoname_ids."""
    lookup: CountryLookup = {}
    for lang, name, cc, ids in (
        exploded
        .group_by("language", "name", "country_code")
        .agg(pl.col("geoname_id").alias("geoname_ids"))
    ).iter_rows():
        lookup[(lang, name, cc)] = ids
    return lookup


def build_state_lookup(exploded: pl.DataFrame) -> StateLookup:
    """Map (language, name, admin1_code) → geoname_ids (across all countries)."""
    lookup: StateLookup = {}
    for lang, name, a1, ids in (
        exploded
        .group_by("language", "name", "admin1_code")
        .agg(pl.col("geoname_id").alias("geoname_ids"))
    ).iter_rows():
        lookup[(lang, name, a1)] = ids
    return lookup


def build_state_country_lookup(exploded: pl.DataFrame) -> StateCountryLookup:
    """Map (language, name, admin1_code, country_code) → geoname_ids."""
    lookup: StateCountryLookup = {}
    for lang, name, a1, cc, ids in (
        exploded
        .group_by("language", "name", "admin1_code", "country_code")
        .agg(pl.col("geoname_id").alias("geoname_ids"))
    ).iter_rows():
        lookup[(lang, name, a1, cc)] = ids
    return lookup


def make_pairs(
    rows_a: list,
    rows_b: list,
    seed_a: int = 42,
    seed_b: int = 43,
    max_pairs: int | None = None,
) -> list[tuple]:
    """Build the Cartesian product of two row lists after deterministic shuffling.

    Each list is shuffled with its own seed, then every combination
    ``(a, b)`` is produced (total = ``len(rows_a) × len(rows_b)``).
    For symmetric cases pass the same list twice with different seeds.

    *max_pairs* limits how many pairs are returned (``None`` = all).
    """
    if not rows_a or not rows_b:
        return []
    sa = list(rows_a)
    sb = list(rows_b)
    random.Random(seed_a).shuffle(sa)
    random.Random(seed_b).shuffle(sb)
    if max_pairs is None:
        return [(a, b) for a in sa for b in sb]
    pairs: list[tuple] = []
    for a in sa:
        for b in sb:
            if a != b:
                pairs.append((a, b))
            if len(pairs) >= max_pairs:
                return pairs
    return pairs
