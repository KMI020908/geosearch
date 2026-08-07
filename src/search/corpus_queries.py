"""The three projections the artifact builder needs out of the staging corpus.

These were SQL — a three-way ``UNION``, a filtered ``SELECT``, and the join
feeding ``build_descriptions``. They are the same shape in Polars, and this
module exists so the shape is stated once rather than re-derived at each call
site.

Scope is applied identically in all three (``countries`` ∩ not
``excluded_feature_codes``), because :func:`load_corpus` and :func:`load_places`
must agree on which places exist: a place the index can retrieve but the places
table omits is dropped from results with nothing raised. The artifact builder
asserts that agreement rather than assuming it.
"""

from __future__ import annotations

import polars as pl

from src import corpus
from src.config import Settings

# One BM25 document: a name and every (geonameid, population) it resolves to.
NameGroup = tuple[tuple[int, int], ...]


def _in_scope(geonames: pl.DataFrame, settings: Settings) -> pl.DataFrame:
    return geonames.filter(
        pl.col("country_code").is_in(settings.countries),
        ~pl.col("feature_code").is_in(settings.excluded_feature_codes),
    )


def name_rows(settings: Settings) -> pl.DataFrame:
    """One row per name spelling: canonical, ASCII, and alternate names.

    The canonical ``name`` and ``asciiname`` are tagged ``'en'`` — that is how
    GeoNames romanises them — while alternate names keep their own language
    code. ``asciiname`` also rides along as its own column, because it is the
    place's English form whatever the spelling's language, which is what the
    ``region_repeats_city`` check compares against the English ``admin1_name``.

    Deliberately unordered, like the ``UNION`` it replaces: determinism is a
    property of the pipeline downstream, which sorts and carries explicit
    tiebreaks.
    """
    places = _in_scope(corpus.load_geonames(settings), settings)
    base = ["geonameid", "country_code", "admin1_code", "population", "asciiname"]

    canonical = places.select(
        *base, pl.col("name").alias("name"), pl.lit("en").alias("isolanguage")
    )
    ascii_ = places.select(
        *base, pl.col("asciiname").alias("name"), pl.lit("en").alias("isolanguage")
    )
    alternates = (
        corpus.load_alternate_names(settings)
        .filter(pl.col("isolanguage").is_in(settings.languages))
        .join(places.select(*base), on="geonameid", how="inner")
        .select(*base, pl.col("alternate_name").alias("name"), "isolanguage")
    )
    return pl.concat([canonical, ascii_, alternates], how="vertical").unique()


def bm25_corpus(settings: Settings) -> tuple[list[NameGroup], list[str]]:
    """Group geonameids by name string — one BM25 document per name.

    "Moscow" becomes a single document backed by every place carrying that
    spelling, homonyms included, so they all inherit the same retriever score
    and tie at retrieval time. Population rides along for the tiebreak that
    resolves them later; the BM25 score itself is per query, so nothing is
    stored here.
    """
    rows = (
        name_rows(settings)
        .select("name", "geonameid", "population")
        .unique()
        .group_by("name")
        .agg("geonameid", "population")
    )

    documents: list[str] = []
    groups: list[NameGroup] = []
    for name, ids, populations in rows.iter_rows():
        documents.append(name)
        groups.append(tuple(zip(ids, [p or 0 for p in populations], strict=True)))
    return groups, documents


def places(settings: Settings) -> pl.DataFrame:
    """The hydration table: exactly the fields a search result carries.

    Sorted by ``geonameid`` because the source has no inherent order, and an
    artifact whose bytes differ between runs on row order alone would defeat the
    checksums in the manifest.
    """
    from src.search.artifacts import PLACES_SCHEMA

    return (
        _in_scope(corpus.load_geonames(settings), settings)
        .select(*PLACES_SCHEMA)
        .sort("geonameid")
        .cast(PLACES_SCHEMA)  # pyright: ignore[reportArgumentType]
    )


def descriptions(settings: Settings) -> pl.DataFrame:
    """The reranker's candidate documents, as a table.

    Delegates to :func:`src.rerank.dataset.build_descriptions` rather than
    reimplementing it, so the served documents are byte-identical to the ones
    the model was trained against — including the ``NAME_SEPARATOR`` join, which
    is load-bearing (see that function).
    """
    from src.dataset.sampling import load_admin1_names
    from src.rerank.dataset import build_descriptions
    from src.search.artifacts import DESCRIPTIONS_SCHEMA

    admin1_names = load_admin1_names(settings.geonames_data_dir, settings.countries)
    built = build_descriptions(name_rows(settings), admin1_names)
    return pl.DataFrame(sorted(built.items()), schema=DESCRIPTIONS_SCHEMA, orient="row")
