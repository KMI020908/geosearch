"""The parsed GeoNames corpus, as two Parquet tables.

This is the staging layer between raw GeoNames dumps and the serving artifacts:

    GeoNames TSV  ->  [this]  ->  make artifacts  ->  Hub

It replaces a PostgreSQL instance that was doing nothing a columnar file could
not. Nothing here needs transactions, concurrent writers, referential
enforcement at write time, or a query more complex than a filter and a
group-by — there is one writer, one reader, and one upstream source. The whole
corpus is ~58 MB in memory, so "stream it from a server" was solving a problem
that does not exist at this size.

**These files are intermediate and are not committed.** They are reproducible
from the raw dumps by ``make etl``, and what gets published is the *compiled*
artifact built from them (:mod:`src.search.artifacts`), not the corpus itself.

Deletes are the one place the old foreign key is missed: ``alternate_name`` had
``ondelete=CASCADE``, so removing a place removed its name variants for free.
:func:`apply_deletes` does that explicitly with an anti-join — the same
guarantee, written down instead of delegated.
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from src.config import Settings

logger = logging.getLogger(__name__)

GEONAMES_FILE = "geonames.parquet"
ALTNAMES_FILE = "alternate_names.parquet"
CORPUS_FILES = (GEONAMES_FILE, ALTNAMES_FILE)

# Mirrors parser.GeonameRow. `modification_date` and the admin2/timezone columns
# ride along unused by serving: they are cheap, and dropping upstream fields at
# the staging layer means re-running ETL to get them back.
GEONAME_SCHEMA: dict[str, pl.DataType] = {
    "geonameid": pl.Int64(),
    "name": pl.String(),
    "asciiname": pl.String(),
    "feature_class": pl.String(),
    "feature_code": pl.String(),
    "country_code": pl.String(),
    "admin1_code": pl.String(),
    "admin2_code": pl.String(),
    "population": pl.Int64(),
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
    "timezone": pl.String(),
    "modification_date": pl.Date(),
}

# Mirrors parser.AlternateNameRow.
ALTNAME_SCHEMA: dict[str, pl.DataType] = {
    "alternate_name_id": pl.Int64(),
    "geonameid": pl.Int64(),
    "isolanguage": pl.String(),
    "alternate_name": pl.String(),
    "is_preferred_name": pl.Boolean(),
    "is_short_name": pl.Boolean(),
    "is_colloquial": pl.Boolean(),
    "is_historic": pl.Boolean(),
}


class CorpusNotBuiltError(RuntimeError):
    """The staging corpus is missing — `make etl` has not been run here."""


def paths(settings: Settings) -> tuple[Path, Path]:
    """Return the (geonames, alternate_names) Parquet paths."""
    directory = Path(settings.corpus_dir)
    return directory / GEONAMES_FILE, directory / ALTNAMES_FILE


def _read(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not path.exists():
        raise CorpusNotBuiltError(
            f"{path} is missing. Run `make etl` to download and parse GeoNames, "
            "or `make hub-pull` if you only want to serve prebuilt artifacts."
        )
    return pl.read_parquet(path).cast(schema)  # pyright: ignore[reportArgumentType]


def load_geonames(settings: Settings) -> pl.DataFrame:
    """Every parsed populated place, unfiltered."""
    return _read(paths(settings)[0], GEONAME_SCHEMA)


def load_alternate_names(settings: Settings) -> pl.DataFrame:
    """Every parsed alternate name, unfiltered."""
    return _read(paths(settings)[1], ALTNAME_SCHEMA)


def write(
    geonames: pl.DataFrame, alternate_names: pl.DataFrame, settings: Settings
) -> tuple[Path, Path]:
    """Replace the corpus with these two frames."""
    geo_path, alt_path = paths(settings)
    geo_path.parent.mkdir(parents=True, exist_ok=True)
    geonames.write_parquet(geo_path, compression="zstd")
    alternate_names.write_parquet(alt_path, compression="zstd")
    logger.info(
        "Corpus written: %d places, %d alternate names -> %s",
        geonames.height,
        alternate_names.height,
        geo_path.parent,
    )
    return geo_path, alt_path


# ---------------------------------------------------------------------------
# Mutation — the operations the daily deltas need
# ---------------------------------------------------------------------------


def upsert(existing: pl.DataFrame, incoming: pl.DataFrame, key: str) -> pl.DataFrame:
    """Merge *incoming* into *existing*, incoming winning on conflict.

    The Polars equivalent of ``INSERT ... ON CONFLICT DO UPDATE``. Order is
    load-bearing: `incoming` is concatenated *last* and `keep="last"` then makes
    it win, which is what makes this an upsert rather than an "ignore
    duplicates".
    """
    if incoming.is_empty():
        return existing
    return (
        pl.concat([existing, incoming.select(existing.columns)], how="vertical")
        .unique(subset=[key], keep="last")
        .sort(key)
    )


def apply_deletes(frame: pl.DataFrame, ids: list[int], key: str) -> pl.DataFrame:
    """Drop rows whose *key* is in *ids*."""
    if not ids:
        return frame
    return frame.filter(~pl.col(key).is_in(ids))


def cascade_orphans(
    alternate_names: pl.DataFrame, geonames: pl.DataFrame
) -> pl.DataFrame:
    """Drop alternate names whose place no longer exists.

    Replaces the ``ondelete=CASCADE`` the foreign key used to provide. Called
    after a geoname delete, so a removed place cannot leave its name variants
    behind to be retrieved and then fail to hydrate.
    """
    return alternate_names.join(
        geonames.select("geonameid"), on="geonameid", how="semi"
    )
