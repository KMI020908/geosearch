"""Parse the raw GeoNames dumps into the staging corpus (:mod:`src.corpus`).

Synchronous and in-memory, both on purpose. The whole corpus is ~1.3M places and
1.2M name variants — about 58 MB once loaded — so batching it through a database
in chunks of 100 was solving a problem this data does not have. Parsing and
writing is one pass over the zips and two Parquet writes.

The upsert semantics the previous ``INSERT ... ON CONFLICT`` provided are kept:
re-running over new dumps replaces a place by ``geonameid`` rather than
duplicating it (:func:`src.corpus.upsert`), so ``make etl`` stays idempotent and
adding a country does not require a wipe.

Run as::

    python -m src.etl.loader
"""

import logging
from pathlib import Path

import polars as pl

from src import corpus
from src.config import Settings, settings
from src.etl.parser import parse_alternate_names, parse_country_file

logger = logging.getLogger(__name__)


def read_country(country_code: str, data_dir: Path) -> pl.DataFrame:
    """Parse one country's zip into a frame of populated places."""
    rows = [row.model_dump() for row in parse_country_file(country_code, data_dir)]
    if not rows:
        return pl.DataFrame(schema=corpus.GEONAME_SCHEMA)
    frame = pl.DataFrame(rows, schema_overrides=corpus.GEONAME_SCHEMA)
    logger.info("  %s: %d places", country_code, frame.height)
    return frame


def read_alternate_names(
    data_dir: Path, geoname_ids: set[int], languages: list[str]
) -> pl.DataFrame:
    """Parse the alternate-name dump, keeping in-scope places and languages."""
    rows = [
        row.model_dump()
        for row in parse_alternate_names(data_dir, geoname_ids, languages)
    ]
    if not rows:
        return pl.DataFrame(schema=corpus.ALTNAME_SCHEMA)
    frame = pl.DataFrame(rows, schema_overrides=corpus.ALTNAME_SCHEMA)
    logger.info("  alternate names: %d rows", frame.height)
    return frame


def load_all(settings: Settings) -> None:
    """Parse every configured country plus its name variants, and write both."""
    data_dir = Path(settings.geonames_data_dir)

    logger.info("Parsing place records…")
    frames = [read_country(cc, data_dir) for cc in settings.countries]
    geonames = pl.concat(frames, how="vertical").unique(
        subset=["geonameid"], keep="last"
    )

    logger.info("Parsing alternate names…")
    alternate_names = read_alternate_names(
        data_dir, set(geonames["geonameid"].to_list()), settings.languages
    ).unique(subset=["alternate_name_id"], keep="last")

    # Merge into whatever is already staged, so adding a country to the config
    # and re-running does not drop the countries loaded before it.
    try:
        geonames = corpus.upsert(
            corpus.load_geonames(settings), geonames, key="geonameid"
        )
        alternate_names = corpus.upsert(
            corpus.load_alternate_names(settings),
            alternate_names,
            key="alternate_name_id",
        )
    except corpus.CorpusNotBuiltError:
        pass  # first run — nothing to merge into

    corpus.write(geonames, alternate_names, settings)
    logger.info("Next: `make artifacts` to compile this into what serving reads.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    load_all(settings)


if __name__ == "__main__":
    main()
