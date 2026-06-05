import polars as pl
from ..config import PROCESSED_DIR


_CITY_KEY_COLUMNS: list[str] = [
    "geoname_id",
    "country_code",
    "admin1_code",
]


def load_cities(
    languages: list[str],
    countries: list[str] | None = None,
) -> pl.DataFrame:
    """Load city parquet files and re-nest into the engine-expected format.

    The saved parquet files store one row per (city, language) with flat
    columns ``iso_language``, ``names`` (list[str]), ``is_preferred``
    (list[int8]).  Downstream consumers (query engine, description
    generator) expect one row per city with a single ``names`` column
    containing ``list[struct{language, names, is_preferred}]``.

    Args:
        languages: Language codes to include (e.g. ``["en", "ru"]``).
        countries: Country codes to load (e.g. ``["RU", "US"]``).
                   ``None`` loads all available countries.

    Returns:
        Polars DataFrame with columns ``geoname_id``, ``country_code``,
        ``admin1_code``, and ``names`` (nested struct list).
    """
    scan = pl.scan_parquet(PROCESSED_DIR / "cities.parquet").explode("info").unnest("info").explode("names")
    filters = pl.col("language").is_in(languages)
    if countries is not None:
        filters = filters & pl.col("country_code").is_in(countries)
    df = (
        scan
        .filter(filters)
        .select(
            "geoname_id", "country_code", "admin1_code",
            "language", "names", "is_preferred",
        )
        .collect()
    )

    # Re-aggregate into the nested struct format the engine expects.
    return df.group_by(_CITY_KEY_COLUMNS).agg(
        pl.struct(
            pl.col("language").alias("language"),
            pl.col("names"),
            pl.col("is_preferred"),
        ).alias("names")
    )
