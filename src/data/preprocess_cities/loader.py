"""Functions for loading and structuring raw GeoNames data into a Polars LazyFrame."""
from pathlib import Path

import polars as pl
from tqdm import tqdm

from ..config import (
    ALTERNATE_NAMES_SCHEMA,
    CITIES_SCHEMA,
    CITY_COLUMNS,
    FEATURE_CODE_MAPPING,
)

# Matches Roman numerals so they can be preserved when checking for Latin chars in Russian names.
_ROMAN_PATTERN = (
    r"\b[Mm]{0,3}(?:[Cc][Mm]|[Cc][Dd]|[Dd]?[Cc]{0,3})"
    r"(?:[Xx][Cc]|[Xx][Ll]|[Ll]?[Xx]{0,3})"
    r"(?:[Ii][Xx]|[Ii][Vv]|[Vv]?[Ii]{0,3})\b"
)


# ── Low-level scanners ────────────────────────────────────────────────────────


def feature_code_df() -> pl.LazyFrame:
    """Lookup table mapping GeoNames feature codes to human-readable place types."""
    return pl.LazyFrame({
        "feature_code": list(FEATURE_CODE_MAPPING.keys()),
        "place_type":   list(FEATURE_CODE_MAPPING.values()),
    })


def scan_cities(file_path: str | Path) -> pl.LazyFrame:
    """Lazily read one country's GeoNames city file, keeping only populated places."""
    return (
        pl.scan_csv(
            file_path,
            separator="\t",
            has_header=False,
            schema=CITIES_SCHEMA,
            quote_char=None,
        )
        .filter((pl.col("population") > 0) & (pl.col("feature_class") == "P"))
    )


def scan_alternate_names(file_path: str | Path) -> pl.LazyFrame:
    """Lazily read alternateNamesV2.txt with properly typed boolean and numeric columns."""
    df = pl.scan_csv(
        file_path,
        separator="\t",
        has_header=False,
        schema=ALTERNATE_NAMES_SCHEMA,
        quote_char=None,
    )
    bool_cols = ["is_preferred_name", "is_short_name", "is_colloquial", "is_historic"]
    df = df.with_columns([
        pl.when(pl.col(col) == "1").then(True).otherwise(False).alias(col)
        for col in bool_cols
    ])
    df = df.with_columns([
        pl.col("alternate_name_id").cast(pl.Int64),
        pl.col("geoname_id").cast(pl.Int64),
        pl.col("from").cast(pl.Int32, strict=False),
        pl.col("to").cast(pl.Int32, strict=False),
    ])
    return df


# ── Higher-level builders ─────────────────────────────────────────────────────


def build_cities_lf(raw_dir: Path, countries: list[str]) -> pl.LazyFrame:
    """Concat city files for the given countries and attach place_type labels."""
    cities = pl.LazyFrame(schema=CITIES_SCHEMA)
    for country in tqdm(countries, desc="Loading countries"):
        cities = pl.concat([cities, scan_cities(raw_dir / "countries" / f"{country}.txt")])
    return cities.join(feature_code_df(), on="feature_code")


def build_names_lf(
    raw_dir: Path,
    cities_lf: pl.LazyFrame,
    languages: list[str],
) -> pl.LazyFrame:
    """
    Build a deduplicated names LazyFrame covering all target languages.

    - Loads alternateNamesV2 filtered to target languages.
    - For Russian: drops names that still contain Latin characters after removing
      Roman numerals (e.g. "Sankt-Peterburg" is excluded, "Санкт-Петербург" stays).
    - Injects ASCII names as English preferred names for every city so every city
      has at least one English entry even if it is absent from alternateNames.
    - Deduplicates by (geoname_id, iso_language, alternate_name), keeping is_preferred=max.
    """
    ascii_names = cities_lf.select(
        pl.col("geoname_id"),
        pl.lit("en").alias("iso_language"),
        pl.col("ascii_name").alias("alternate_name"),
        pl.lit(True).alias("is_preferred_name"),
    )

    alt_names = (
        scan_alternate_names(raw_dir / "alternateNamesV2.txt")
        .filter(
            (pl.col("iso_language").is_in(languages))
            & (
                (pl.col("iso_language") != "ru")
                | (
                    pl.col("alternate_name")
                    .fill_null("")
                    .str.replace_all(_ROMAN_PATTERN, "")
                    .str.count_matches(r"[a-zA-Z]") == 0
                )
            )
        )
        .select("geoname_id", "iso_language", "alternate_name", "is_preferred_name")
    )

    return (
        pl.concat([alt_names, ascii_names])
        .group_by("geoname_id", "iso_language", "alternate_name")
        .agg(pl.max("is_preferred_name"))
        .with_columns(is_preferred_name=pl.col("is_preferred_name").cast(pl.Int8()))
    )


def build_dataset_lf(
    raw_dir: Path,
    countries: list[str],
    languages: list[str],
) -> pl.LazyFrame:
    """
    Build the full structured city dataset as a LazyFrame.

    Each row represents a unique (ascii_name, country_code, admin1_code) combination
    and contains:
    - Geographic fields: latitude, longitude, dem, timezone, population.
    - place_type: deduplicated list of GeoNames feature type labels.
    - info: list of {language, names[], is_preferred[]} structs — one entry per language.

    The double group_by handles duplicate geoname entries for the same logical city
    (e.g. a city appearing under multiple feature codes) by merging them into one record.
    """
    cities = build_cities_lf(raw_dir, countries)
    names = build_names_lf(raw_dir, cities, languages)

    return (
        cities
        .join(names, on="geoname_id", how="left")
        .group_by(*CITY_COLUMNS, "iso_language")
        .agg(
            names=pl.col("alternate_name"),
            is_preferred=pl.col("is_preferred_name"),
        )
        .group_by(*CITY_COLUMNS)
        .agg(
            info=pl.struct(
                language=pl.col("iso_language"),
                names=pl.col("names"),
                is_preferred=pl.col("is_preferred"),
            )
        )
        .group_by("ascii_name", "country_code", "admin1_code")
        .agg(
            pl.max("geoname_id").alias("geoname_id"),
            pl.max("latitude").alias("latitude"),
            pl.max("longitude").alias("longitude"),
            pl.max("population").alias("population"),
            pl.max("dem").alias("dem"),
            pl.max("timezone").alias("timezone"),
            pl.col("place_type").flatten().unique().alias("place_type"),
            pl.col("info"),
        )
    )
