"""Build the sample plan for the synthetic query dataset.

The whole thing is a Polars pipeline:

1. Load every ``(geonameid, country, admin1, population, name, language)`` row,
   where ``name`` is flattened from the canonical name, the ASCII name, and
   every in-scope alternate name (see :func:`load_name_rows`).
2. Group by ``(name, language)`` into a *cities* frame — one row per distinct
   spelling in one language, carrying the geonameids that share it plus homonym
   metadata (:func:`build_cities`).
3. For each ``(language, country)`` pair, take a population-stratified sample of
   names (:func:`sample_stratified`) and assign each a query style and topic
   (:func:`build_sample_plan`).

Every random choice goes through an explicit ``random.Random`` / Polars ``seed``
so a fixed ``cfg.seed`` fully reproduces the plan.
"""

from __future__ import annotations

import random
from pathlib import Path

import polars as pl
from sqlalchemy import literal, select, union
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.config import DatasetConfig, Settings
from src.db.models import AlternateName, Geoname

_ADMIN1_FILE = "admin1CodesASCII.txt"

CITY_ROW_SCHEMA = [
    "geonameid",
    "country_code",
    "admin1_code",
    "population",
    "name",
    "isolanguage",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_admin1_names(data_dir: str, countries: list[str]) -> pl.DataFrame:
    """Read ``admin1CodesASCII.txt`` into a ``(country, admin1, name)`` frame.

    The file is keyed by ``"<CC>.<admin1>"`` (e.g. ``"RU.48" -> "Moscow"``); we
    split that back into its two columns so it joins onto the geoname rows.
    Returns an empty frame if the file is absent, so region names degrade to "".
    """
    schema = ["country_code", "admin1_code", "admin1_name"]
    path = Path(data_dir) / _ADMIN1_FILE
    if not path.exists():
        return pl.DataFrame(schema={c: pl.String for c in schema})

    wanted = set(countries)
    rows: list[tuple[str, str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) < 2 or "." not in parts[0]:
            continue
        country_code, admin1_code = parts[0].split(".", 1)
        if country_code in wanted:
            rows.append((country_code, admin1_code, parts[1]))
    return pl.DataFrame(rows, schema=schema, orient="row")


async def load_name_rows(
    session_factory: async_sessionmaker, settings: Settings
) -> pl.DataFrame:
    """Load one row per name spelling for every in-scope populated place.

    The three name sources are unioned into a single ``name``/``isolanguage``
    shape: the canonical ``name`` and the ``asciiname`` are tagged ``'en'``
    (that is how GeoNames romanises them), and every alternate name keeps its
    own language code. Returns a Polars frame with :data:`CITY_ROW_SCHEMA`.
    """
    base = [
        Geoname.geonameid,
        Geoname.country_code,
        Geoname.admin1_code,
        Geoname.population,
    ]
    geoname_filter = [
        Geoname.country_code.in_(settings.countries),
        Geoname.feature_code.not_in(settings.excluded_feature_codes),
    ]

    canonical = select(
        *base, Geoname.name.label("name"), literal("en").label("isolanguage")
    ).where(*geoname_filter)
    ascii_ = select(
        *base, Geoname.asciiname.label("name"), literal("en").label("isolanguage")
    ).where(*geoname_filter)
    alternates = (
        select(
            *base,
            AlternateName.alternate_name.label("name"),
            AlternateName.isolanguage,
        )
        .outerjoin(AlternateName, Geoname.geonameid == AlternateName.geonameid)
        .where(*geoname_filter, AlternateName.isolanguage.in_(settings.languages))
    )

    async with session_factory() as session:
        rows = (await session.execute(union(canonical, ascii_, alternates))).all()

    return pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row")


# ---------------------------------------------------------------------------
# Cities frame + homonyms
# ---------------------------------------------------------------------------


def build_cities(name_rows: pl.DataFrame, admin1_names: pl.DataFrame) -> pl.DataFrame:
    """Collapse name rows into one row per ``(name, language)`` spelling.

    Each row aggregates the geonameids that share the spelling, their countries
    and admin1 regions (both raw-aligned and de-duplicated), the region names,
    and ``max_population_log1p`` used to rank/stratify. ``country_codes_unique``
    / ``admin1_codes_unique`` sizes are what the homonym filters key on.
    """
    return (
        name_rows.join(admin1_names, on=["country_code", "admin1_code"], how="left")
        .with_columns(admin1_name=pl.col("admin1_name").fill_null(""))
        .group_by("name", "isolanguage")
        .agg(
            pl.col("geonameid"),
            pl.col("country_code").alias("country_codes"),
            pl.col("country_code").unique().alias("country_codes_unique"),
            pl.col("admin1_code").alias("admin1_codes"),
            pl.col("admin1_code").unique().alias("admin1_codes_unique"),
            pl.col("admin1_name").alias("admin1_names"),
            pl.max("population").log1p().alias("max_population_log1p"),
            pl.col("population").alias("populations"),
        )
    )


def country_level_homonyms(cities: pl.DataFrame) -> pl.DataFrame:
    """Names shared across two or more countries (e.g. Moscow US vs RU)."""
    return cities.filter(pl.col("country_codes_unique").list.len() > 1)


def admin1_level_homonyms(cities: pl.DataFrame) -> pl.DataFrame:
    """Names shared across regions of a single country (e.g. two Rostovs in RU)."""
    return cities.filter(
        (pl.col("admin1_codes_unique").list.len() > 1)
        & (pl.col("country_codes_unique").list.len() == 1)
    )


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def sample_stratified(
    df: pl.DataFrame, cfg: DatasetConfig, source: str
) -> pl.DataFrame:
    """Stratified sample: most-populous, random middle, least-populous.

    Taking the top ``cfg.n_top`` names, a random ``cfg.n_mid`` from the middle,
    and the bottom ``cfg.n_low`` guarantees coverage of both
    big cities and tiny villages instead of drowning in the long
    tail. Small frames collapse gracefully (the slices overlap), so a final
    de-dupe keeps rows distinct.
    """
    top, mid, low = cfg.n_top, cfg.n_mid, cfg.n_low
    middle = df.slice(top, max(df.height - top - low, 0))
    picked = pl.concat(
        [
            df.head(top),
            middle.sample(n=min(mid, middle.height), seed=cfg.seed),
            df.tail(low),
        ]
    )
    return picked.unique(
        subset=["name", "isolanguage"], maintain_order=True
    ).with_columns(sample_source=pl.lit(source))


def build_sample_plan(
    cities: pl.DataFrame, settings: Settings, cfg: DatasetConfig, rng: random.Random
) -> pl.DataFrame:
    """Assemble the full per-request plan across every language and country.

    For each ``(language, country)`` we stratified-sample the names occurring in
    that country, then attach a random query ``style`` and ``topic`` to every
    row. The returned frame is the input to generation: one row == one LLM call,
    keyed by a stable ``request_id`` the checkpoint uses for warm-start.

    ponytail: only the ``one_city`` source is wired up — the notebook's
    homonym sources are still experimental. :func:`country_level_homonyms` /
    :func:`admin1_level_homonyms` already expose the frames, so adding them is
    another ``sample_stratified(..., "homonym_*")`` append here.
    """
    parts: list[pl.DataFrame] = []
    for language in settings.languages:
        for country in settings.countries:
            in_country = cities.filter(
                pl.lit(country).is_in(pl.col("country_codes"))
                & (pl.col("isolanguage") == language)
            )
            if in_country.height:
                country_max_population = []
                for row in in_country.iter_rows(named=True):
                    max_population = -1
                    for cc, population in zip(row["country_codes"], row["populations"]):
                        if cc == country and max_population < population:
                            max_population = population
                    country_max_population.append(max_population)
                in_country = (
                    in_country
                    .with_columns(
                        max_population_log1p=pl.Series(country_max_population).log1p()
                    )
                    .sort("max_population_log1p", descending=True)
                )
                parts.append(sample_stratified(in_country, cfg, "one_city"))

    plan = pl.concat(parts)
    n = plan.height
    styles = list(cfg.style_weights.keys())
    return plan.with_columns(
        style=pl.Series(
            rng.choices(styles, weights=list(cfg.style_weights.values()), k=n)
        ),
        topic=pl.Series(rng.choices(cfg.topics, k=n)),
    ).with_row_index("request_id")
