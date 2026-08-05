"""Build the sample plan for the synthetic query dataset.

The plan is one row per intended LLM call. It is assembled bucket by bucket,
where a bucket is a ``"<sample_source>:<pool>"`` key of
:data:`src.config.PLAN_KINDS` — the query kind plus where its names come from.
Every bucket goes through the *same* three steps:

1. **Pool** — the eligible names of one group (:func:`_group_pool`), ranked by
   population.
2. **Quota** — ``n_top`` from the head, ``n_mid`` sampled from the middle,
   ``n_low`` from the tail (:func:`sample_stratified`), so capitals and
   ``population=0`` villages are both represented.
3. **Fan-out** — each picked name expands into its concrete region/country
   targets (:func:`_expand_targets`), capped at ``cfg.n_targets_per_name``.

That uniformity is the point: every kind's row count per group is bounded by its
own quota (``quota.total`` names, so ``quota.total`` to
``quota.total * n_targets_per_name`` rows), rather than one kind having a per-group
budget and the rest inheriting whatever the data happened to yield.

**Groups are keyed per kind.** ``one_city`` / ``multi_city`` / ``city_admin1`` and
every ``:unique`` bucket are keyed by ``(language, country)``. The two
country-level homonym buckets are keyed by **language alone**: their names are
homonyms *across* countries, so iterating countries would draw the same name once
per country it lives in. See :data:`KINDS`.

**Everything is derived, not ordered.** The plan is a pure function of the *set*
of rows the database returns: ``load_name_rows``' ``UNION`` has no ``ORDER BY``,
so every aggregation sorts its lists and every ranking carries explicit tiebreak
keys. A fixed ``cfg.seed`` then reproduces the plan byte for byte, and every
group draws from an independently derived seed.
"""

from __future__ import annotations

import hashlib
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import polars as pl
from country_list import countries_for_language
from sqlalchemy import literal, select, union
from sqlalchemy.ext.asyncio import async_sessionmaker

from src.config import PLAN_KINDS, DatasetConfig, GroupQuota, Settings
from src.db.models import AlternateName, Geoname

_ADMIN1_FILE = "admin1CodesASCII.txt"

CITY_ROW_SCHEMA = [
    "geonameid",
    "country_code",
    "admin1_code",
    "population",
    "asciiname",
    "name",
    "isolanguage",
]

# One row per intended LLM call. Fixed and uniform across every kind — the plan is
# an artefact other steps read, so it carries no builder internals.
PLAN_SCHEMA: dict[str, pl.DataType] = {
    "name": pl.String(),  # display string; comma-joined for multi_city
    "isolanguage": pl.String(),
    "geonameid": pl.List(pl.Int64()),  # gold; narrowed for disambiguation kinds
    "sample_source": pl.String(),
    "pool": pl.String(),  # all | homonym | unique
    "admin1_name": pl.String(),  # "" unless the query names a region
    "country_name": pl.String(),  # "" unless the query names a country
    # Provenance: plan-parquet only, joinable on request_id, never sent to the API.
    "group_language": pl.String(),
    "group_country": pl.String(),  # "" for language-keyed groups
    "strat_band": pl.String(),  # top | mid | low
    "target_country_code": pl.String(),  # "" for one_city / multi_city
    "target_admin1_code": pl.String(),
}

# What makes two plan rows the same request. Rows are de-duplicated on this at the
# end: the same name can legitimately appear as several kinds, but never twice
# within one.
DEDUPE_KEY = ["name", "isolanguage", "sample_source", "admin1_name", "country_name"]

# Band label attached to each picked name, so the plan records which slice of the
# population range a row came from.
_BANDS = ("top", "mid", "low")


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

    ``asciiname`` also rides along as its own column (not just as one of the
    unioned spellings): it is the place's English form whatever the spelling's
    language, which is what the ``region_repeats_city`` flag compares against the
    English ``admin1_name``. It is functionally dependent on ``geonameid``, so
    carrying it does not change the ``union``'s DISTINCT semantics.

    Deliberately unordered — an ``ORDER BY`` over millions of rows would cost
    Postgres time to impose an order the callers do not rely on. Determinism is a
    property of the Polars pipeline downstream (see the module docstring); this
    function is shared with ``src.search.engine`` and ``src.rerank.dataset``, so
    its schema is not free to change.
    """
    base = [
        Geoname.geonameid,
        Geoname.country_code,
        Geoname.admin1_code,
        Geoname.population,
        Geoname.asciiname,
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
# Derived frames — built once, queried per group
# ---------------------------------------------------------------------------


def build_place_groups(
    name_rows: pl.DataFrame, admin1_names: pl.DataFrame
) -> pl.DataFrame:
    """Collapse name rows into one row per ``(name, language, country, admin1)``.

    That tuple *is* an admin1-level disambiguation target: the places sharing one
    spelling inside one region of one country. Aggregating to it once, in Polars,
    is what lets every other frame here be derived rather than regrouped per row.

    ``region_repeats_city`` flags a target whose region is named after the city
    itself (Shanghai in Shanghai) — a degenerate "city + region" query that
    teaches the reranker nothing, so region-naming kinds drop it.

    The region name is matched against the ``asciiname`` *and* the spelling
    itself, because neither alone is enough. ``admin1_name`` is always English
    (``admin1CodesASCII.txt``) while the spelling may be in any language, so
    ``asciiname`` is what catches "Шанхай" in Shanghai; but GeoNames' ASCII form is
    not always the plain name — New York City's is ``"New York City"`` against a
    ``name`` of ``"New York"`` — so the spelling itself is what catches "New York,
    New York".
    """
    matches_region = pl.col("admin1_name").str.strip_chars().str.to_lowercase()
    return (
        name_rows.join(admin1_names, on=["country_code", "admin1_code"], how="left")
        .with_columns(
            admin1_code=pl.col("admin1_code").fill_null(""),
            admin1_name=pl.col("admin1_name").fill_null(""),
        )
        .group_by("name", "isolanguage", "country_code", "admin1_code")
        .agg(
            # Sorted so the gold list is a function of the row *set*, not of the
            # order Postgres happened to return.
            pl.col("geonameid").sort(),
            pl.col("admin1_name").first(),
            pl.max("population").alias("max_population"),
            ascii_repeats_region=pl.col("asciiname")
            .str.strip_chars()
            .str.to_lowercase()
            .eq(matches_region.first())
            .any(),
        )
        .with_columns(
            region_repeats_city=(pl.col("admin1_name").str.strip_chars() != "")
            & (
                pl.col("ascii_repeats_region")
                | pl.col("name").str.strip_chars().str.to_lowercase().eq(matches_region)
            )
        )
        .drop("ascii_repeats_region")
    )


def build_name_gold(name_rows: pl.DataFrame) -> pl.DataFrame:
    """``(name, language) -> every geonameid carrying that spelling``.

    The gold list for ``one_city`` and ``multi_city``: those queries name no
    region or country, so every homonym of the spelling is a correct answer.
    """
    return name_rows.group_by("name", "isolanguage").agg(pl.col("geonameid").sort())


def build_name_pools(place_groups: pl.DataFrame) -> pl.DataFrame:
    """``(name, language, country)`` with the population ranking and pool flags.

    One frame answering every question the bucket driver asks of a name, computed
    once instead of per (language, country) pass:

    * ``pop_in_country`` — max population of this spelling *in this country*, the
      ranking key for country-keyed groups. A homonym is a big city in one country
      and a hamlet in another, so ranking it by a global maximum would put the
      hamlet at the top of its own country's pool.
    * ``pop_anywhere`` — max over all in-scope countries, the ranking key for the
      language-keyed groups, which have no single country to rank within.
    * ``n_admin1_in_country > 1`` — an admin1-level homonym **in this country**.
      Computed per country on purpose: a name that also exists abroad is still a
      perfectly good "two Rostovs in Russia" query, and a global "single country
      only" test would throw it away.
    * ``n_countries > 1`` — a country-level homonym.
    * ``has_valid_region`` — at least one admin1 target whose region is not named
      after the city, i.e. this name can produce a region-naming query at all.
    """
    per_country = place_groups.group_by("name", "isolanguage", "country_code").agg(
        pl.max("max_population").alias("pop_in_country"),
        pl.col("admin1_code").n_unique().alias("n_admin1_in_country"),
        has_valid_region=(~pl.col("region_repeats_city")).any(),
    )
    return per_country.with_columns(
        pop_anywhere=pl.max("pop_in_country").over("name", "isolanguage"),
        n_countries=pl.col("country_code").n_unique().over("name", "isolanguage"),
    )


def build_country_targets(place_groups: pl.DataFrame) -> pl.DataFrame:
    """One country-level target per ``(name, language, country)``.

    A "city + country" query's gold is every place of that name in that country,
    so the admin1 groups of :func:`build_place_groups` are folded up one level.
    """
    return place_groups.group_by("name", "isolanguage", "country_code").agg(
        pl.col("geonameid").list.explode().sort(),
        pl.max("max_population"),
    )


# ---------------------------------------------------------------------------
# Kinds
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class KindSpec:
    """How one ``"<sample_source>:<pool>"`` bucket is drawn and rendered.

    ``sample_source`` is one of the five values :mod:`src.dataset.prompts` has a
    prompt for — a bucket is *not* a new source. Generation groups rows by
    ``sample_source`` to earn DeepSeek's prefix cache
    (:func:`src.dataset.generate.group_order`), and a source absent from
    ``PROMPTS`` would be dropped from generation entirely, so ``pool`` rides along
    as its own column instead.
    """

    sample_source: str
    pool: str  # all | homonym | unique
    group_by_country: bool  # False => one group per language (see module docstring)
    target_level: str  # none | admin1 | country
    names_region: bool  # fills admin1_name, so region-repeats targets are dropped
    names_country: bool  # fills country_name
    diversify_country: bool = False  # one target per country before the cap

    @property
    def key(self) -> str:
        return f"{self.sample_source}:{self.pool}"


def _kinds() -> dict[str, KindSpec]:
    """The bucket table, keyed exactly as :data:`src.config.PLAN_KINDS`."""
    specs = [
        KindSpec("one_city", "all", True, "none", False, False),
        KindSpec("multi_city", "all", True, "none", False, False),
        # admin1-level homonyms live inside one country, so this kind is keyed by
        # (language, country) like one_city.
        KindSpec("city_admin1", "homonym", True, "admin1", True, False),
        KindSpec("city_admin1", "unique", True, "admin1", True, False),
        KindSpec("city_country", "homonym", False, "country", False, True),
        KindSpec("city_country", "unique", True, "country", False, True),
        # `diversify_country`: the whole point of a country-level homonym here is
        # that the same spelling resolves differently per country, so take the most
        # populous valid region *per country* before the cap. Ranking targets by
        # population alone spends both slots on two regions of the same country
        # (measured: en "Shanghai" -> two US regions), which shows the model
        # nothing about the country signal.
        KindSpec("city_admin1_country", "homonym", False, "admin1", True, True, True),
        KindSpec("city_admin1_country", "unique", True, "admin1", True, True),
    ]
    return {spec.key: spec for spec in specs}


KINDS: dict[str, KindSpec] = _kinds()

assert set(KINDS) == set(PLAN_KINDS), (
    f"KINDS and config.PLAN_KINDS disagree: {sorted(set(KINDS) ^ set(PLAN_KINDS))}"
)


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------


def _derive_seed(*parts: object) -> int:
    """A stable seed derived from ``parts``.

    ``blake2b`` rather than the builtin ``hash()``: string hashing is salted per
    process (``PYTHONHASHSEED``), so a seed derived from it would silently differ
    between runs.
    """
    digest = hashlib.blake2b("|".join(str(p) for p in parts).encode(), digest_size=8)
    return int.from_bytes(digest.digest(), "big")


def sample_stratified(
    df: pl.DataFrame, *, n_top: int, n_mid: int, n_low: int, seed: int
) -> pl.DataFrame:
    """Stratified sample of ``df``, which must already be ranked most-populous first.

    Takes the head ``n_top``, a random ``n_mid`` from the middle band and the tail
    ``n_low``, and labels each pick with the band it came from (``strat_band``).
    Small frames collapse gracefully: the slices overlap and the de-dupe keeps the
    row distinct, so a group of two names never yields more than two picks.

    Takes explicit counts and an explicit seed rather than reading the config, so
    every kind reuses it and every group can draw from its own derived seed.
    """
    middle = df.slice(n_top, max(df.height - n_top - n_low, 0))
    bands = [
        df.head(n_top),
        middle.sample(n=min(n_mid, middle.height), seed=seed),
        df.tail(n_low),
    ]
    picked = pl.concat(
        [
            band.with_columns(strat_band=pl.lit(label))
            for band, label in zip(bands, _BANDS, strict=True)
        ]
    )
    return picked.unique(subset=["name", "isolanguage"], maintain_order=True)


def _pool_predicate(spec: KindSpec) -> pl.Expr:
    """The pool membership test for one bucket.

    ``homonym`` tests the axis the kind actually disambiguates along: the regions
    of one country for a country-keyed ``city_admin1``, the countries themselves
    for the two language-keyed kinds.

    ``unique`` requires uniqueness on **both** axes, not just the negation of the
    kind's own — a name that is unique among its country's regions but exists in
    another country is still a homonym, and labelling its row ``pool="unique"``
    would make the column a lie. It also means a ``unique`` name has exactly one
    target, so those buckets hit their quota exactly.
    """
    if spec.pool == "all":
        return pl.lit(True)
    if spec.pool == "unique":
        return (pl.col("n_countries") == 1) & (pl.col("n_admin1_in_country") == 1)
    homonym_axis = (
        pl.col("n_admin1_in_country")
        if spec.target_level == "admin1" and spec.group_by_country
        else pl.col("n_countries")
    )
    return homonym_axis > 1


def _group_pool(
    pools: pl.DataFrame,
    spec: KindSpec,
    language: str,
    country: str,
    taken: set[tuple[str, str]],
) -> pl.DataFrame:
    """The eligible names of one group, ranked most-populous first.

    Applies, in order: the language/country filter, the kind's pool predicate, the
    "can produce at least one usable target" filter, and the ``taken`` exclusion.

    Filtering on ``has_valid_region`` *before* the quota is what stops a name whose
    only region is named after itself from eating a slot and emitting nothing —
    the old fan-out dropped such a name silently and left the group short.

    The ranking key differs by group shape: ``pop_in_country`` when the group is
    one country, ``pop_anywhere`` when it spans a language. Tiebreak keys are
    explicit because ties are the common case (every ``population=0`` village
    shares a rank) and the input row order is not defined.
    """
    frame = pools.filter(pl.col("isolanguage") == language)
    rank = "pop_anywhere"
    if spec.group_by_country:
        frame = frame.filter(pl.col("country_code") == country)
        rank = "pop_in_country"
    else:
        # A language-keyed group ranks names, not (name, country) pairs: the
        # country comes from the fan-out, so collapse to one row per name first.
        # Sorted before the collapse because `keep="any"` would otherwise return
        # whichever country's row Polars happened to hold.
        frame = frame.sort("country_code").unique(
            subset=["name", "isolanguage"], keep="first", maintain_order=True
        )

    frame = frame.filter(_pool_predicate(spec))
    if spec.names_region:
        frame = frame.filter(pl.col("has_valid_region"))
    # Only this language's taken names: the same spelling can be a name in two
    # languages (a transliteration that matches the English form), and those are
    # two different requests.
    already = [name for name, language_ in taken if language_ == language]
    if already:
        frame = frame.filter(
            ~pl.col("name").is_in(pl.Series("taken", already, pl.String).implode())
        )
    return frame.sort(
        [rank, "name", "country_code"], descending=[True, False, False], nulls_last=True
    )


def _expand_targets(
    name_row: dict,
    spec: KindSpec,
    place_groups: pl.DataFrame,
    country_targets: pl.DataFrame,
    cfg: DatasetConfig,
    country: str,
) -> list[dict]:
    """The concrete targets one picked name fans out into, most-populous first.

    Each target carries the *narrowed* gold — the geonameids in that region or
    country only — which is what turns the name's other homonyms into hard
    negatives for the reranker.

    Region-naming kinds drop region-repeats-city targets *before* the cap, so a
    dropped target does not eat a slot; the next most-populous valid region takes
    its place. ``diversify_country`` then keeps one target per country, so the cap
    is spent on distinct countries rather than on two regions of the same one.

    The sort carries the whole target key, not just the population: a homonym's
    targets are usually tiny villages that all tie at ``population=0``, and the
    frame's row order is not defined, so population alone leaves the pick to
    chance — the same plan would name a different region on every run.
    """
    admin1_level = spec.target_level == "admin1"
    frame = place_groups if admin1_level else country_targets
    frame = frame.filter(
        (pl.col("name") == name_row["name"])
        & (pl.col("isolanguage") == name_row["isolanguage"])
    )
    if spec.group_by_country:
        frame = frame.filter(pl.col("country_code") == country)
    if spec.names_region:
        frame = frame.filter(~pl.col("region_repeats_city"))
    keys = ["max_population", "country_code"] + (
        ["admin1_code"] if admin1_level else []
    )
    frame = frame.sort(
        keys,
        descending=[True] + [False] * (len(keys) - 1),
        nulls_last=True,
    )
    if spec.diversify_country:
        frame = frame.unique(subset=["country_code"], keep="first", maintain_order=True)
    return frame.head(cfg.n_targets_per_name).to_dicts()


def _plan_row(
    *,
    name: str,
    language: str,
    gold: list[int],
    spec: KindSpec,
    admin1_name: str = "",
    country_name: str = "",
    group_country: str = "",
    strat_band: str,
    target_country_code: str = "",
    target_admin1_code: str = "",
) -> dict:
    """One plan row in :data:`PLAN_SCHEMA` shape."""
    return {
        "name": name,
        "isolanguage": language,
        "geonameid": gold,
        "sample_source": spec.sample_source,
        "pool": spec.pool,
        "admin1_name": admin1_name,
        "country_name": country_name,
        "group_language": language,
        "group_country": group_country,
        "strat_band": strat_band,
        "target_country_code": target_country_code,
        "target_admin1_code": target_admin1_code,
    }


def _build_multi_city_rows(
    picked: pl.DataFrame,
    pool: pl.DataFrame,
    spec: KindSpec,
    gold_by_name: dict[tuple[str, str], list[int]],
    cfg: DatasetConfig,
    rng: random.Random,
    language: str,
    country: str,
) -> list[dict]:
    """One row per picked anchor name: the anchor plus a few more cities.

    Each anchor is joined with ``multi_city_extra_min``..``multi_city_extra_max``
    additional names drawn from the same group, so the row count equals the quota
    exactly and the anchors stay population-stratified. The extras are drawn from
    the group's pool by name (not by list position) so the result does not depend
    on the pool's internal ordering.

    Gold is the union of the named cities' geonameids — the same simple labelling
    as ``one_city``, no disambiguation. A group with nothing to pair the anchor
    with yields no rows rather than raising.
    """
    anchors = set(picked["name"])
    others = [
        (row["name"], row["isolanguage"])
        for row in pool.iter_rows(named=True)
        if row["name"] not in anchors
    ]
    rows: list[dict] = []
    for anchor in picked.iter_rows(named=True):
        want = rng.randint(cfg.multi_city_extra_min, cfg.multi_city_extra_max)
        n_extra = min(want, len(others))
        if n_extra < cfg.multi_city_extra_min:
            continue
        extras = rng.sample(sorted(others), n_extra)
        names = [anchor["name"], *(n for n, _ in extras)]
        gold: list[int] = []
        for key in [(anchor["name"], anchor["isolanguage"]), *extras]:
            gold.extend(g for g in gold_by_name[key] if g not in gold)
        rows.append(
            _plan_row(
                name=", ".join(names),
                language=language,
                gold=sorted(gold),
                spec=spec,
                group_country=country,
                strat_band=anchor["strat_band"],
            )
        )
    return rows


def _bucket_rows(
    key: str,
    quota: GroupQuota,
    frames: dict[str, pl.DataFrame],
    gold_by_name: dict[tuple[str, str], list[int]],
    country_names: dict[str, dict[str, str]],
    settings: Settings,
    cfg: DatasetConfig,
    taken: dict[str, set[tuple[str, str]]],
) -> list[dict]:
    """Draw one bucket's rows across all of its groups."""
    spec = KINDS[key]
    groups = [
        (language, country if spec.group_by_country else "")
        for language in settings.languages
        for country in (settings.countries if spec.group_by_country else [""])
    ]

    rows: list[dict] = []
    for language, country in groups:
        pool = _group_pool(
            frames["pools"], spec, language, country, taken[spec.sample_source]
        )
        if pool.is_empty():
            continue
        seed = _derive_seed(cfg.seed, key, language, country)
        picked = sample_stratified(
            pool,
            n_top=quota.n_top,
            n_mid=quota.n_mid,
            n_low=quota.n_low,
            seed=seed,
        )
        # Scoped per sample_source, not globally: one name may be both a one_city
        # and a city_country query, but must not be drawn twice as either — that
        # would be two byte-identical (and separately billed) LLM calls.
        taken[spec.sample_source] |= {
            (row["name"], row["isolanguage"]) for row in picked.iter_rows(named=True)
        }

        if spec.sample_source == "multi_city":
            rows += _build_multi_city_rows(
                picked,
                pool,
                spec,
                gold_by_name,
                cfg,
                random.Random(seed),
                language,
                country,
            )
            continue

        for name_row in picked.iter_rows(named=True):
            if spec.target_level == "none":
                rows.append(
                    _plan_row(
                        name=name_row["name"],
                        language=language,
                        gold=gold_by_name[(name_row["name"], language)],
                        spec=spec,
                        group_country=country,
                        strat_band=name_row["strat_band"],
                    )
                )
                continue
            for target in _expand_targets(
                name_row,
                spec,
                frames["place_groups"],
                frames["country_targets"],
                cfg,
                country,
            ):
                rows.append(
                    _plan_row(
                        name=name_row["name"],
                        language=language,
                        gold=target["geonameid"],
                        spec=spec,
                        admin1_name=(
                            target["admin1_name"] if spec.names_region else ""
                        ),
                        country_name=(
                            country_names[language].get(
                                target["country_code"], target["country_code"]
                            )
                            or ""
                            if spec.names_country
                            else ""
                        ),
                        group_country=country,
                        strat_band=name_row["strat_band"],
                        target_country_code=target["country_code"],
                        target_admin1_code=target.get("admin1_code", ""),
                    )
                )
    return rows


def attach_style_topic(plan: pl.DataFrame, cfg: DatasetConfig) -> pl.DataFrame:
    """Attach a query ``style`` and ``topic`` to every row.

    One RNG per column, each seeded from ``cfg.seed`` and the column's own name.
    Drawing both from a single RNG makes the second column's stream start at
    offset ``n``, so adding rows of one kind silently shifts every other row's
    topic — and the plan on disk then disagrees with the checkpoint generated
    against the previous draw.

    Both columns are still positional, so changing a quota does reshuffle the
    styles and topics of the rows after the affected bucket. That is a property of
    the whole plan changing, not a leak between columns.
    """
    n = plan.height
    style_rng = random.Random(f"{cfg.seed}|style")
    topic_rng = random.Random(f"{cfg.seed}|topic")
    return plan.with_columns(
        style=pl.Series(
            style_rng.choices(
                list(cfg.style_weights),
                weights=list(cfg.style_weights.values()),
                k=n,
            ),
            dtype=pl.String,
        ),
        topic=pl.Series(topic_rng.choices(cfg.topics, k=n), dtype=pl.String),
    )


def plan_group_counts(plan: pl.DataFrame) -> pl.DataFrame:
    """Rows per ``(sample_source, pool, language, country)`` — the quota check.

    The verification surface for ``--plan-only`` and for the row-count tests: every
    non-empty group should sit between ``quota.total`` and
    ``quota.total * n_targets_per_name``, with equality for the buckets whose names
    have a single target.
    """
    return (
        plan.group_by("sample_source", "pool", "group_language", "group_country")
        .len()
        .sort("sample_source", "pool", "group_language", "group_country")
    )


def build_sample_plan(
    place_groups: pl.DataFrame,
    name_gold: pl.DataFrame,
    settings: Settings,
    cfg: DatasetConfig,
) -> pl.DataFrame:
    """Assemble the full per-request plan: one row per intended LLM call.

    Buckets are drawn in :data:`src.config.PLAN_KINDS` order, which groups each
    ``sample_source``'s rows together — generation re-groups by ``sample_source``
    anyway (:func:`src.dataset.generate.group_order`), but a plan whose row order
    matches is easier to read.

    ``request_id`` is positional, so it is only stable while the plan is. Changing
    a quota renumbers every row after the affected bucket, and
    :func:`src.dataset.generate.load_done_ids` keys the warm start on
    ``(request_id, prompt_version)`` — so a quota change on top of a retained
    checkpoint would attribute finished queries to different plan rows, with a
    different gold and no error. Bump ``PROMPT_VERSION`` when quotas change, or
    start from an empty checkpoint.
    """
    frames = {
        "place_groups": place_groups,
        "pools": build_name_pools(place_groups),
        "country_targets": build_country_targets(place_groups),
    }
    gold_by_name = {
        (row["name"], row["isolanguage"]): row["geonameid"]
        for row in name_gold.iter_rows(named=True)
    }
    country_names = {
        language: dict(countries_for_language(language))
        for language in settings.languages
    }
    taken: dict[str, set[tuple[str, str]]] = defaultdict(set)

    rows: list[dict] = []
    for key in PLAN_KINDS:
        quota = cfg.quotas[key]
        if quota.total == 0:
            continue
        rows += _bucket_rows(
            key, quota, frames, gold_by_name, country_names, settings, cfg, taken
        )

    plan = pl.DataFrame(rows, schema=PLAN_SCHEMA)
    plan = plan.unique(subset=DEDUPE_KEY, keep="first", maintain_order=True)
    return attach_style_topic(plan, cfg).with_row_index("request_id")
