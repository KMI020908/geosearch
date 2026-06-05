"""Generic query generation engine.

Replaces the three nearly-identical ``functions.py`` modules that lived
under ``english/queries/``, ``russian/queries/``, and ``turkish/queries/``.
The only per-language variation is the template dispatch, which is handled
by calling methods on a :class:`~src.data.languages.base.LanguagePlugin`.
"""
import polars as pl
import random
from tqdm.auto import tqdm
from typing import Any, Callable, Optional
from country_list import countries_for_language
from src.data.utils import make_region_resolver
from src.data.languages.base import LanguagePlugin
from src.data.queries.utils import (
    build_bare_lookup,
    build_country_lookup,
    build_state_lookup,
    build_state_country_lookup,
    make_pairs,
)


# -- DataFrame helpers --------------------------------------------------------


def _explode_names(cities: pl.DataFrame) -> pl.DataFrame:
    """Explode the nested ``names`` struct into one row per city/language/name."""
    return (
        cities
        .explode("names")
        .unnest("names")
        .rename({"names": "name"})
    )


# -- Per-template generators (single city) ------------------------------------
# Each accepts pre-built *rows* (flat tuples from a lookup) so that filtering
# and row construction happen once in ``generate_for_language``.


def _single_city(
    rows: list[tuple],
    plugin: LanguagePlugin,
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for single-city templates (city name only)."""
    source = "templates_single_city_queries"
    data: list[tuple[Any, ...]] = []
    for language, name, geoname_ids in tqdm(rows, desc="single_city"):
        queries = plugin.templates_single_city(name)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], geoname_ids, language, source, i))
    return data


def _single_city_country(
    rows: list[tuple],
    plugin: LanguagePlugin,
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for city + country templates."""
    source = "templates_single_city_country_queries"
    data: list[tuple[Any, ...]] = []
    for language, name, country_code, geoname_ids in tqdm(
        rows, desc="single_city_country"
    ):
        country_name = country_by_code.get(str(country_code), country_code)
        queries = plugin.templates_single_city_country(name, country_name)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], geoname_ids, language, source, i))
    return data


def _single_city_state(
    rows: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for city + state/region templates."""
    source = "templates_single_city_state_queries"
    data: list[tuple[Any, ...]] = []
    for language, name, admin1_code, country_code, geoname_ids in tqdm(
        rows, desc="single_city_state"
    ):
        state_name = get_region_name(
            str(country_code), str(admin1_code), str(language)
        )
        if name.lower() == state_name.lower():
            continue
        queries = plugin.templates_single_city_state(name, state_name)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], geoname_ids, language, source, i))
    return data


def _single_city_state_country(
    rows: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for city + state + country templates."""
    source = "templates_single_city_state_country_queries"
    data: list[tuple[Any, ...]] = []
    for language, name, admin1_code, country_code, geoname_ids in tqdm(
        rows, desc="single_city_state_country"
    ):
        state_name = get_region_name(
            str(country_code), str(admin1_code), str(language)
        )
        if name.lower() == state_name.lower():
            continue
        country_name = country_by_code.get(str(country_code), country_code)
        queries = plugin.templates_single_city_state_country(
            name, state_name, country_name
        )
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], geoname_ids, language, source, i))
    return data


# -- Two-city generators (symmetric) -----------------------------------------
# Each accepts pre-built *pairs* (list of (row_a, row_b) tuples).


def _two_cities(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for two-city templates (bare names)."""
    source = "templates_two_cities_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, ids1), (_, name2, ids2) in tqdm(pairs, desc="two_cities"):
        if name1 == name2:
            continue
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities(name1, name2)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


def _two_cities_country(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for two cities, each with country."""
    source = "templates_two_cities_country_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, cc1, ids1), (_, name2, cc2, ids2) in tqdm(
        pairs, desc="two_cities_country"
    ):
        if name1 == name2 and cc1 == cc2:
            continue
        co1 = country_by_code.get(str(cc1), cc1)
        co2 = country_by_code.get(str(cc2), cc2)
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_country(name1, co1, name2, co2)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


def _two_cities_state(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for two cities, each with state."""
    source = "templates_two_cities_state_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, a1, cc1, ids1), (_, name2, a2, cc2, ids2) in tqdm(
        pairs, desc="two_cities_state"
    ):
        st1 = get_region_name(str(cc1), str(a1), str(lang1))
        st2 = get_region_name(str(cc2), str(a2), str(lang1))
        if name1 == name2 and st1 == st2:
            continue
        if name1.lower() == st1.lower() or name2.lower() == st2.lower():
            continue
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_state(name1, st1, name2, st2)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


def _two_cities_state_country(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows for two cities, each with state + country."""
    source = "templates_two_cities_state_country_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, a1, cc1, ids1), (_, name2, a2, cc2, ids2) in tqdm(
        pairs, desc="two_cities_state_country"
    ):
        st1 = get_region_name(str(cc1), str(a1), str(lang1))
        st2 = get_region_name(str(cc2), str(a2), str(lang1))
        if name1 == name2 and st1 == st2 and cc1 == cc2:
            continue
        if name1.lower() == st1.lower() or name2.lower() == st2.lower():
            continue
        co1 = country_by_code.get(str(cc1), cc1)
        co2 = country_by_code.get(str(cc2), cc2)
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_state_country(
            name1, st1, co1, name2, st2, co2
        )
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


# -- Two-city generators (asymmetric) ----------------------------------------
# Pairs are built via ``make_cross_pairs`` from two different row lists so that
# city1 and city2 carry the correct ids for their respective specificity level.


def _two_cities_one_country(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows: city1 with country, city2 bare name."""
    source = "templates_two_cities_one_country_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, cc1, ids1), (_, name2, ids2) in tqdm(
        pairs, desc="two_cities_one_country"
    ):
        co1 = country_by_code.get(str(cc1), cc1)
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_one_country(name1, co1, name2)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


def _two_cities_one_state(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows: city1 with state, city2 bare name."""
    source = "templates_two_cities_one_state_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, a1, cc1, ids1), (_, name2, ids2) in tqdm(
        pairs, desc="two_cities_one_state"
    ):
        st1 = get_region_name(str(cc1), str(a1), str(lang1))
        if name1.lower() == st1.lower():
            continue
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_one_state(name1, st1, name2)
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


def _two_cities_one_state_country(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows: city1 with state + country, city2 bare name."""
    source = "templates_two_cities_one_state_country_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, a1, cc1, ids1), (_, name2, ids2) in tqdm(
        pairs, desc="two_cities_one_state_country"
    ):
        st1 = get_region_name(str(cc1), str(a1), str(lang1))
        if name1.lower() == st1.lower():
            continue
        co1 = country_by_code.get(str(cc1), cc1)
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_one_state_country(
            name1, st1, co1, name2
        )
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


def _two_cities_state_and_country(
    pairs: list[tuple],
    plugin: LanguagePlugin,
    get_region_name: Callable[[str, str, str], str],
    country_by_code: dict[str, str],
    template_sample: int | None,
    rng: Optional[random.Random],
) -> list[tuple[Any, ...]]:
    """Build query rows: city1 with state, city2 with country.

    city1 comes from state_rows (state-level ids),
    city2 comes from country_rows (country-level ids).
    """
    source = "templates_two_cities_state_and_country_queries"
    data: list[tuple[Any, ...]] = []
    for (lang1, name1, a1, cc1, ids1), (_, name2, cc2, ids2) in tqdm(
        pairs, desc="two_cities_state_and_country"
    ):
        st1 = get_region_name(str(cc1), str(a1), str(lang1))

        if name1.lower() == st1.lower():
            continue

        co2 = country_by_code.get(str(cc2), cc2)
        combined_ids = list(set(ids1) | set(ids2))
        queries = plugin.templates_two_cities_state_and_country(
            name1, st1, name2, co2
        )
        indices = list(range(len(queries)))
        if rng is not None and template_sample is not None and template_sample > 0:
            if template_sample < len(indices):
                indices = rng.sample(indices, template_sample)
        for i in indices:
            data.append((queries[i], combined_ids, lang1, source, i))
    return data


# -- Public API ---------------------------------------------------------------


def generate_for_language(
    cities: pl.DataFrame,
    lang_code: str,
    get_region_name: Callable[[str, str, str], str],
    max_pairs: int | None = 10_000,
    template_sample: int | None = None,
    template_seed: int | None = 1337,
) -> list[tuple[Any, ...]]:
    """Generate all query rows for a single language.

    Explodes the cities DataFrame, filters to *lang_code*, builds the four
    geoname_id lookup tables once, then runs all template generators.

    Single-city generators run for **all** cities.  Two-city generators run
    on sampled pairs controlled by *max_pairs* (``None`` = no limit).
    """
    from src.data.languages import get_language

    exploded = _explode_names(cities)
    df_lang = exploded.filter(pl.col("language") == lang_code)
    country_by_code = dict(countries_for_language(lang_code))

    plugin = get_language(lang_code)

    rng: Optional[random.Random]
    if template_sample is not None and template_sample > 0:
        rng = random.Random(template_seed)
    else:
        rng = None

    bare = build_bare_lookup(df_lang)
    country = build_country_lookup(df_lang)
    state = build_state_lookup(df_lang)
    state_country = build_state_country_lookup(df_lang)

    # -- rows (4 types, matching the 4 lookups) --------------------------------
    bare_rows = sorted((*k, v) for k, v in bare.items())
    country_rows = sorted((*k, v) for k, v in country.items())
    state_country_rows = sorted((*k, v) for k, v in state_country.items())
    state_rows = sorted(
        (lang, name, a1, cc, state.get((lang, name, a1), []))
        for lang, name, a1, cc, _ in state_country_rows
    )

    # -- single city: all rows (with optional per-city template sampling) ------
    data: list[tuple[Any, ...]] = []
    data += _single_city(bare_rows, plugin, template_sample, rng)
    data += _single_city_country(
        country_rows, plugin, country_by_code, template_sample, rng
    )
    data += _single_city_state(
        state_rows, plugin, get_region_name, template_sample, rng
    )
    data += _single_city_state_country(
        state_country_rows,
        plugin,
        get_region_name,
        country_by_code,
        template_sample,
        rng,
    )

    # -- two cities: sampled pairs (Cartesian product) --------------------------
    mp = max_pairs
    data += _two_cities(
        make_pairs(bare_rows, bare_rows, 100, 101, mp),
        plugin,
        template_sample,
        rng,
    )
    data += _two_cities_country(
        make_pairs(country_rows, country_rows, 102, 103, mp),
        plugin,
        country_by_code,
        template_sample,
        rng,
    )
    data += _two_cities_state(
        make_pairs(state_rows, state_rows, 104, 105, mp),
        plugin,
        get_region_name,
        template_sample,
        rng,
    )
    data += _two_cities_state_country(
        make_pairs(state_country_rows, state_country_rows, 106, 107, mp),
        plugin,
        get_region_name,
        country_by_code,
        template_sample,
        rng,
    )
    data += _two_cities_one_country(
        make_pairs(country_rows, bare_rows, 108, 109, mp),
        plugin,
        country_by_code,
        template_sample,
        rng,
    )
    data += _two_cities_one_state(
        make_pairs(state_rows, bare_rows, 110, 111, mp),
        plugin,
        get_region_name,
        template_sample,
        rng,
    )
    data += _two_cities_one_state_country(
        make_pairs(state_country_rows, bare_rows, 112, 113, mp),
        plugin,
        get_region_name,
        country_by_code,
        template_sample,
        rng,
    )
    data += _two_cities_state_and_country(
        make_pairs(state_rows, country_rows, 114, 115, mp),
        plugin,
        get_region_name,
        country_by_code,
        template_sample,
        rng,
    )
    return data


def generate(
    cities: pl.DataFrame,
    languages: Optional[list[str]] = None,
    max_pairs: int | None = None,
    template_sample: int | None = None,
    template_seed: int | None = 1337,
) -> list[tuple[Any, ...]]:
    """Generate query rows for all requested languages.

    If *languages* is ``None``, uses every language in the registry.
    *max_pairs* limits how many city pairs each two-city generator uses.
    """
    from src.data.languages import list_languages

    if languages is None:
        languages = list_languages()

    get_region_name = make_region_resolver()

    data: list[tuple[Any, ...]] = []
    for lang_code in languages:
        data += generate_for_language(
            cities,
            lang_code,
            get_region_name,
            max_pairs=max_pairs,
            template_sample=template_sample,
            template_seed=template_seed,
        )
    return data
