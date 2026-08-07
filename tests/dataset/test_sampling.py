"""Unit tests for the query-dataset sampler — no DB, no network.

Everything here runs the real pipeline (``build_place_groups`` ->
``build_sample_plan``) over a synthetic ``CITY_ROW_SCHEMA`` frame, so the tests
exercise the same code path generation does.

The central contract is :func:`test_rows_per_group_match_the_quota`: every
``"<kind>:<pool>"`` bucket draws ``quota.total`` names from *its own group's*
pool, so its row count per group is bounded by that quota — exactly for the
buckets whose names have a single target, and by ``quota.total *
n_targets_per_name`` for the homonym buckets that fan out.
"""

from types import SimpleNamespace

import polars as pl
import pytest

from src.config import PLAN_KINDS, DatasetConfig, GroupQuota
from src.dataset.generate import _KEEP_FIELDS
from src.dataset.sampling import (
    CITY_ROW_SCHEMA,
    DEDUPE_KEY,
    KINDS,
    PLAN_SCHEMA,
    build_name_gold,
    build_name_pools,
    build_place_groups,
    build_sample_plan,
    plan_group_counts,
    sample_stratified,
)

# (geonameid, country, admin1, population, asciiname, name)
# Every place is spelled in English only; the language axis is exercised
# separately by `test_taken_names_are_scoped_per_language`. The pools this has to
# be generous enough for are, per (en, country) group: >= 4 admin1-level
# homonyms, >= 4 fully-unique names, and >= 4 country-level homonyms overall.
_PLACES = [
    # Country-level homonyms: the same spelling in RU and US.
    (1, "RU", "48", 11_500_000, "Moscow", "Moscow"),
    (2, "US", "ID", 25_000, "Moscow", "Moscow"),
    (3, "RU", "61", 500, "Odessa", "Odessa"),
    (4, "US", "TX", 90_000, "Odessa", "Odessa"),
    # ... and Odessa is *also* in a second RU region, so it is an admin1-level
    # homonym inside RU while living in two countries. The old global "single
    # country only" rule made that combination invisible to city_admin1.
    (5, "RU", "76", 200, "Odessa", "Odessa"),
    (6, "RU", "17", 300, "Berlin", "Berlin"),
    (7, "US", "IA", 1_200, "Berlin", "Berlin"),
    (8, "RU", "35", 100, "Salem", "Salem"),
    (9, "US", "MT", 800, "Salem", "Salem"),
    # admin1-level homonyms inside RU only.
    (10, "RU", "61", 1_100_000, "Rostov-na-Donu", "Rostov"),
    (11, "RU", "76", 31_000, "Rostov", "Rostov"),
    (12, "RU", "48", 900, "Ivanovka", "Ivanovka"),
    (13, "RU", "92", 400, "Ivanovka", "Ivanovka"),
    (14, "RU", "35", 700, "Sosnovka", "Sosnovka"),
    (15, "RU", "66", 200, "Sosnovka", "Sosnovka"),
    (16, "RU", "17", 600, "Aleksandrovka", "Aleksandrovka"),
    (17, "RU", "35", 100, "Aleksandrovka", "Aleksandrovka"),
    # admin1-level homonyms inside US only.
    (18, "US", "IA", 116_000, "Springfield", "Springfield"),
    (19, "US", "MT", 8_000, "Springfield", "Springfield"),
    (20, "US", "CA", 60_000, "Franklin", "Franklin"),
    (21, "US", "TX", 9_000, "Franklin", "Franklin"),
    (22, "US", "IA", 26_000, "Clinton", "Clinton"),
    (23, "US", "MT", 700, "Clinton", "Clinton"),
    # New York sits in the region named after it, and in Iowa: the NY target is
    # the degenerate "New York, New York" one, the Iowa target is usable.
    (24, "US", "NY", 8_000_000, "New York", "New York"),
    (25, "US", "IA", 900, "New York", "New York"),
    # Fully-unique names: one country, one region.
    (26, "RU", "92", 640_000, "Tyumen", "Tyumen"),
    (27, "RU", "35", 12_000, "Suzdal", "Suzdal"),
    (28, "RU", "17", 0, "Malinovka", "Malinovka"),
    (29, "RU", "66", 1_200_000, "Kazan", "Kazan"),
    (30, "RU", "61", 340_000, "Sochi", "Sochi"),
    (31, "US", "CA", 3_900_000, "Los Angeles", "Los Angeles"),
    (32, "US", "TX", 2_300_000, "Houston", "Houston"),
    (33, "US", "MT", 0, "Ovando", "Ovando"),
    (34, "US", "IA", 500, "Elkhart", "Elkhart"),
    (35, "US", "CA", 540_000, "Fresno", "Fresno"),
]

_ADMIN1 = [
    ("RU", "48", "Moscow Oblast"),
    ("RU", "61", "Rostov Oblast"),
    ("RU", "76", "Yaroslavl Oblast"),
    ("RU", "92", "Tyumen Oblast"),
    ("RU", "17", "Bryansk Oblast"),
    ("RU", "35", "Vladimir Oblast"),
    ("RU", "66", "Tatarstan"),
    ("US", "ID", "Idaho"),
    ("US", "CA", "California"),
    ("US", "NY", "New York"),
    ("US", "IA", "Iowa"),
    ("US", "TX", "Texas"),
    ("US", "MT", "Montana"),
]


def _name_rows(extra: list[tuple] | None = None) -> pl.DataFrame:
    """The fixture places in :data:`CITY_ROW_SCHEMA` shape."""
    rows = [
        (gid, cc, admin1, population, ascii_name, name, "en")
        for gid, cc, admin1, population, ascii_name, name in _PLACES
    ]
    return pl.DataFrame(rows + (extra or []), schema=CITY_ROW_SCHEMA, orient="row")


def _admin1_names() -> pl.DataFrame:
    return pl.DataFrame(
        _ADMIN1, schema=["country_code", "admin1_code", "admin1_name"], orient="row"
    )


def _settings(languages: list[str] | None = None) -> SimpleNamespace:
    # The sampler only reads these two fields off Settings.
    return SimpleNamespace(languages=languages or ["en"], countries=["RU", "US"])


def _quotas(**overrides: GroupQuota) -> dict[str, GroupQuota]:
    """Every bucket at one name per band, with per-bucket overrides."""
    quotas = {key: GroupQuota(n_top=1, n_mid=1, n_low=1) for key in PLAN_KINDS}
    quotas.update(overrides)
    return quotas


def _cfg(**overrides) -> DatasetConfig:
    return DatasetConfig(quotas=_quotas(), **overrides)


def _plan(cfg: DatasetConfig | None = None, **kwargs) -> pl.DataFrame:
    name_rows = kwargs.pop("name_rows", None)
    if name_rows is None:
        name_rows = _name_rows()
    settings = kwargs.pop("settings", None) or _settings()
    place_groups = build_place_groups(name_rows, _admin1_names())
    return build_sample_plan(
        place_groups, build_name_gold(name_rows), settings, cfg or _cfg()
    )


# ---------------------------------------------------------------------------
# Derived frames
# ---------------------------------------------------------------------------


def test_place_groups_are_one_row_per_admin1_target():
    """A name in two regions of one country becomes two targets, gold sorted."""
    groups = build_place_groups(_name_rows(), _admin1_names())
    rostov = groups.filter(pl.col("name") == "Rostov").sort("admin1_code")

    assert rostov.height == 2
    assert rostov["admin1_name"].to_list() == ["Rostov Oblast", "Yaroslavl Oblast"]
    assert rostov["geonameid"].to_list() == [[10], [11]]
    assert rostov["max_population"].to_list() == [1_100_000, 31_000]


def test_place_groups_flag_a_region_named_after_the_city():
    """ "New York, New York" is flagged; the same city's Iowa target is not."""
    groups = build_place_groups(_name_rows(), _admin1_names())
    flags = dict(
        groups.filter(pl.col("name") == "New York")
        .select("admin1_code", "region_repeats_city")
        .iter_rows()
    )

    assert flags == {"NY": True, "IA": False}


def test_place_groups_flag_a_repeat_the_ascii_name_hides():
    """GeoNames' ASCII form is not always the plain name.

    New York City's ``asciiname`` is "New York City" against a ``name`` of "New
    York", so matching the region against the ASCII form alone lets "New York, New
    York" through.
    """
    rows = [(1, "US", "NY", 8_000_000, "New York City", "New York", "en")]
    groups = build_place_groups(
        pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row"), _admin1_names()
    )

    assert groups["region_repeats_city"].to_list() == [True]


def test_place_groups_treat_a_missing_region_name_as_unflagged():
    """A region absent from admin1CodesASCII.txt degrades to "", never a repeat."""
    groups = build_place_groups(_name_rows(), _admin1_names().head(0))

    assert (groups["admin1_name"] == "").all()
    assert not groups["region_repeats_city"].any()


def test_admin1_homonym_is_computed_per_country():
    """A name living abroad is still an admin1-level homonym at home.

    "Odessa" is in two RU regions *and* in the US. Requiring the name to exist in
    a single country — as the old ``admin1_level_homonyms`` did — threw that away.
    """
    pools = build_name_pools(build_place_groups(_name_rows(), _admin1_names()))
    odessa = pools.filter(pl.col("name") == "Odessa").sort("country_code")

    assert odessa["country_code"].to_list() == ["RU", "US"]
    assert odessa["n_admin1_in_country"].to_list() == [2, 1]
    assert odessa["n_countries"].to_list() == [2, 2]

    rows = _plan().filter(
        (pl.col("name") == "Odessa")
        & (pl.col("sample_source") == "city_admin1")
        & (pl.col("pool") == "homonym")
    )
    assert rows.height > 0
    assert (rows["target_country_code"] == "RU").all()


def test_pop_in_country_ranks_a_homonym_within_each_country():
    """The ranking key is per country, so a homonym is ranked where it is big."""
    pools = build_name_pools(build_place_groups(_name_rows(), _admin1_names()))
    moscow = pools.filter(pl.col("name") == "Moscow").sort("country_code")

    assert moscow["pop_in_country"].to_list() == [11_500_000, 25_000]
    assert moscow["pop_anywhere"].to_list() == [11_500_000, 11_500_000]


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _ranked(n: int) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "name": [f"n{i:02d}" for i in range(n)],
            "isolanguage": ["en"] * n,
            "pop_in_country": list(range(n, 0, -1)),
        }
    )


def test_sample_stratified_takes_head_middle_and_tail():
    picked = sample_stratified(_ranked(10), n_top=2, n_mid=1, n_low=2, seed=1)

    assert picked.height == 5
    assert picked["name"].to_list()[:2] == ["n00", "n01"]
    assert picked["name"].to_list()[-2:] == ["n08", "n09"]
    assert picked["strat_band"].to_list() == ["top", "top", "mid", "low", "low"]
    assert picked["name"][2] not in {"n00", "n01", "n08", "n09"}


def test_sample_stratified_collapses_on_a_small_pool():
    """Overlapping slices on a tiny frame yield the frame, not duplicates."""
    picked = sample_stratified(_ranked(2), n_top=2, n_mid=2, n_low=2, seed=1)

    assert picked["name"].to_list() == ["n00", "n01"]


def test_sample_stratified_middle_depends_on_the_seed():
    """Each group derives its own seed, so middles are not drawn in lockstep."""
    middles = {
        sample_stratified(_ranked(20), n_top=1, n_mid=1, n_low=1, seed=seed)["name"][1]
        for seed in range(8)
    }

    assert len(middles) > 1


# ---------------------------------------------------------------------------
# The quota contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("key", PLAN_KINDS)
def test_rows_per_group_match_the_quota(key: str):
    """Every non-empty group of every bucket stays inside its own quota.

    Exact for the buckets whose names have a single target (``one_city``,
    ``multi_city``, and every ``:unique`` pool, which is unique on both the
    country and the region axis); bounded by the fan-out cap for the homonym
    pools, where one name can expand into several concrete targets.
    """
    cfg = _cfg()
    quota = cfg.quotas[key]
    spec = KINDS[key]
    counts = plan_group_counts(_plan(cfg)).filter(
        (pl.col("sample_source") == spec.sample_source) & (pl.col("pool") == spec.pool)
    )

    assert counts.height > 0, f"{key} produced no rows at all"
    if spec.pool == "homonym":
        assert (counts["len"] >= quota.total).all()
        assert (counts["len"] <= quota.total * cfg.n_targets_per_name).all()
    else:
        assert (counts["len"] == quota.total).all()

    # A country-keyed bucket has one group per (language, country); a
    # language-keyed one has a single group per language, with no country.
    if spec.group_by_country:
        assert set(counts["group_country"]) == {"RU", "US"}
    else:
        assert set(counts["group_country"]) == {""}


def test_a_zero_quota_drops_its_bucket_entirely():
    cfg = _cfg()
    cfg.quotas["multi_city:all"] = GroupQuota()
    plan = _plan(cfg)

    assert not (plan["sample_source"] == "multi_city").any()


def test_quota_keys_match_the_kind_table():
    assert set(KINDS) == set(PLAN_KINDS)
    assert [spec.key for spec in KINDS.values()] == list(PLAN_KINDS)


def test_unknown_quota_key_is_rejected():
    with pytest.raises(ValueError, match="PLAN_KINDS"):
        DatasetConfig(quotas={"city_admin1:nope": GroupQuota(n_top=1)})


# ---------------------------------------------------------------------------
# Row shape and duplicates
# ---------------------------------------------------------------------------


def test_plan_schema_is_exact():
    """The plan carries the declared columns; nothing generation needs is missing."""
    plan = _plan()

    assert list(plan.columns) == ["request_id", *PLAN_SCHEMA, "style", "topic"]
    assert set(_KEEP_FIELDS) <= set(plan.columns)
    assert plan["request_id"].to_list() == list(range(plan.height))


def test_plan_has_no_duplicate_requests():
    plan = _plan()

    assert plan.select(DEDUPE_KEY).n_unique() == plan.height


def test_one_city_names_are_unique_across_groups():
    """A name in two countries is a one_city query once, not once per country.

    Its gold is every homonym either way, so the second row would be a
    byte-identical — and separately billed — request.
    """
    one_city = _plan().filter(pl.col("sample_source") == "one_city")

    assert one_city.select("name", "isolanguage").n_unique() == one_city.height


def test_language_keyed_kinds_name_each_country_of_a_homonym():
    """One homonym name yields one row per country: "Moscow, Russia" and ", USA"."""
    plan = _plan()
    moscow = plan.filter(
        (pl.col("name") == "Moscow") & (pl.col("sample_source") == "city_country")
    )

    assert set(moscow["target_country_code"]) == {"RU", "US"}
    assert set(moscow["country_name"]) == {"Russia", "United States"}
    # The gold is narrowed per country, which is what makes the other homonym a
    # hard negative rather than another right answer.
    assert sorted(moscow["geonameid"].to_list()) == [[1], [2]]


def test_city_admin1_country_diversifies_by_country():
    """The fully-qualified kind spends its targets on distinct countries.

    Ranking a country-level homonym's admin1 targets by population alone puts two
    regions of the same country at the top, which shows the model nothing about
    the country signal.
    """
    rows = _plan().filter(
        (pl.col("sample_source") == "city_admin1_country")
        & (pl.col("pool") == "homonym")
    )

    for _, group in rows.group_by("name"):
        assert group["target_country_code"].n_unique() == group.height


def test_region_named_after_the_city_never_reaches_a_region_query():
    """No region-naming row repeats the city in its region."""
    plan = _plan()
    region_rows = plan.filter(pl.col("admin1_name") != "")

    assert region_rows.height > 0
    for row in region_rows.iter_rows(named=True):
        assert row["admin1_name"].casefold() != row["name"].casefold()
    # New York can still be a region query — via its Iowa target, not its own.
    new_york = region_rows.filter(pl.col("name") == "New York")
    assert set(new_york["admin1_name"]) <= {"Iowa"}


def test_unique_and_homonym_pools_are_disjoint():
    """A name is never both, so no query kind draws it twice."""
    plan = _plan()
    for source in ("city_admin1", "city_country", "city_admin1_country"):
        rows = plan.filter(pl.col("sample_source") == source)
        by_pool = {pool: set(part["name"]) for (pool,), part in rows.group_by("pool")}
        assert not (by_pool.get("homonym", set()) & by_pool.get("unique", set()))


def test_disambiguation_rows_fill_only_their_own_columns():
    plan = _plan()
    by_source = {
        source: plan.filter(pl.col("sample_source") == source)
        for source in ("one_city", "city_admin1", "city_country", "city_admin1_country")
    }

    assert (by_source["one_city"]["admin1_name"] == "").all()
    assert (by_source["one_city"]["country_name"] == "").all()
    assert (by_source["city_admin1"]["admin1_name"] != "").all()
    assert (by_source["city_admin1"]["country_name"] == "").all()
    assert (by_source["city_country"]["admin1_name"] == "").all()
    assert (by_source["city_country"]["country_name"] != "").all()
    assert (by_source["city_admin1_country"]["admin1_name"] != "").all()
    assert (by_source["city_admin1_country"]["country_name"] != "").all()


def test_taken_names_are_scoped_per_language():
    """The same spelling in two languages is two requests, not a duplicate.

    A transliteration can equal the English form, and those are different rows —
    so the cross-group "already drawn" set is keyed by language, not by name.
    """
    rows = [
        (1, "RU", "48", 640_000, "Tyumen", "Tyumen", "en"),
        (1, "RU", "48", 640_000, "Tyumen", "Tyumen", "ru"),
    ]
    cfg = _cfg()
    plan = _plan(
        cfg,
        name_rows=pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row"),
        settings=SimpleNamespace(languages=["en", "ru"], countries=["RU"]),
    )
    tyumen = plan.filter(pl.col("sample_source") == "one_city")

    assert set(tyumen["isolanguage"]) == {"en", "ru"}


# ---------------------------------------------------------------------------
# multi_city
# ---------------------------------------------------------------------------


def test_multi_city_row_count_equals_the_quota_and_gold_is_the_union():
    cfg = _cfg()
    plan = _plan(cfg)
    multi = plan.filter(pl.col("sample_source") == "multi_city")
    gold_by_name = {
        row["name"]: row["geonameid"]
        for row in _plan(cfg)
        .filter(pl.col("sample_source") == "one_city")
        .iter_rows(named=True)
    }

    counts = plan_group_counts(multi)
    assert (counts["len"] == cfg.quotas["multi_city:all"].total).all()
    for row in multi.iter_rows(named=True):
        names = row["name"].split(", ")
        assert 1 + cfg.multi_city_extra_min <= len(names)
        assert len(names) <= 1 + cfg.multi_city_extra_max
        assert len(set(names)) == len(names)
        for name in names:
            if name in gold_by_name:  # a name this plan also drew as one_city
                assert set(gold_by_name[name]) <= set(row["geonameid"])


def test_multi_city_skips_a_group_it_cannot_pair():
    """A group with a single name yields no row instead of raising."""
    rows = [(1, "RU", "48", 100, "Solo", "Solo", "en")]
    cfg = _cfg()
    cfg.quotas["multi_city:all"] = GroupQuota(n_top=1)
    plan = _plan(
        cfg,
        name_rows=pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row"),
        settings=SimpleNamespace(languages=["en"], countries=["RU"]),
    )

    assert not (plan["sample_source"] == "multi_city").any()


def test_multi_city_clamps_the_extras_to_what_the_group_has():
    """`multi_city_extra_max` above the pool size is clamped, not an error."""
    rows = [
        (1, "RU", "48", 100, "Alpha", "Alpha", "en"),
        (2, "RU", "61", 50, "Beta", "Beta", "en"),
    ]
    cfg = _cfg(multi_city_extra_min=1, multi_city_extra_max=5)
    cfg.quotas["multi_city:all"] = GroupQuota(n_top=1)
    plan = _plan(
        cfg,
        name_rows=pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row"),
        settings=SimpleNamespace(languages=["en"], countries=["RU"]),
    )
    multi = plan.filter(pl.col("sample_source") == "multi_city")

    assert multi.height == 1
    assert multi["name"].to_list() == ["Alpha, Beta"]


def test_multi_city_needs_extra_min_partners():
    """A group that cannot supply `extra_min` partners is skipped, not raised on."""
    rows = [
        (1, "RU", "48", 100, "Alpha", "Alpha", "en"),
        (2, "RU", "61", 50, "Beta", "Beta", "en"),
    ]
    cfg = _cfg(multi_city_extra_min=3, multi_city_extra_max=4)
    cfg.quotas["multi_city:all"] = GroupQuota(n_top=1)
    plan = _plan(
        cfg,
        name_rows=pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row"),
        settings=SimpleNamespace(languages=["en"], countries=["RU"]),
    )

    assert not (plan["sample_source"] == "multi_city").any()


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_topic_is_stable_when_other_kinds_are_added():
    """Adding rows of one kind must not reshuffle another kind's topics.

    ``style`` and ``topic`` drawn from one RNG make the second column's stream
    start at offset ``n``, so changing the row count silently rewrites every
    topic — and the plan on disk then disagrees with the checkpoint generated
    against the previous draw.
    """
    only_one_city = _cfg()
    for key in PLAN_KINDS:
        if key != "one_city:all":
            only_one_city.quotas[key] = GroupQuota()

    cols = ["name", "isolanguage", "style", "topic"]
    lean = _plan(only_one_city).filter(pl.col("sample_source") == "one_city")
    full = _plan().filter(pl.col("sample_source") == "one_city")

    assert full.height > 0
    assert full.select(cols).equals(lean.select(cols))


def test_target_choice_is_stable_when_every_target_ties_on_population():
    """The realistic tie: a homonym whose targets are all population-0 villages.

    Sorting targets by population alone leaves the pick to the frame's row order,
    which is not defined — so the plan would name a different region every run.
    The fixture below has no population signal at all, which is exactly the case
    the tiebreak keys exist for.
    """
    rows = [
        (i, "RU", admin1, 0, "Sosnovka", "Sosnovka", "en")
        for i, admin1 in enumerate(["48", "61", "76", "92", "17", "35", "66"], start=1)
    ]
    name_rows = pl.DataFrame(rows, schema=CITY_ROW_SCHEMA, orient="row")
    cfg = _cfg()
    settings = SimpleNamespace(languages=["en"], countries=["RU"])

    plans = [
        _plan(
            cfg,
            name_rows=name_rows.sample(fraction=1.0, shuffle=True, seed=seed),
            settings=settings,
        )
        for seed in range(5)
    ]
    regions = {tuple(plan["admin1_name"].to_list()) for plan in plans}

    assert len(regions) == 1


def test_plan_is_invariant_to_input_row_order():
    """The plan is a function of the row *set*: the source SQL has no ORDER BY.

    Ties in population are the common case (every ``population=0`` village shares
    a rank), so without explicit tiebreaks the head/middle/tail slices would pick
    different names run to run at the same seed.
    """
    name_rows = _name_rows()
    shuffled = name_rows.sample(fraction=1.0, shuffle=True, seed=99)

    assert _plan(name_rows=name_rows).equals(_plan(name_rows=shuffled))


def test_plan_is_reproducible_for_a_seed():
    assert _plan().equals(_plan())
