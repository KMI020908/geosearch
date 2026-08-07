"""The place store's contract: resolve what it knows, omit what it does not."""

import polars as pl
import pytest

from src.search.artifacts import PLACES_SCHEMA
from src.search.places import ParquetPlaceStore, Place

ROWS = [
    (524901, "Moscow", "RU", 10381222, "PPLC", 55.75222, 37.61556),
    (5128581, "New York City", "US", 8804190, "PPL", 40.71427, -74.00597),
    (745044, "Istanbul", "TR", 15701602, "PPLA", 41.01384, 28.94966),
    # feature_code and coordinates are nullable in GeoNames, and `Place`
    # declares them optional — a store that cannot carry a null would turn
    # missing data into 0.0, i.e. a place at (0, 0) off the coast of Africa.
    (999999, "Nowhere", "RU", 0, None, None, None),
]


@pytest.fixture
def store(tmp_path) -> ParquetPlaceStore:
    path = tmp_path / "places.parquet"
    pl.DataFrame(ROWS, schema=PLACES_SCHEMA, orient="row").write_parquet(path)
    return ParquetPlaceStore.from_parquet(path)


@pytest.mark.asyncio
async def test_fetches_known_ids(store: ParquetPlaceStore) -> None:
    got = await store.fetch([524901, 745044])
    assert set(got) == {524901, 745044}
    assert got[524901] == Place(
        geonameid=524901,
        asciiname="Moscow",
        country_code="RU",
        population=10381222,
        feature_code="PPLC",
        latitude=55.75222,
        longitude=37.61556,
    )


@pytest.mark.asyncio
async def test_unknown_id_is_omitted_not_defaulted(store: ParquetPlaceStore) -> None:
    """The contract search relies on: a missing id drops the match entirely."""
    assert await store.fetch([-1]) == {}
    assert set(await store.fetch([524901, -1])) == {524901}


@pytest.mark.asyncio
async def test_nullable_fields_survive(store: ParquetPlaceStore) -> None:
    place = (await store.fetch([999999]))[999999]
    assert place.feature_code is None
    assert place.latitude is None and place.longitude is None


@pytest.mark.asyncio
async def test_empty_request_needs_no_work(store: ParquetPlaceStore) -> None:
    assert await store.fetch([]) == {}


@pytest.mark.asyncio
async def test_duplicate_ids_collapse(store: ParquetPlaceStore) -> None:
    """`_rank_candidates` can hand the same id twice; the result is a mapping."""
    assert set(await store.fetch([524901, 524901])) == {524901}


def test_missing_file_names_the_fix(tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="make artifacts"):
        ParquetPlaceStore.from_parquet(tmp_path / "absent.parquet")
