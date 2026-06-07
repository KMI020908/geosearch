"""Bulk-load parsed GeoNames data into PostgreSQL."""

import asyncio
from pathlib import Path

from sqlalchemy.dialects.postgresql import insert

from src.config import settings
from src.db.models import AlternateName, Geoname
from src.db.session import AsyncSessionFactory
from src.etl.parser import (
    AlternateNameRow,
    GeonameRow,
    parse_alternate_names,
    parse_country_file,
)

BATCH_SIZE = 100


async def _upsert_geonames(rows: list[GeonameRow]) -> None:
    """Insert or update a batch of Geoname rows."""
    data = [
        {
            "geonameid": r.geonameid,
            "name": r.name,
            "asciiname": r.asciiname,
            "feature_class": r.feature_class,
            "feature_code": r.feature_code,
            "country_code": r.country_code,
            "admin1_code": r.admin1_code,
            "admin2_code": r.admin2_code,
            "population": r.population,
            "latitude": r.latitude,
            "longitude": r.longitude,
            "timezone": r.timezone,
            "modification_date": r.modification_date,
        }
        for r in rows
    ]
    stmt = insert(Geoname).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["geonameid"],
        set_={c: stmt.excluded[c] for c in data[0] if c != "geonameid"},
    )
    async with AsyncSessionFactory() as session:
        await session.execute(stmt)
        await session.commit()


async def _upsert_alternate_names(rows: list[AlternateNameRow]) -> None:
    """Insert or update a batch of AlternateName rows."""
    data = [
        {
            "alternate_name_id": r.alternate_name_id,
            "geonameid": r.geonameid,
            "isolanguage": r.isolanguage,
            "alternate_name": r.alternate_name,
            "is_preferred_name": r.is_preferred_name,
            "is_short_name": r.is_short_name,
            "is_colloquial": r.is_colloquial,
            "is_historic": r.is_historic,
            "source": "geonames",
        }
        for r in rows
    ]
    stmt = insert(AlternateName).values(data)
    stmt = stmt.on_conflict_do_update(
        index_elements=["alternate_name_id"],
        set_={c: stmt.excluded[c] for c in data[0] if c != "alternate_name_id"},
    )
    async with AsyncSessionFactory() as session:
        await session.execute(stmt)
        await session.commit()


async def load_country(country_code: str, data_dir: Path) -> set[int]:
    """Parse and load all populated places for a country. Returns loaded geoname IDs."""
    print(f"  Loading {country_code}...")
    batch: list[GeonameRow] = []
    loaded_ids: set[int] = set()
    total = 0

    for row in parse_country_file(country_code, data_dir):
        batch.append(row)
        loaded_ids.add(row.geonameid)
        if len(batch) >= BATCH_SIZE:
            await _upsert_geonames(batch)
            total += len(batch)
            batch.clear()

    if batch:
        await _upsert_geonames(batch)
        total += len(batch)

    print(f"  {country_code}: {total} places loaded")
    return loaded_ids


async def load_alternate_names(data_dir: Path, geoname_ids: set[int]) -> None:
    """Parse and load alternate names for all loaded geoname IDs."""
    print("  Loading alternate names...")
    batch: list[AlternateNameRow] = []
    total = 0

    for row in parse_alternate_names(data_dir, geoname_ids, settings.languages):
        batch.append(row)
        if len(batch) >= BATCH_SIZE:
            await _upsert_alternate_names(batch)
            total += len(batch)
            batch.clear()

    if batch:
        await _upsert_alternate_names(batch)
        total += len(batch)

    print(f"  Alternate names: {total} rows loaded")


async def load_all() -> None:
    """Load all countries then their alternate names."""
    data_dir = Path(settings.geonames_data_dir)
    all_ids: set[int] = set()

    print("Loading place records...")
    for cc in settings.countries:
        ids = await load_country(cc, data_dir)
        all_ids.update(ids)

    print("Loading alternate names...")
    await load_alternate_names(data_dir, all_ids)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(load_all())
