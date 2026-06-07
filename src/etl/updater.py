"""Apply GeoNames daily delta files (modifications and deletes)."""

import asyncio
from datetime import date
from pathlib import Path

import httpx
from sqlalchemy import delete

from src.config import settings
from src.db.models import AlternateName, Geoname
from src.db.session import AsyncSessionFactory
from src.etl.loader import BATCH_SIZE, _upsert_alternate_names, _upsert_geonames
from src.etl.parser import (
    AlternateNameRow,
    GeonameRow,
    _ALTNAME_COLS,
    _GEONAME_COLS,
)

GEONAMES_BASE = "https://download.geonames.org/export/dump"

_COUNTRY_CODES = set(settings.countries)


async def _download_delta(client: httpx.AsyncClient, url: str, dest: Path) -> bool:
    """Download a delta file. Returns False if file doesn't exist on server (404)."""
    try:
        async with client.stream("GET", url) as response:
            if response.status_code == 404:
                return False
            response.raise_for_status()
            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=1024 * 256):
                    f.write(chunk)
        return True
    except httpx.HTTPStatusError:
        return False


def _parse_geoname_delta_line(line: str) -> GeonameRow | None:
    """Parse one line from modifications-<date>.txt, returning None if not feature_class='P'."""
    cols = line.split("\t")
    if len(cols) < 19:
        return None
    if cols[6] != "P":
        return None
    if cols[8] not in _COUNTRY_CODES:
        return None
    row_data = {name: cols[idx] for idx, name in _GEONAME_COLS.items()}
    return GeonameRow(**row_data)


def _parse_altname_delta_line(line: str) -> AlternateNameRow | None:
    """Parse one line from alternateNamesModifications-<date>.txt."""
    cols = line.split("\t")
    if len(cols) < 10:
        return None
    lang = cols[2]
    if lang not in settings.languages:
        return None
    row_data = {name: cols[idx] for idx, name in _ALTNAME_COLS.items()}
    return AlternateNameRow(**row_data)


async def _apply_geoname_deletes(delta_file: Path) -> None:
    """Delete geoname records listed in a deletes file."""
    ids: list[int] = []
    with delta_file.open("r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0].isdigit():
                ids.append(int(cols[0]))

    if not ids:
        return

    async with AsyncSessionFactory() as session:
        await session.execute(delete(Geoname).where(Geoname.geonameid.in_(ids)))
        await session.commit()
    print(f"  Deleted {len(ids)} geoname records")


async def _apply_altname_deletes(delta_file: Path) -> None:
    """Delete alternate name records listed in a deletes file."""
    ids: list[int] = []
    with delta_file.open("r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0].isdigit():
                ids.append(int(cols[0]))

    if not ids:
        return

    async with AsyncSessionFactory() as session:
        await session.execute(
            delete(AlternateName).where(AlternateName.alternate_name_id.in_(ids))
        )
        await session.commit()
    print(f"  Deleted {len(ids)} alternate name records")


async def _apply_geoname_modifications(delta_file: Path) -> None:
    batch: list[GeonameRow] = []
    total = 0
    with delta_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = _parse_geoname_delta_line(line.rstrip("\n"))
            if row is None:
                continue
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                await _upsert_geonames(batch)
                total += len(batch)
                batch.clear()
    if batch:
        await _upsert_geonames(batch)
        total += len(batch)
    print(f"  Applied {total} geoname modifications")


async def _apply_altname_modifications(delta_file: Path) -> None:
    batch: list[AlternateNameRow] = []
    total = 0
    with delta_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = _parse_altname_delta_line(line.rstrip("\n"))
            if row is None:
                continue
            batch.append(row)
            if len(batch) >= BATCH_SIZE:
                await _upsert_alternate_names(batch)
                total += len(batch)
                batch.clear()
    if batch:
        await _upsert_alternate_names(batch)
        total += len(batch)
    print(f"  Applied {total} alternate name modifications")


async def apply_daily_delta(delta_date: date | None = None) -> None:
    """Download and apply all delta files for a given date (defaults to today)."""
    if delta_date is None:
        delta_date = date.today()

    date_str = delta_date.strftime("%Y-%m-%d")
    data_dir = Path(settings.geonames_data_dir) / "deltas"

    delta_files = {
        "modifications": f"modifications-{date_str}.txt",
        "deletes": f"deletes-{date_str}.txt",
        "altname_modifications": f"alternateNamesModifications-{date_str}.txt",
        "altname_deletes": f"alternateNamesDeletes-{date_str}.txt",
    }

    print(f"Downloading delta files for {date_str}...")
    async with httpx.AsyncClient(timeout=120) as client:
        for key, filename in delta_files.items():
            url = f"{GEONAMES_BASE}/{filename}"
            dest = data_dir / filename
            ok = await _download_delta(client, url, dest)
            print(f"  {filename}: {'downloaded' if ok else 'not available'}")

    print("Applying deletes...")
    delete_path = data_dir / delta_files["deletes"]
    if delete_path.exists():
        await _apply_geoname_deletes(delete_path)

    altname_delete_path = data_dir / delta_files["altname_deletes"]
    if altname_delete_path.exists():
        await _apply_altname_deletes(altname_delete_path)

    print("Applying modifications...")
    mod_path = data_dir / delta_files["modifications"]
    if mod_path.exists():
        await _apply_geoname_modifications(mod_path)

    altname_mod_path = data_dir / delta_files["altname_modifications"]
    if altname_mod_path.exists():
        await _apply_altname_modifications(altname_mod_path)

    print("Delta update complete.")


if __name__ == "__main__":
    asyncio.run(apply_daily_delta())
