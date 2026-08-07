"""Apply GeoNames daily delta files (modifications and deletes).

The corpus is two Parquet files, so a delta is read-modify-write rather than an
in-place UPDATE: load both frames, merge the modifications, drop the deletes,
write back. At ~1.3M rows that is seconds, and it keeps the whole staging layer
as plain files.

The one thing the old schema gave for free was ``ondelete=CASCADE`` on the
alternate-name foreign key. :func:`src.corpus.cascade_orphans` does it
explicitly after a geoname delete — otherwise a removed place leaves name
variants behind that retrieval can still match and hydration then cannot resolve.
"""

import asyncio
import logging
from datetime import date
from pathlib import Path

import httpx
import polars as pl

from src import corpus
from src.config import settings
from src.etl.parser import (
    _ALTNAME_COLS,
    _GEONAME_COLS,
    AlternateNameRow,
    GeonameRow,
)

logger = logging.getLogger(__name__)

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
    """Parse one modifications-<date>.txt line; None if not feature_class='P'."""
    cols = line.split("\t")
    if len(cols) < 19:
        return None
    if cols[6] != "P":
        return None
    if cols[8] not in _COUNTRY_CODES:
        return None
    row_data = {name: cols[idx] for idx, name in _GEONAME_COLS.items()}
    return GeonameRow.model_validate(row_data)


def _parse_altname_delta_line(line: str) -> AlternateNameRow | None:
    """Parse one line from alternateNamesModifications-<date>.txt."""
    cols = line.split("\t")
    if len(cols) < 10:
        return None
    lang = cols[2]
    if lang not in settings.languages:
        return None
    row_data = {name: cols[idx] for idx, name in _ALTNAME_COLS.items()}
    return AlternateNameRow.model_validate(row_data)


def _read_ids(delta_file: Path) -> list[int]:
    """Read the geonameids listed in a deletes file."""
    ids: list[int] = []
    with delta_file.open("r", encoding="utf-8") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if cols and cols[0].isdigit():
                ids.append(int(cols[0]))
    return ids


def _read_rows(delta_file: Path, parse) -> list[dict]:
    """Parse a modifications file, dropping lines outside our scope."""
    rows = []
    with delta_file.open("r", encoding="utf-8") as f:
        for line in f:
            row = parse(line.rstrip("\n"))
            if row is not None:
                rows.append(row.model_dump())
    return rows


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

    logger.info("Downloading delta files for %s…", date_str)
    async with httpx.AsyncClient(timeout=120) as client:
        for filename in delta_files.values():
            url = f"{GEONAMES_BASE}/{filename}"
            dest = data_dir / filename
            ok = await _download_delta(client, url, dest)
            logger.info("  %s: %s", filename, "downloaded" if ok else "not available")

    geonames = corpus.load_geonames(settings)
    alternate_names = corpus.load_alternate_names(settings)
    before = (geonames.height, alternate_names.height)

    delete_path = data_dir / delta_files["deletes"]
    if delete_path.exists():
        ids = _read_ids(delete_path)
        geonames = corpus.apply_deletes(geonames, ids, key="geonameid")
        # The FK cascade, made explicit.
        alternate_names = corpus.cascade_orphans(alternate_names, geonames)
        logger.info("  deletes: %d geonameids", len(ids))

    altname_delete_path = data_dir / delta_files["altname_deletes"]
    if altname_delete_path.exists():
        ids = _read_ids(altname_delete_path)
        alternate_names = corpus.apply_deletes(
            alternate_names, ids, key="alternate_name_id"
        )
        logger.info("  deletes: %d alternate names", len(ids))

    mod_path = data_dir / delta_files["modifications"]
    if mod_path.exists():
        rows = _read_rows(mod_path, _parse_geoname_delta_line)
        if rows:
            incoming = pl.DataFrame(rows, schema_overrides=corpus.GEONAME_SCHEMA)
            geonames = corpus.upsert(geonames, incoming, key="geonameid")
        logger.info("  modifications: %d places", len(rows))

    altname_mod_path = data_dir / delta_files["altname_modifications"]
    if altname_mod_path.exists():
        rows = _read_rows(altname_mod_path, _parse_altname_delta_line)
        if rows:
            incoming = pl.DataFrame(rows, schema_overrides=corpus.ALTNAME_SCHEMA)
            # Only names for places we actually hold — a modification can arrive
            # for a geonameid outside our country scope.
            incoming = incoming.join(
                geonames.select("geonameid"), on="geonameid", how="semi"
            )
            alternate_names = corpus.upsert(
                alternate_names, incoming, key="alternate_name_id"
            )
        logger.info("  modifications: %d alternate names", len(rows))

    corpus.write(geonames, alternate_names, settings)
    logger.info(
        "Places %d -> %d, alternate names %d -> %d",
        before[0],
        geonames.height,
        before[1],
        alternate_names.height,
    )
    logger.info("Re-run `make artifacts` so serving picks the change up.")
    logger.info("Delta update complete.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    asyncio.run(apply_daily_delta())


if __name__ == "__main__":
    main()
