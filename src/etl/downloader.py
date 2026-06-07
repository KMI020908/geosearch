"""Download raw GeoNames files for configured countries."""

import asyncio
from pathlib import Path

import httpx

from src.config import settings

GEONAMES_BASE = "https://download.geonames.org/export/dump"

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; geosearch-etl/1.0)"}


async def _download_file(client: httpx.AsyncClient, url: str, dest: Path) -> None:
    """Stream a remote file to disk, skipping if already present."""
    if dest.exists():
        print(f"  [skip] {dest.name} already exists")
        return
    print(f"  [get]  {url}")
    async with client.stream("GET", url, headers=_HEADERS) as response:
        response.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            async for chunk in response.aiter_bytes(chunk_size=1024 * 256):
                f.write(chunk)
    print(f"  [ok]   {dest.name}")


async def download_country(country_code: str, data_dir: Path) -> None:
    """Download the main place file for one country."""
    url = f"{GEONAMES_BASE}/{country_code}.zip"
    dest = data_dir / f"{country_code}.zip"
    async with httpx.AsyncClient(timeout=300, follow_redirects=True) as client:
        await _download_file(client, url, dest)


async def download_alternate_names(data_dir: Path) -> None:
    """Download the full alternateNamesV2 archive (shared across all countries)."""
    url = f"{GEONAMES_BASE}/alternateNamesV2.zip"
    dest = data_dir / "alternateNamesV2.zip"
    async with httpx.AsyncClient(timeout=600, follow_redirects=True) as client:
        await _download_file(client, url, dest)


async def download_admin1_codes(data_dir: Path) -> None:
    """Download region codes file."""
    url = f"{GEONAMES_BASE}/admin1CodesASCII.txt"
    dest = data_dir / "admin1CodesASCII.txt"
    async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
        await _download_file(client, url, dest)


async def download_all() -> None:
    """Download all required files for configured countries."""
    data_dir = Path(settings.geonames_data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    print("Downloading country files...")
    await asyncio.gather(
        *[download_country(cc, data_dir) for cc in settings.countries]
    )

    print("Downloading alternateNamesV2...")
    await download_alternate_names(data_dir)

    print("Downloading admin1 codes...")
    await download_admin1_codes(data_dir)

    print("Done.")


if __name__ == "__main__":
    asyncio.run(download_all())
