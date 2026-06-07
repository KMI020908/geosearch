"""Parse GeoNames TSV files into typed Pydantic models."""

import zipfile
from collections.abc import Iterator
from datetime import date
from pathlib import Path

from pydantic import BaseModel, field_validator


class GeonameRow(BaseModel):
    """One row from a {CC}.txt country file."""

    geonameid: int
    name: str
    asciiname: str
    feature_class: str
    feature_code: str | None
    country_code: str
    admin1_code: str | None
    admin2_code: str | None
    population: int
    latitude: float | None
    longitude: float | None
    timezone: str | None
    modification_date: date | None

    @field_validator("feature_code", "admin1_code", "admin2_code", "timezone", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str) -> str | None:
        return v if v else None

    @field_validator("population", mode="before")
    @classmethod
    def coerce_population(cls, v: str) -> int:
        return int(v) if v else 0

    @field_validator("latitude", "longitude", mode="before")
    @classmethod
    def coerce_float(cls, v: str) -> float | None:
        return float(v) if v else None

    @field_validator("modification_date", mode="before")
    @classmethod
    def coerce_date(cls, v: str) -> date | None:
        return date.fromisoformat(v) if v else None


class AlternateNameRow(BaseModel):
    """One row from alternateNamesV2.txt."""

    alternate_name_id: int
    geonameid: int
    isolanguage: str | None
    alternate_name: str
    is_preferred_name: bool
    is_short_name: bool
    is_colloquial: bool
    is_historic: bool

    @field_validator("isolanguage", mode="before")
    @classmethod
    def empty_str_to_none(cls, v: str) -> str | None:
        return v if v else None

    @field_validator(
        "is_preferred_name", "is_short_name", "is_colloquial", "is_historic", mode="before"
    )
    @classmethod
    def coerce_bool(cls, v: str) -> bool:
        return v == "1"


# Columns in {CC}.txt (19 total; we pick by index)
_GEONAME_COLS = {
    0: "geonameid",
    1: "name",
    2: "asciiname",
    6: "feature_class",
    7: "feature_code",
    8: "country_code",
    10: "admin1_code",
    11: "admin2_code",
    14: "population",
    4: "latitude",
    5: "longitude",
    17: "timezone",
    18: "modification_date",
}

# Columns in alternateNamesV2.txt (10 total)
_ALTNAME_COLS = {
    0: "alternate_name_id",
    1: "geonameid",
    2: "isolanguage",
    3: "alternate_name",
    4: "is_preferred_name",
    5: "is_short_name",
    6: "is_colloquial",
    7: "is_historic",
}


def _open_txt_from_zip(zip_path: Path, txt_name: str):
    """Open the text file inside a zip archive."""
    zf = zipfile.ZipFile(zip_path)
    return zf, zf.open(txt_name)


def parse_country_file(country_code: str, data_dir: Path) -> Iterator[GeonameRow]:
    """Yield GeonameRow objects for feature_class='P' entries."""
    zip_path = data_dir / f"{country_code}.zip"
    zf, f = _open_txt_from_zip(zip_path, f"{country_code}.txt")
    with zf, f:
        for raw_line in f:
            line = raw_line.decode("utf-8").rstrip("\n")
            cols = line.split("\t")
            if len(cols) < 19:
                continue
            if cols[6] != "P":
                continue
            row_data = {name: cols[idx] for idx, name in _GEONAME_COLS.items()}
            yield GeonameRow(**row_data)


def parse_alternate_names(
    data_dir: Path, geoname_ids: set[int], languages: list[str]
) -> Iterator[AlternateNameRow]:
    """Yield AlternateNameRow for entries matching our geoname_ids and languages."""
    zip_path = data_dir / "alternateNamesV2.zip"
    zf, f = _open_txt_from_zip(zip_path, "alternateNamesV2.txt")
    with zf, f:
        for raw_line in f:
            line = raw_line.decode("utf-8").rstrip("\n")
            cols = line.split("\t")
            if len(cols) < 10:
                continue
            try:
                geonameid = int(cols[1])
            except ValueError:
                continue
            if geonameid not in geoname_ids:
                continue
            lang = cols[2]
            if lang not in languages:
                continue
            row_data = {name: cols[idx] for idx, name in _ALTNAME_COLS.items()}
            yield AlternateNameRow(**row_data)
