"""Unit tests for ETL parser — no DB, no network, no real files needed."""

import io
import zipfile
from datetime import date

from src.etl.parser import (
    AlternateNameRow,
    GeonameRow,
    parse_alternate_names,
    parse_country_file,
)


def _make_zip(filename: str, content: str) -> bytes:
    """Create an in-memory zip containing one text file."""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        zf.writestr(filename, content)
    return buf.getvalue()


def _geoname_tsv(**overrides) -> str:
    """Build a valid 19-column tab-separated geoname line."""
    defaults = [
        "12345",  # 0  geonameid
        "Москва",  # 1  name
        "Moskva",  # 2  asciiname
        "",  # 3  alternatenames (ignored)
        "55.75",  # 4  latitude
        "37.62",  # 5  longitude
        "P",  # 6  feature_class
        "PPLC",  # 7  feature_code
        "RU",  # 8  country_code
        "",  # 9  cc2
        "77",  # 10 admin1_code
        "",  # 11 admin2_code
        "",  # 12 admin3_code
        "",  # 13 admin4_code
        "12692466",  # 14 population
        "0",  # 15 elevation
        "144",  # 16 dem
        "Europe/Moscow",  # 17 timezone
        "2023-01-15",  # 18 modification_date
    ]
    for k, v in overrides.items():
        defaults[int(k)] = v
    return "\t".join(defaults)


def _altname_tsv(alt_id: str, geonameid: str, lang: str, name: str) -> str:
    cols = [alt_id, geonameid, lang, name, "1", "0", "0", "0", "", ""]
    return "\t".join(cols)


# ---------------------------------------------------------------------------
# GeonameRow validation
# ---------------------------------------------------------------------------


def test_geoname_row_parses_valid_line():
    row = GeonameRow(
        geonameid="12345",
        name="Москва",
        asciiname="Moskva",
        feature_class="P",
        feature_code="PPLC",
        country_code="RU",
        admin1_code="77",
        admin2_code="",
        population="12692466",
        latitude="55.75",
        longitude="37.62",
        timezone="Europe/Moscow",
        modification_date="2023-01-15",
    )
    assert row.geonameid == 12345
    assert row.name == "Москва"
    assert row.population == 12692466
    assert row.modification_date == date(2023, 1, 15)
    assert row.admin2_code is None


def test_geoname_row_empty_optional_fields_become_none():
    row = GeonameRow(
        geonameid="1",
        name="X",
        asciiname="X",
        feature_class="P",
        feature_code="",
        country_code="US",
        admin1_code="",
        admin2_code="",
        population="",
        latitude="",
        longitude="",
        timezone="",
        modification_date="",
    )
    assert row.feature_code is None
    assert row.admin1_code is None
    assert row.latitude is None
    assert row.population == 0
    assert row.modification_date is None


# ---------------------------------------------------------------------------
# AlternateNameRow validation
# ---------------------------------------------------------------------------


def test_alternate_name_row_parses_flags():
    row = AlternateNameRow(
        alternate_name_id="99",
        geonameid="12345",
        isolanguage="ru",
        alternate_name="Москва",
        is_preferred_name="1",
        is_short_name="0",
        is_colloquial="0",
        is_historic="0",
    )
    assert row.is_preferred_name is True
    assert row.is_short_name is False
    assert row.isolanguage == "ru"


def test_alternate_name_row_empty_language_becomes_none():
    row = AlternateNameRow(
        alternate_name_id="1",
        geonameid="1",
        isolanguage="",
        alternate_name="Test",
        is_preferred_name="0",
        is_short_name="0",
        is_colloquial="0",
        is_historic="0",
    )
    assert row.isolanguage is None


# ---------------------------------------------------------------------------
# parse_country_file
# ---------------------------------------------------------------------------


def test_parse_country_file_yields_only_feature_class_p(tmp_path):
    lines = [
        _geoname_tsv(**{"6": "P"}),
        _geoname_tsv(**{"6": "A", "0": "99999"}),  # admin area — skip
    ]
    zip_bytes = _make_zip("RU.txt", "\n".join(lines))
    (tmp_path / "RU.zip").write_bytes(zip_bytes)

    rows = list(parse_country_file("RU", tmp_path))
    assert len(rows) == 1
    assert rows[0].geonameid == 12345


def test_parse_country_file_skips_short_lines(tmp_path):
    zip_bytes = _make_zip("RU.txt", "only\ttwo\tcolumns\n")
    (tmp_path / "RU.zip").write_bytes(zip_bytes)
    rows = list(parse_country_file("RU", tmp_path))
    assert rows == []


# ---------------------------------------------------------------------------
# parse_alternate_names
# ---------------------------------------------------------------------------


def test_parse_alternate_names_filters_by_id_and_language(tmp_path):
    lines = [
        _altname_tsv("1", "12345", "ru", "Москва"),  # match
        _altname_tsv("2", "12345", "de", "Moskau"),  # wrong language
        _altname_tsv("3", "99999", "ru", "Other"),  # wrong geoname_id
    ]
    zip_bytes = _make_zip("alternateNamesV2.txt", "\n".join(lines))
    (tmp_path / "alternateNamesV2.zip").write_bytes(zip_bytes)

    rows = list(parse_alternate_names(tmp_path, {12345}, ["ru", "en"]))
    assert len(rows) == 1
    assert rows[0].alternate_name == "Москва"
    assert rows[0].isolanguage == "ru"


def test_parse_alternate_names_skips_short_lines(tmp_path):
    zip_bytes = _make_zip("alternateNamesV2.txt", "short\tline\n")
    (tmp_path / "alternateNamesV2.zip").write_bytes(zip_bytes)
    rows = list(parse_alternate_names(tmp_path, {1}, ["ru"]))
    assert rows == []
