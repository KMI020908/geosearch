"""Global configuration: paths, language settings, schemas, and feature code mappings."""
from pathlib import Path

import polars as pl

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT_DIR / "data" / "raw"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"

# ── Data scope ────────────────────────────────────────────────────────────────
COUNTRIES: list[str] = ["RU", "US", "TR"]
LANGUAGES: list[str] = ["ru", "en", "tr"]
INDEX_PATH = PROCESSED_DIR / "bm25_index.pkl"
GLINER_MODEL = str(ROOT_DIR / "models" / "gliner_multi-v2.1")

# ── Language groups ───────────────────────────────────────────────────────────
# Languages that use the Cyrillic script.
CYRILLIC_LANGUAGES: frozenset[str] = frozenset({"ru", "bg", "uk", "sr", "mk", "be", "kk", "mn"})

# Languages for which missing city names are filled via LLM translation.
# Currently limited to Cyrillic-script languages; others fall back to the English name.
TRANSLATED_LANGUAGES: frozenset[str] = CYRILLIC_LANGUAGES

# ── Output columns for the final city records ─────────────────────────────────
CITY_COLUMNS: list[str] = [
    "geoname_id",
    "ascii_name",
    "latitude",
    "longitude",
    "country_code",
    "admin1_code",
    "population",
    "dem",
    "timezone",
    "place_type",
]

# ── GeoNames feature codes → human-readable place type labels ─────────────────
FEATURE_CODE_MAPPING: dict[str, str] = {
    "PPL":   "populated place",
    "PPLA":  "seat of a first-order administrative division",
    "PPLA2": "seat of a second-order administrative division",
    "PPLA3": "seat of a third-order administrative division",
    "PPLA4": "seat of a fourth-order administrative division",
    "PPLA5": "seat of a fifth-order administrative division",
    "PPLC":  "capital of a political entity",
    "PPLCH": "historical capital of a political entity",
    "PPLF":  "farm village",
    "PPLG":  "seat of government of a political entity",
    "PPLH":  "historical populated place",
    "PPLL":  "populated locality",
    "PPLQ":  "abandoned populated place",
    "PPLR":  "religious populated place",
    "PPLS":  "populated places",
    "PPLW":  "destroyed populated place",
    "PPLX":  "section of populated place",
    "STLMT": "israeli settlement",
}

# ── Polars schemas for raw GeoNames files ─────────────────────────────────────
CITIES_SCHEMA: dict = {
    "geoname_id":        pl.Int64,
    "name":              pl.Utf8,
    "ascii_name":        pl.Utf8,
    "alternate_names":   pl.Utf8,
    "latitude":          pl.Float64,
    "longitude":         pl.Float64,
    "feature_class":     pl.Utf8,
    "feature_code":      pl.Utf8,
    "country_code":      pl.Utf8,
    "cc2":               pl.Utf8,
    "admin1_code":       pl.Utf8,
    "admin2_code":       pl.Utf8,
    "admin3_code":       pl.Utf8,
    "admin4_code":       pl.Utf8,
    "population":        pl.Int64,
    "elevation":         pl.Int32,
    "dem":               pl.Int32,
    "timezone":          pl.Utf8,
    "modification_date": pl.Date,
}

ALTERNATE_NAMES_SCHEMA: dict = {
    "alternate_name_id": pl.Utf8,
    "geoname_id":        pl.Utf8,
    "iso_language":      pl.Utf8,
    "alternate_name":    pl.Utf8,
    "is_preferred_name": pl.Utf8,
    "is_short_name":     pl.Utf8,
    "is_colloquial":     pl.Utf8,
    "is_historic":       pl.Utf8,
    "from":              pl.Utf8,
    "to":                pl.Utf8,
}
