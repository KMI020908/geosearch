"""Helpers for region preprocessing: file scanning and per-country translation assembly."""
from pathlib import Path

import polars as pl


def scan_admin1_codes(path: str | Path) -> pl.LazyFrame:
    """Parse admin1CodesASCII.txt into a LazyFrame.

    Columns: country_code, admin1_code, name, ascii_name, geoname_id.
    Source file is tab-separated:
    ``{countryCode}.{admin1Code}  name  asciiName  geonameId``
    """
    schema = {
        "codes": pl.Utf8,
        "name": pl.Utf8,
        "ascii_name": pl.Utf8,
        "geoname_id": pl.Int64,
    }
    return (
        pl.scan_csv(
            path,
            separator="\t",
            has_header=False,
            schema=schema,
            quote_char=None,
        )
        .with_columns(codes=pl.col("codes").str.split("."))
        .with_columns(
            country_code=pl.col("codes").list.get(0),
            admin1_code=pl.col("codes").list.get(1),
        )
        .drop("codes")
    )


# ── Per-country processors ────────────────────────────────────────────────────


def process_RU(
    df: pl.DataFrame,
    languages: list[str],
    translated: dict[str, dict[str, str]],
) -> list[dict]:
    """RU: en=ascii_name, ru/tr and any other non-English → LLM, fallback=ascii_name."""
    out = []
    for row in df.iter_rows(named=True):
        cache_key = f"RU.{row['admin1_code']}"
        for lang in languages:
            if lang == "en":
                name = row["ascii_name"]
            elif cache_key in translated.get(lang, {}):
                name = translated[lang][cache_key]
            else:
                name = row["ascii_name"]
            out.append({"admin1_code": row["admin1_code"], "country_code": "RU", "language": lang, "name": name})
    return out


def process_TR(
    df: pl.DataFrame,
    languages: list[str],
    translated: dict[str, dict[str, str]],
) -> list[dict]:
    """TR: en=ascii_name, tr=name column (native Turkish), other non-English → LLM, fallback=ascii_name."""
    out = []
    for row in df.iter_rows(named=True):
        cache_key = f"TR.{row['admin1_code']}"
        for lang in languages:
            if lang == "en":
                name = row["ascii_name"]
            elif lang == "tr":
                name = row["name"]
            elif cache_key in translated.get(lang, {}):
                name = translated[lang][cache_key]
            else:
                name = row["ascii_name"]
            out.append({"admin1_code": row["admin1_code"], "country_code": "TR", "language": lang, "name": name})
    return out


def process_US(
    df: pl.DataFrame,
    languages: list[str],
    translated: dict[str, dict[str, str]],
) -> list[dict]:
    """US: en=ascii_name, tr=ascii_name (English used as-is), other non-English → LLM, fallback=ascii_name."""
    out = []
    for row in df.iter_rows(named=True):
        cache_key = f"US.{row['admin1_code']}"
        for lang in languages:
            if lang in ("en", "tr"):
                name = row["ascii_name"]
            elif cache_key in translated.get(lang, {}):
                name = translated[lang][cache_key]
            else:
                name = row["ascii_name"]
            out.append({"admin1_code": row["admin1_code"], "country_code": "US", "language": lang, "name": name})
    return out


def process_fallback(
    df: pl.DataFrame,
    languages: list[str],
    translated: dict[str, dict[str, str]],
) -> list[dict]:
    """Unknown country: ascii_name for every language (no LLM translation)."""
    out = []
    for row in df.iter_rows(named=True):
        for lang in languages:
            out.append(
                {
                    "admin1_code": row["admin1_code"],
                    "country_code": row["country_code"],
                    "language": lang,
                    "name": row["ascii_name"]
                }
            )
    return out


# ── Dispatch tables (all per-country knowledge lives here) ────────────────────

# Maps country_code -> processor function(df, languages, translated) -> list[dict].
# Countries absent from this dict are handled by process_fallback.
PROCESSORS: dict[str, callable] = {
    "RU": process_RU,
    "TR": process_TR,
    "US": process_US,
}

# Maps country_code -> languages that need LLM translation for that country.
# Pipeline uses this to collect translation candidates without any country-specific logic.
TRANSLATE_LANGS: dict[str, callable] = {
    "RU": lambda langs: [lang for lang in langs if lang != "en"],
    "TR": lambda langs: [lang for lang in langs if lang not in ("en", "tr")],
    "US": lambda langs: [lang for lang in langs if lang not in ("en", "tr")],
}
