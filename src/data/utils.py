"""Shared utilities for the data pipeline."""
import polars as pl
from typing import Callable
from .config import PROCESSED_DIR


def make_region_resolver() -> Callable[[str, str, str], str]:
    """Build a resolver ``(country_code, admin1_code, lang_code) -> name``.

    Expects the layout ``PROCESSED_DIR / regions / {CC} / {lang}.json``
    where each JSON maps *admin1_code → name*.  Adding a new country or
    language only requires dropping a file — no code changes needed.
    """
    regions = pl.read_parquet(PROCESSED_DIR / "regions.parquet")
    languages = regions["language"].unique().to_list()
    country_codes = regions["country_code"].unique().to_list()
    region_data = {}
    for cc in country_codes:
        region_data[cc] = {l: {} for l in languages}
    for row in regions.iter_rows(named=True):
        region_data[row["country_code"]][row["language"]][row["admin1_code"]] = row["name"]

    def resolve(country_code: str, admin1_code: str, language_code: str) -> str:
        cc = str(country_code)
        ac1 = str(admin1_code)
        lc = str(language_code)
        lang_map = region_data.get(cc, {}).get(lc, {})
        return lang_map.get(ac1, ac1)

    return resolve