"""Region name preprocessing: translate admin1 region names into all target languages."""
import asyncio
import json
from pathlib import Path

import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm.asyncio import tqdm as atqdm

from ..config import CYRILLIC_LANGUAGES, PROCESSED_DIR, RAW_DIR
from ..translation import build_messages, geo_translator, latin_to_cyrillic
from .utils import PROCESSORS, TRANSLATE_LANGS, process_fallback, scan_admin1_codes

_DEFAULT_CACHE_PATH = PROCESSED_DIR / "regions_translations_cache.json"


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


async def build_regions(
    languages: list[str],
    countries: list[str],
    max_concurrent: int = 512,
    warm_start: bool = False,
    cache_path: Path | None = None,
    raw_dir: Path | None = None,
) -> pl.DataFrame:
    """
    Build a multilingual region name DataFrame.

    Output schema: admin1_code, country_code, language, name

    All per-country translation logic is defined in utils.py (``PROCESSORS`` and
    ``TRANSLATE_LANGS``).  Countries absent from those dicts fall back to ascii_name
    for every language with no LLM calls, so newly added countries work out of the box.
    """
    raw_dir = raw_dir or RAW_DIR
    cache_path = cache_path or _DEFAULT_CACHE_PATH

    regions = (
        scan_admin1_codes(raw_dir / "admin1CodesASCII.txt")
        .filter(pl.col("country_code").is_in(countries))
        .collect()
    )

    # ── Collect which regions need LLM translation ────────────────────────────
    # TRANSLATE_LANGS[country] defines which languages to translate for each country.
    # Cache key: "{country_code}.{admin1_code}" to avoid cross-country collisions.
    to_translate: dict[str, dict[str, str]] = {lang: {} for lang in languages}

    for row in regions.iter_rows(named=True):
        country = row["country_code"]
        get_langs = TRANSLATE_LANGS.get(country)
        if get_langs is None:
            continue
        cache_key = f"{country}.{row['admin1_code']}"
        for lang in get_langs(languages):
            to_translate[lang][cache_key] = row["ascii_name"]

    # ── LLM translation pass ──────────────────────────────────────────────────
    cache: dict[str, dict[str, str]] = _load_cache(cache_path) if warm_start else {}
    translated: dict[str, dict[str, str]] = {}
    sem = asyncio.Semaphore(max_concurrent)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _translate(msg: list) -> str:
        async with sem:
            return await geo_translator.ainvoke(msg)

    for lang, entries in to_translate.items():
        if not entries:
            continue

        lang_cache = cache.setdefault(lang, {})
        pending = {key: name for key, name in entries.items() if key not in lang_cache}
        cached_count = len(entries) - len(pending)
        if cached_count:
            print(f"warm_start: skipping {cached_count} cached [{lang}] entries")

        if pending:
            messages = [build_messages(name, lang) for name in pending.values()]
            responses = await atqdm.gather(
                *[_translate(m) for m in messages],
                desc=f"Translating regions [{lang}]",
            )
            for key, r in zip(pending.keys(), responses):
                result = latin_to_cyrillic(r.content) if lang in CYRILLIC_LANGUAGES else r.content
                lang_cache[key] = result
            _save_cache(cache, cache_path)

        translated[lang] = {key: lang_cache[key] for key in entries}

    # ── Build output via per-country processors ───────────────────────────────
    rows: list[dict] = []
    for country in countries:
        country_df = regions.filter(pl.col("country_code") == country)
        processor = PROCESSORS.get(country, process_fallback)
        rows.extend(processor(country_df, languages, translated))

    return pl.DataFrame(rows)
