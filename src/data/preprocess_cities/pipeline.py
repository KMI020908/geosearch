"""Core preprocessing pipeline: fill missing multilingual city names via LLM."""
import asyncio
import json
from pathlib import Path

import polars as pl
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm
from tqdm.asyncio import tqdm as atqdm

from ..config import CYRILLIC_LANGUAGES, LANGUAGES, PROCESSED_DIR, TRANSLATED_LANGUAGES
from .translation import build_messages, geo_translator, latin_to_cyrillic

_DEFAULT_CACHE_PATH = PROCESSED_DIR / "translations_cache.json"


def _load_cache(path: Path) -> dict[str, dict[str, str]]:
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, dict[str, str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(cache, f, ensure_ascii=False)


async def fill_missing_names(
    df: pl.DataFrame,
    max_concurrent: int = 512,
    warm_start: bool = False,
    cache_path: Path | None = None,
) -> pl.DataFrame:
    """
    Ensure every city has at least one name in every target language.

    The input ``df`` comes from ``build_dataset_lf(...).collect()`` and contains
    a nested ``info`` column that is a list-of-lists of structs:
    ``[[{language, names[], is_preferred[]}, ...], ...]``.
    The outer list exists because two group_by levels were used to merge
    duplicate city entries — it is flattened here.

    Filling strategy
    ----------------
    - Languages in TRANSLATED_LANGUAGES (Cyrillic): missing names are translated
      from English by the LLM.  Latin-lookalike chars in the response are then
      corrected to proper Cyrillic via ``latin_to_cyrillic``.
    - All other languages: fall back to the first available English name.
    - Cities with no English name at all are skipped (no name is added).

    Cities are processed in descending population order so the most important
    cities are translated first if the batch is cut short.

    Parameters
    ----------
    df:             Collected city DataFrame.
    max_concurrent: Maximum number of simultaneous LLM requests.

    Returns
    -------
    Same DataFrame with the ``info`` column replaced by a flat list of
    ``{language, names[], is_preferred[]}`` dicts — one per language.
    """
    # ── Pass 1: restructure info and collect translation candidates ───────────
    structured: dict[str, list] = {"geoname_id": [], "info": []}
    empty_names: dict[str, set] = {lang: set() for lang in LANGUAGES}
    to_translate: dict[str, dict[int, str]] = {
        lang: {} for lang in set(LANGUAGES) & TRANSLATED_LANGUAGES
    }

    rows = df.sort("population", descending=True).iter_rows(named=True)
    for row in tqdm(rows, total=df.height, desc="Structuring cities"):
        # Flatten list-of-lists produced by the two-level group_by aggregation.
        flat_info: list[dict] = sum(row["info"], [])

        # Merge entries from all original geoname_ids into one per-language dict.
        lang_data: dict[str, dict] = {
            lang: {"names": [], "is_preferred": []} for lang in LANGUAGES
        }
        for entry in flat_info:
            lang = entry["language"]
            if lang in lang_data:
                lang_data[lang]["names"].extend(entry["names"])
                lang_data[lang]["is_preferred"].extend(entry["is_preferred"])

        # Deduplicate names per language; sort so preferred names come first.
        for lang, data in lang_data.items():
            seen: set[str] = set()
            deduped: list[tuple] = []
            for name, pref in zip(data["names"], data["is_preferred"]):
                if name not in seen:
                    deduped.append((name, pref))
                    seen.add(name)
            deduped.sort(key=lambda x: -x[1])
            lang_data[lang]["names"] = [p[0] for p in deduped]
            lang_data[lang]["is_preferred"] = [p[1] for p in deduped]

        # Queue languages that ended up with no names.
        en_names = lang_data.get("en", {}).get("names", [])
        for lang, data in lang_data.items():
            if not data["names"]:
                empty_names[lang].add(row["geoname_id"])
                if lang in to_translate and en_names:
                    to_translate[lang][row["geoname_id"]] = en_names[0]

        structured["geoname_id"].append(row["geoname_id"])
        structured["info"].append(lang_data)

    # ── Pass 2: batch LLM translation ─────────────────────────────────────────
    cache_path = cache_path or _DEFAULT_CACHE_PATH

    # Cache stores {lang: {str(geoname_id): translated_name}}.
    cache: dict[str, dict[str, str]] = _load_cache(cache_path) if warm_start else {}
    if warm_start and cache:
        total = sum(len(v) for v in cache.values())
        print(f"warm_start: loaded {total} cached translations from {cache_path}")

    translated: dict[str, dict[int, str]] = {}
    sem = asyncio.Semaphore(max_concurrent)

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=30))
    async def _translate(msg: list) -> str:
        async with sem:
            return await geo_translator.ainvoke(msg)

    for lang, entries in to_translate.items():
        if not entries:
            continue

        lang_cache = cache.setdefault(lang, {})

        # Skip entries already translated in a previous run.
        pending = {gid: name for gid, name in entries.items() if str(gid) not in lang_cache}
        cached_count = len(entries) - len(pending)
        if cached_count:
            print(f"warm_start: skipping {cached_count} already-translated [{lang}] entries")

        if pending:
            messages = [build_messages(name, lang) for name in pending.values()]
            responses = await atqdm.gather(
                *[_translate(m) for m in messages],
                desc=f"Translating [{lang}]",
            )
            for gid, r in zip(pending.keys(), responses):
                name = latin_to_cyrillic(r.content) if lang in CYRILLIC_LANGUAGES else r.content
                lang_cache[str(gid)] = name
            _save_cache(cache, cache_path)

        translated[lang] = {gid: lang_cache[str(gid)] for gid in entries}

    # ── Pass 3: apply translations / fallbacks and normalise info structure ────
    result: dict[str, list] = {"geoname_id": [], "info": []}
    rows = pl.DataFrame(structured, strict=False).iter_rows(named=True)
    for row in tqdm(rows, total=len(structured["geoname_id"]), desc="Applying translations"):
        result["geoname_id"].append(row["geoname_id"])
        info_list = []
        for lang, data in row["info"].items():
            if row["geoname_id"] in empty_names[lang]:
                gid = row["geoname_id"]
                if gid in translated.get(lang, {}):
                    # LLM provided a translation.
                    data["names"] = [translated[lang][gid]]
                    data["is_preferred"] = [-1]
                else:
                    # Fallback: use first English name.
                    en_names = row["info"].get("en", {}).get("names", [])
                    data["names"] = [en_names[0]] if en_names else []
                    data["is_preferred"] = [-1] if data["names"] else []
            info_list.append({"language": lang, **data})
        result["info"].append(info_list)

    return df.drop("info").join(pl.DataFrame(result, strict=False), on="geoname_id")
