"""Generate a synthetic query dataset via DeepSeek.

The sample plan (:mod:`src.dataset.sampling`) fixes *which* cities to write
queries about; this module turns each plan row into one DeepSeek call and
records the returned query. The sample plan is saved alongside the queries so
downstream steps (reranker, retriever eval, …) can reuse the same sampled set.

Three operational features on top of the notebook prototype:

* **tqdm** progress with a live kept/failed count.
* **Batched concurrency** — up to ``cfg.max_workers`` requests are in flight at
  once (the OpenAI SDK is synchronous, so a thread pool is the batch unit).
* **Warm start** — every completed row is appended to a JSONL checkpoint as it
  lands. Re-running skips whatever is already in the checkpoint, so an API
  outage mid-run costs only the unfinished rows. The Parquet is assembled from
  the checkpoint at the end.

Run as::

    python -m src.dataset.generate

Tunables come from ``settings.dataset``; the API key from
``settings.deepseek_api_key`` (env ``DEEPSEEK_API_KEY``).
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import polars as pl
from openai import OpenAI
from tqdm import tqdm

from src.config import DatasetConfig, settings
from src.db.session import AsyncSessionFactory
from src.dataset.sampling import (
    build_cities,
    build_sample_plan,
    load_admin1_names,
    load_name_rows,
)

logger = logging.getLogger(__name__)

_STYLE_INSTRUCTIONS = {
    "search": "Terse browser search input: short, all lowercase, no punctuation.",
    "casual": "Casual and informal, like a chat message.",
    "formal": "One or two complete, grammatically formal sentences.",
    "rich": "A long, multi-sentence passage with rich surrounding context.",
}

_SYSTEM_PROMPT = """
You generate training queries for a geographic search reranker.

For each request you receive a language code, a city name, a style, and a topic.
Return exactly one query that mentions the city, written in the given style,
and naturally related to the given topic.

Rules:
- Write entirely in the specified language.
- The city name must appear in the query but may use any natural
  grammatical form (e.g. genitive or locative case).
- Never return the city name alone as the entire query.
- The topic must be reflected in the query content, not just appended.
- Never add explanation or formatting. Return the query string only.
""".strip()

# Fields carried from a plan row into the checkpoint/output row.
_KEEP_FIELDS = (
    "request_id",
    "geonameid",
    "name",
    "isolanguage",
    "style",
    "topic",
    "sample_source",
)


def build_messages(row: dict) -> list[dict]:
    """Build the chat messages for one ``one_city`` generation call."""
    user = (
        f"language: {row['isolanguage']}\n"
        f"name: {row['name']}\n"
        f"style: {_STYLE_INSTRUCTIONS[row['style']]}\n"
        f"topic: {row['topic']}"
    )
    return [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_one(client: OpenAI, cfg: DatasetConfig, row: dict) -> str:
    """Call DeepSeek for one row and return the query text.

    The OpenAI SDK handles transport errors and 429s internally with backoff
    (``max_retries``); anything it still raises propagates so the caller can
    leave this row for the next warm-start run.
    """
    response = client.chat.completions.create(
        model=cfg.deepseek_model,
        messages=build_messages(row),
        temperature=cfg.temperature,
        reasoning_effort=cfg.reasoning_effort,
        stream=False,
    )
    return (response.choices[0].message.content or "").strip()


def _checkpoint_row(row: dict, query: str) -> dict:
    """Project a plan row + query down to the persisted fields."""
    out = {k: row[k] for k in _KEEP_FIELDS}
    out["request_id"] = int(out["request_id"])
    out["query"] = query
    return out


def load_done_ids(checkpoint_path: str) -> set[int]:
    """Return the ``request_id``s already present in the checkpoint."""
    path = Path(checkpoint_path)
    if not path.exists():
        return set()
    done: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            done.add(int(json.loads(line)["request_id"]))
    return done


def generate_dataset(plan: pl.DataFrame, client: OpenAI, cfg: DatasetConfig) -> None:
    """Generate a query for every plan row, appending each to the checkpoint.

    Rows already in the checkpoint are skipped (warm start). Completed rows are
    written from this single thread as workers finish, so no file lock is
    needed. Rows whose call raises (after the SDK's own retries) or that come
    back empty are logged and left out — a later run picks them up.
    """
    done = load_done_ids(cfg.checkpoint_path)
    todo = [r for r in plan.iter_rows(named=True) if int(r["request_id"]) not in done]
    if done:
        logger.info("warm start: %d already done, %d to go", len(done), len(todo))
    if not todo:
        logger.info("nothing to generate; checkpoint already complete")
        return

    kept = failed = 0
    with (
        open(cfg.checkpoint_path, "a", encoding="utf-8") as sink,
        ThreadPoolExecutor(max_workers=cfg.max_workers) as pool,
    ):
        futures = {pool.submit(generate_one, client, cfg, r): r for r in todo}
        bar = tqdm(as_completed(futures), total=len(todo), unit="q")
        for future in bar:
            row = futures[future]
            try:
                query = future.result()
            except Exception as exc:  # noqa: BLE001 — keep the run alive, retry later
                failed += 1
                logger.warning("request %s failed: %s", row["request_id"], exc)
                continue
            if not query:
                failed += 1
                logger.warning("request %s returned empty query", row["request_id"])
                continue
            sink.write(json.dumps(_checkpoint_row(row, query), ensure_ascii=False))
            sink.write("\n")
            sink.flush()
            kept += 1
            bar.set_postfix(kept=kept, failed=failed)

    logger.info("generated %d queries, %d failed (left for next run)", kept, failed)


def checkpoint_to_parquet(cfg: DatasetConfig) -> int:
    """Materialise the JSONL checkpoint into the final Parquet. Returns rows."""
    df = pl.read_ndjson(cfg.checkpoint_path).rename({"isolanguage": "language"})
    df.write_parquet(cfg.output_path)
    return df.height


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = settings.dataset
    if not settings.deepseek_api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set (add it to .env)")

    logger.info("Loading city names from database…")
    name_rows = asyncio.run(load_name_rows(AsyncSessionFactory, settings))
    admin1_names = load_admin1_names(settings.geonames_data_dir, settings.countries)
    cities = build_cities(name_rows, admin1_names)
    plan = build_sample_plan(cities, settings, cfg, random.Random(cfg.seed))
    logger.info("Planned %d queries over %d distinct names", plan.height, cities.height)

    plan.write_parquet(cfg.plan_path)
    logger.info("Wrote %d plan rows to %s", plan.height, cfg.plan_path)

    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        timeout=cfg.request_timeout,
        max_retries=cfg.max_retries,
    )
    generate_dataset(plan, client, cfg)

    written = checkpoint_to_parquet(cfg)
    logger.info("Wrote %d records to %s", written, cfg.output_path)


if __name__ == "__main__":
    main()
