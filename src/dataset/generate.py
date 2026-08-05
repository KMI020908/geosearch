"""Generate a synthetic query dataset via DeepSeek.

The sample plan (:mod:`src.dataset.sampling`) fixes *which* cities to write
queries about; this module turns each plan row into one DeepSeek call and
records the returned query together with its gold NER spans. The sample plan is
saved alongside the queries so downstream steps (reranker, retriever eval, …)
can reuse the same sampled set.

**Structured output.** Each call returns json — the query, the geographic spans
it contains (surface form, as if GLiNER had already run over it), and the
model's self-reported confidence (:mod:`src.dataset.schemas`). DeepSeek's json
mode has no server-side schema enforcement, so the contract is enforced here in
two layers: Pydantic parsing, then :func:`validate_generation`, a deterministic
check driven by the query type's own :class:`~src.dataset.prompts.PromptSpec`
flags. Span offsets are computed by :func:`locate_spans` from the query text,
never taken from the model.

**Per-type prompts, in type order.** The system prompt is chosen per
``sample_source`` (:mod:`src.dataset.prompts`), so no prompt has to describe a
case that does not apply. Rows are therefore generated one ``sample_source``
group at a time, and the first row of each group is sent *alone* before the rest
fan out: DeepSeek's prefix cache is populated by a completed request, so firing
the whole batch at a cold prefix would miss on every one of them. Cache
hit/miss token counts are logged per group so the effect is measured, not
assumed.

**The model reasons, and that shares the token budget.** ``deepseek-v4-flash``
emits a hidden chain of thought on every call and offers no way to disable it,
so ``cfg.max_tokens`` has to cover the reasoning *and* the answer. Under-size it
and the reasoning consumes the whole budget: the response comes back
``finish_reason="length"`` with **empty** content, which looks like a malformed
body rather than an exhausted budget — :func:`generate_one` names that case
explicitly so the two are distinguishable. ``cfg.reasoning_effort`` is left
unset on purpose; the model's own default is measurably lighter than its
lowest explicit setting. Reasoning tokens are logged per group, peak included,
so the budget stays a measurement rather than a guess.

Other operational features:

* **tqdm** progress with a live kept/invalid/failed count.
* **Batched concurrency** — up to ``cfg.max_workers`` requests are in flight at
  once (the OpenAI SDK is synchronous, so a thread pool is the batch unit).
* **Warm start** — every completed row is appended to a JSONL checkpoint as it
  lands. Re-running skips whatever is already in the checkpoint *at the current
  ``PROMPT_VERSION``*, so an API outage mid-run costs only the unfinished rows,
  while a prompt change invalidates the rows it would have changed. The Parquet
  is assembled from the checkpoint at the end.
* **One in-run retry pass** for rows whose call never returned a body, before
  falling back to the warm start — a dropped connection belongs to the minute it
  happened in, not to the row.

Run as::

    python -m src.dataset.generate               # plan + generate + write Parquet
    python -m src.dataset.generate --plan-only   # plan only, no API calls, no key
    python -m src.dataset.generate --limit 2     # 2 rows per kind, a cheap smoke run

Tunables come from ``settings.dataset``; the API key from
``settings.deepseek_api_key`` (env ``DEEPSEEK_API_KEY``).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import httpx
import polars as pl
from openai import OpenAI
from openai.types.shared import ReasoningEffort
from tqdm import tqdm

from src.config import DatasetConfig, settings
from src.dataset.prompts import (
    PROMPT_VERSION,
    PROMPTS,
    PromptSpec,
    build_messages,
    get_spec,
)
from src.dataset.sampling import (
    build_name_gold,
    build_place_groups,
    build_sample_plan,
    load_admin1_names,
    load_name_rows,
    plan_group_counts,
)
from src.dataset.schemas import Entity, GeneratedQuery, parse_response
from src.db.session import AsyncSessionFactory

# `LABEL_BUCKETS` is the documented single source of truth for the label -> feature
# column mapping ("REGION and STATE both mean admin1"); importing it — rather than
# restating the rule here — is what keeps the validator from drifting away from the
# reranker's view of the same labels. `bucket_entities` then gives the dataset's gold
# bucket columns for free, in the exact shape the engine emits at serve time.
from src.rerank.features import LABEL_BUCKETS, bucket_entities

logger = logging.getLogger(__name__)

_ADMIN1_LABELS = LABEL_BUCKETS["admin1_entities"]

# Fields carried from a plan row into the checkpoint/output row. The plan's other
# columns are provenance (which group and population band a row was drawn from) and
# stay in the plan Parquet, which joins back on `request_id`. `pool` is the
# exception: `sample_source` alone no longer says whether a disambiguation row came
# from a homonym or a unique name, and that is what you filter the dataset on.
_KEEP_FIELDS = (
    "request_id",
    "geonameid",
    "name",
    "isolanguage",
    "style",
    "topic",
    "sample_source",
    "pool",
    "admin1_name",
    "country_name",
)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def locate_spans(query: str, entities: list[Entity]) -> list[Entity] | None:
    """Fill each entity's ``start`` / ``end`` by searching ``query`` for its text.

    Offsets come from ``str.find``, never from the model — an LLM gets character
    positions subtly wrong, and ``str.find`` cannot. The search walks a cursor
    forward so a name occurring twice in one query yields two distinct spans
    instead of both pointing at the first occurrence.

    If the cursor search misses, one whole-string search is tried: spans listed
    out of order is a forgivable slip, a span that does not occur verbatim is
    not. Returns ``None`` in the latter case — this is the check that catches a
    model that lemmatised ("Москва" for a query saying "Москве") or re-spelled a
    span instead of copying it.
    """
    located: list[Entity] = []
    cursor = 0
    for entity in entities:
        start = query.find(entity.text, cursor)
        if start == -1:
            start = query.find(entity.text)
        if start == -1:
            return None
        end = start + len(entity.text)
        located.append(entity.model_copy(update={"start": start, "end": end}))
        cursor = end
    return located


def _expected_city_count(row: dict, spec: PromptSpec) -> int:
    """How many CITY spans this row's query must contain.

    A ``multi_city`` plan row's ``name`` is the comma-joined display string of the
    cities it names (:func:`src.dataset.sampling._build_multi_city_rows`), so the
    count is however many names it lists; every other type names exactly one.
    """
    if not spec.multi_city:
        return 1
    return len([part for part in row["name"].split(",") if part.strip()])


def validate_generation(
    row: dict, spec: PromptSpec, gen: GeneratedQuery
) -> tuple[GeneratedQuery, str | None]:
    """Check one generation against its query type's entity contract.

    Returns the generation (with span offsets filled in) and a failure reason, or
    ``None`` as the reason when the row is clean. Every check reads the
    :class:`~src.dataset.prompts.PromptSpec` flags that the prompt itself was
    built from, so the instruction and the check enforcing it cannot drift apart.

    What this *cannot* check: that the city written is the city requested.
    Declension changes the surface form ("Москва" -> "Москве"), so requiring the
    requested spelling to appear in the span would reject correct Russian and
    Turkish inflections. The CITY-span count is the proxy, and the prompt's
    "invent no place that was not requested" rule carries the rest.
    """
    located = locate_spans(gen.query, gen.entities)
    if located is None:
        missing = [e.text for e in gen.entities if e.text not in gen.query]
        return gen, f"span not found verbatim in query: {missing}"
    gen = gen.model_copy(update={"entities": located})

    counts = {"CITY": 0, "COUNTRY": 0, "admin1": 0}
    for entity in located:
        if entity.label == "CITY":
            counts["CITY"] += 1
        elif entity.label == "COUNTRY":
            counts["COUNTRY"] += 1
        elif entity.label in _ADMIN1_LABELS:
            counts["admin1"] += 1

    expected = {
        "CITY": _expected_city_count(row, spec),
        "admin1": 1 if spec.needs_admin1 else 0,
        "COUNTRY": 1 if spec.needs_country else 0,
    }
    for kind, want in expected.items():
        if counts[kind] != want:
            return gen, f"expected {want} {kind} span(s), got {counts[kind]}"

    if gen.query.strip().casefold() == row["name"].strip().casefold():
        return gen, "query is the bare name"

    return gen, None


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def reasoning_tokens(usage: object) -> int:
    """Reasoning tokens spent on one response, or 0 if the model reports none.

    Read defensively through the whole chain: ``completion_tokens_details`` is
    absent on non-reasoning models and its ``reasoning_tokens`` field is ``None``
    rather than ``0`` when unused.
    """
    details = getattr(usage, "completion_tokens_details", None)
    return int(getattr(details, "reasoning_tokens", 0) or 0)


@dataclass
class UsageStats:
    """Running token accounting for a run or a single ``sample_source`` group.

    Two things are tracked, both because they are the levers this script is
    tuned on. ``prompt_cache_hit_tokens`` / ``prompt_cache_miss_tokens`` are
    DeepSeek extensions to the usage object, absent from the OpenAI SDK's typed
    model, so they are read defensively — they say whether generating one group
    at a time is actually earning its prefix cache. ``reasoning`` is the current
    model's hidden chain of thought, which shares the ``max_tokens`` budget with
    the answer: ``reasoning_peak`` is what says whether that budget is sized
    right for the heavy groups, measured rather than assumed.

    Mutated from the worker threads. The increments race, so the totals are
    accurate to within a few concurrent calls — fine for a budget signal, and not
    worth a lock on the hot path.
    """

    hit: int = 0
    miss: int = 0
    reasoning: int = 0
    reasoning_peak: int = 0

    def add(self, usage: object) -> None:
        self.hit += int(getattr(usage, "prompt_cache_hit_tokens", 0) or 0)
        self.miss += int(getattr(usage, "prompt_cache_miss_tokens", 0) or 0)
        spent = reasoning_tokens(usage)
        self.reasoning += spent
        self.reasoning_peak = max(self.reasoning_peak, spent)

    @property
    def hit_rate(self) -> float:
        total = self.hit + self.miss
        return self.hit / total if total else 0.0

    def merge(self, other: UsageStats) -> None:
        """Fold one group's totals into this one (peak takes the larger of the two)."""
        self.hit += other.hit
        self.miss += other.miss
        self.reasoning += other.reasoning
        self.reasoning_peak = max(self.reasoning_peak, other.reasoning_peak)


@dataclass
class Attempt:
    """The outcome of generating one row: the result, why it failed, what it cost."""

    generation: GeneratedQuery | None
    reason: str | None
    attempts: int


class BudgetExceededError(RuntimeError):
    """``max_tokens`` was spent before the answer began.

    Its own type rather than a ``ValueError`` because it is the one response
    failure that is *deterministic*: the same prompt under the same budget fails
    the same way, so retrying it only pays for the reasoning again.
    """


def generate_one(
    client: OpenAI, cfg: DatasetConfig, row: dict, usage: UsageStats
) -> GeneratedQuery:
    """Call DeepSeek for one row and parse the json response.

    The OpenAI SDK handles transport errors and 429s internally with backoff
    (``max_retries``); anything it still raises propagates so the caller can
    leave this row for the next warm-start run. A malformed body raises here
    instead, and is retried in-process by :func:`generate_validated`.
    """
    extra = {}
    if cfg.reasoning_effort is not None:
        # Passed straight through to DeepSeek, whose accepted set is its own, not
        # the OpenAI SDK's `ReasoningEffort` literal — hence the cast rather than
        # a narrower config type we would only be guessing at. Omitted entirely
        # when unset: sending `null` is not the same as not sending the field.
        extra["reasoning_effort"] = cast(ReasoningEffort, cfg.reasoning_effort)

    response = client.chat.completions.create(
        model=cfg.deepseek_model,
        messages=build_messages(row),
        temperature=cfg.temperature,
        # DeepSeek json mode: no server-side schema, so this only guarantees the
        # body parses as json. The schema is enforced by `parse_response`.
        response_format={"type": "json_object"},
        max_tokens=cfg.max_tokens,
        stream=False,
        **extra,
    )
    usage.add(response.usage)

    choice = response.choices[0]
    if choice.finish_reason == "length":
        # The reasoning tokens ate the budget before the answer started, so the
        # content is empty rather than a half-written query. Naming it here keeps
        # it out of the generic "unparsable response" bucket, where a budget
        # overrun looks identical to a model that cannot produce valid json.
        raise BudgetExceededError(
            f"response hit max_tokens={cfg.max_tokens} "
            f"({reasoning_tokens(response.usage)} reasoning tokens); "
            "raise DATASET__MAX_TOKENS"
        )
    return parse_response(choice.message.content or "")


def generate_validated(
    client: OpenAI, cfg: DatasetConfig, row: dict, usage: UsageStats
) -> Attempt:
    """Generate one row, retrying while the response fails its own contract.

    Up to ``cfg.max_generation_attempts`` tries; a parse failure and a validation
    failure both count as an attempt, since both mean the model ignored the
    contract. The last attempt's result is returned either way — a row the model
    reliably mangles is persisted with its reason (``valid=False``) rather than
    dropped, so it shows up for analysis instead of being silently re-paid for on
    every future run.

    An exhausted token budget is the one failure *not* retried: the prompt and the
    budget are unchanged between attempts, so the next call would spend the same
    ``max_tokens`` to fail identically. It returns immediately with the reason,
    which names the config field to raise.

    Transport errors are *not* caught here: they propagate so the row is left out
    of the checkpoint entirely and picked up by the next warm start.
    """
    spec = get_spec(row["sample_source"])
    generation: GeneratedQuery | None = None
    reason: str | None = None
    for attempt in range(1, cfg.max_generation_attempts + 1):
        try:
            generation = generate_one(client, cfg, row, usage)
        except BudgetExceededError as exc:
            return Attempt(None, f"token budget exhausted: {exc}", attempt)
        except (ValueError, TypeError) as exc:  # json / pydantic rejection
            generation, reason = None, f"unparsable response: {exc}"
            continue
        generation, reason = validate_generation(row, spec, generation)
        if reason is None:
            return Attempt(generation, None, attempt)
    return Attempt(generation, reason, cfg.max_generation_attempts)


def _checkpoint_row(row: dict, result: Attempt) -> dict:
    """Project a plan row + generation down to the persisted fields.

    A row that never produced a usable body is persisted too, as an empty query
    with ``valid=False`` and its reason. It costs one line on disk and stops the
    row from being re-paid for on every future warm start; nothing reaches the
    dataset either way, since :func:`checkpoint_to_parquet` drops ``valid=False``.
    """
    generation = result.generation
    out = {k: row[k] for k in _KEEP_FIELDS}
    out["request_id"] = int(out["request_id"])
    out["query"] = generation.query if generation else ""
    out["entities"] = (
        [e.model_dump() for e in generation.entities] if generation else []
    )
    out["confidence"] = generation.confidence if generation else 0.0
    out["prompt_version"] = PROMPT_VERSION
    out["valid"] = result.reason is None
    out["invalid_reason"] = result.reason or ""
    out["attempts"] = result.attempts
    return out


def load_done_ids(checkpoint_path: str) -> set[int]:
    """Return the ``request_id``s already generated at the current prompt version.

    Rows stamped with an older ``prompt_version`` are *not* counted as done: a
    prompt or contract change means their content no longer matches what this
    code would produce, so they are regenerated. The checkpoint is append-only,
    so those stale rows stay on disk and are filtered out when the Parquet is
    assembled (:func:`checkpoint_to_parquet`) rather than rewritten in place.

    A line that does not parse is skipped rather than fatal. Rows are flushed
    individually, but a process killed mid-write can still leave one truncated
    line behind, and that must not make every future run unstartable.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return set()
    done: set[int] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            logger.warning("skipping malformed checkpoint line %d", lineno)
            continue
        if record.get("prompt_version") == PROMPT_VERSION and "request_id" in record:
            done.add(int(record["request_id"]))
    return done


def group_order(plan: pl.DataFrame) -> list[str]:
    """The ``sample_source`` groups present in ``plan``, in prompt-declaration order.

    Generating one group at a time is what makes DeepSeek's prefix cache useful:
    consecutive requests then share a byte-identical system prompt. The order
    itself is arbitrary but fixed (``PROMPTS`` insertion order) so runs are
    comparable.
    """
    present = set(plan["sample_source"].unique().to_list())
    return [source for source in PROMPTS if source in present]


def generate_dataset(plan: pl.DataFrame, client: OpenAI, cfg: DatasetConfig) -> None:
    """Generate a query for every plan row, appending each to the checkpoint.

    Rows already in the checkpoint at the current ``PROMPT_VERSION`` are skipped
    (warm start). Rows are processed one ``sample_source`` group at a time, and
    each group's first row is sent alone so its system prompt is cached before
    the remaining rows fan out across the pool — a cold prefix hit by
    ``max_workers`` simultaneous requests would miss on all of them.

    Completed rows are written from this single thread as workers finish, so no
    file lock is needed. Rows that come back but fail the entity contract — or
    never produce a usable body at all — are written with ``valid=False`` and
    their reason, so a row the model reliably mangles is paid for once.

    Rows whose call raises (after the SDK's own retries) are collected and given
    one more pass at the end of the run: a dropped connection is a property of
    the minute it happened in, not of the row, and re-running a handful of them
    costs seconds against re-invoking the whole script. Whatever still fails is
    left out of the checkpoint for the next warm start, as before.
    """
    done = load_done_ids(cfg.checkpoint_path)
    todo = [r for r in plan.iter_rows(named=True) if int(r["request_id"]) not in done]
    if done:
        logger.info("warm start: %d already done, %d to go", len(done), len(todo))
    if not todo:
        logger.info("nothing to generate; checkpoint already complete")
        return

    by_source: dict[str, list[dict]] = {}
    for row in todo:
        by_source.setdefault(row["sample_source"], []).append(row)

    # `transport` (the call never returned a body) is counted apart from
    # `unusable` (it returned, repeatedly, but never one that passed the
    # contract): the first says the network or the API flaked, the second says
    # the prompt and the model disagree. One summary number for both would hide
    # which of those a run actually hit.
    # `pending` is the transport failures awaiting their retry pass; it is drained
    # into `transport` only by rows that fail a second time, so the bar's `failed`
    # never counts one row twice.
    counts = {"kept": 0, "invalid": 0, "unusable": 0, "transport": 0, "pending": 0}
    total = UsageStats()
    retry_rows: list[dict] = []

    with (
        open(cfg.checkpoint_path, "a", encoding="utf-8") as sink,
        tqdm(total=len(todo), unit="q") as bar,
    ):

        def handle(
            row: dict, result: Attempt | BaseException, *, retrying: bool = False
        ) -> None:
            """Record one finished row and refresh the progress bar."""
            if isinstance(result, BaseException):
                if retrying:
                    counts["transport"] += 1
                    logger.warning(
                        "request %s failed again: %s", row["request_id"], result
                    )
                else:
                    retry_rows.append(row)
                    counts["pending"] += 1
                    logger.warning("request %s failed: %s", row["request_id"], result)
            else:
                if result.generation is None:
                    counts["unusable"] += 1
                    logger.warning(
                        "request %s unusable after %d attempts: %s",
                        row["request_id"],
                        result.attempts,
                        result.reason,
                    )
                elif result.reason is None:
                    counts["kept"] += 1
                else:
                    counts["invalid"] += 1
                    logger.info(
                        "request %s invalid: %s", row["request_id"], result.reason
                    )
                sink.write(json.dumps(_checkpoint_row(row, result), ensure_ascii=False))
                sink.write("\n")
                sink.flush()
            bar.update(1)
            bar.set_postfix(
                kept=counts["kept"],
                invalid=counts["invalid"],
                failed=counts["transport"] + counts["unusable"] + counts["pending"],
                cache=f"{total.hit_rate:.0%}",
            )

        def run(row: dict, usage: UsageStats) -> Attempt | BaseException:
            try:
                return generate_validated(client, cfg, row, usage)
            except Exception as exc:  # noqa: BLE001 — keep the run alive, retry later
                return exc

        def fan_out(rows: list[dict], usage: UsageStats, *, retrying: bool) -> None:
            """Run ``rows`` across the pool, recording each as it lands."""
            with ThreadPoolExecutor(max_workers=cfg.max_workers) as pool:
                futures = {pool.submit(run, r, usage): r for r in rows}
                for future in as_completed(futures):
                    handle(futures[future], future.result(), retrying=retrying)

        for source in group_order(plan):
            rows = by_source.get(source, [])
            if not rows:
                continue
            group = UsageStats()
            bar.set_description(source)
            # Warm the group's prefix with one completed request before fanning
            # out; DeepSeek only caches prefixes of requests that have finished.
            head, tail = rows[0], rows[1:]
            handle(head, run(head, group))
            if tail:
                fan_out(tail, group, retrying=False)
            total.merge(group)
            logger.info(
                "%s: %d rows, prompt cache %d hit / %d miss (%.0f%%), "
                "reasoning %d tokens (peak %d/request)",
                source,
                len(rows),
                group.hit,
                group.miss,
                group.hit_rate * 100,
                group.reasoning,
                group.reasoning_peak,
            )

        if retry_rows:
            logger.info("retrying %d rows that failed transport", len(retry_rows))
            bar.set_description("retry")
            bar.total += len(retry_rows)
            counts["pending"] = 0
            fan_out(retry_rows, total, retrying=True)

    logger.info(
        "generated %d valid, %d invalid (kept with reason), %d unusable, "
        "%d failed transport (left for next run); prompt cache hit rate %.0f%%, "
        "peak %d reasoning tokens/request against max_tokens=%d",
        counts["kept"],
        counts["invalid"],
        counts["unusable"],
        counts["transport"],
        total.hit_rate * 100,
        total.reasoning_peak,
        cfg.max_tokens,
    )


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def _gold_buckets(entities: list[dict] | None) -> dict[str, str]:
    """The three typed entity strings for one row's gold spans.

    Built with the same :func:`~src.rerank.features.bucket_entities` the engine
    uses on GLiNER output, so ``gold_*_entities`` is directly comparable to the
    live ``entity_buckets`` — that comparison is the point of storing them.
    """
    spans = [(e["text"], e["label"]) for e in entities or []]
    return {f"gold_{k}": v for k, v in bucket_entities(spans).items()}


def checkpoint_to_parquet(cfg: DatasetConfig) -> int:
    """Materialise the JSONL checkpoint into the final Parquet. Returns rows.

    Only rows at the current ``PROMPT_VERSION`` that passed validation are
    written. The checkpoint is append-only, so a regenerated ``request_id`` can
    appear more than once — the last occurrence wins.
    """
    # `infer_schema_length=None` reads the whole file before deciding the schema.
    # The default (100 rows) is a landmine on an append-only checkpoint whose
    # *early* rows are the older, narrower shape: the current columns
    # (`prompt_version`, `entities`, `confidence`) fall outside the inference
    # window and are dropped silently, which then trips the guard below on a file
    # that is full of perfectly good rows.
    df = pl.read_ndjson(cfg.checkpoint_path, infer_schema_length=None)
    if "prompt_version" not in df.columns:
        # Every row predates the structured contract: there is nothing this
        # version can honestly emit, and silently writing the old columns would
        # hand downstream code a schema it no longer expects.
        raise SystemExit(
            f"{cfg.checkpoint_path} has no prompt_version column — every row "
            "predates the current prompt contract. Re-run generation to rebuild."
        )

    df = df.filter(pl.col("prompt_version") == PROMPT_VERSION).unique(
        subset=["request_id"], keep="last", maintain_order=True
    )
    invalid = int((~df["valid"]).sum())
    df = df.filter(pl.col("valid")).drop("valid", "invalid_reason")

    buckets = [_gold_buckets(e) for e in df["entities"].to_list()]
    df = df.with_columns(
        pl.Series(name, [b[name] for b in buckets], dtype=pl.String)
        for name in (
            "gold_city_entities",
            "gold_country_entities",
            "gold_admin1_entities",
        )
    ).rename({"isolanguage": "language"})

    df.write_parquet(cfg.output_path)
    if df.height:
        confidence = df["confidence"]
        logger.info(
            "confidence: min %.2f, median %.2f, mean %.2f, %d rows below 0.8",
            confidence.min(),
            confidence.median(),
            confidence.mean(),
            int((confidence < 0.8).sum()),
        )
    logger.info("dropped %d invalid rows", invalid)
    return df.height


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def _limit_per_source(plan: pl.DataFrame, limit: int) -> pl.DataFrame:
    """Keep the first ``limit`` rows of each ``sample_source`` — a cheap smoke run.

    Per source rather than off the top of the plan, so every prompt is still
    exercised (including the group's lone cache-warming request).
    """
    return plan.filter(pl.int_range(pl.len()).over("sample_source") < limit)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="build and write the sample plan, then exit without calling the API",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="generate at most N rows per sample_source (smoke run); 0 means all",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    cfg = settings.dataset

    logger.info("Loading city names from database…")
    name_rows = asyncio.run(load_name_rows(AsyncSessionFactory, settings))
    admin1_names = load_admin1_names(settings.geonames_data_dir, settings.countries)
    place_groups = build_place_groups(name_rows, admin1_names)
    plan = build_sample_plan(place_groups, build_name_gold(name_rows), settings, cfg)

    plan.write_parquet(cfg.plan_path)
    logger.info("Wrote %d plan rows to %s", plan.height, cfg.plan_path)
    with pl.Config(tbl_rows=-1):
        logger.info("rows per group:\n%s", plan_group_counts(plan))

    # The API key is checked *after* the plan exists: --plan-only is the way to
    # inspect what a run would cost, and it must not need a key to do that.
    if args.plan_only:
        return
    if not settings.deepseek_api_key:
        raise SystemExit("DEEPSEEK_API_KEY is not set (add it to .env)")
    if args.limit:
        plan = _limit_per_source(plan, args.limit)
        logger.info("--limit %d: generating %d rows", args.limit, plan.height)

    client = OpenAI(
        api_key=settings.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        # Per phase, not one number for all of them: connecting to a healthy API
        # takes well under a second, while a single reasoning response has been
        # seen taking 35s under load. A flat budget generous enough for the read
        # would also let a genuinely dead connection hang for that long.
        timeout=httpx.Timeout(
            connect=cfg.connect_timeout,
            read=cfg.read_timeout,
            write=cfg.read_timeout,
            pool=cfg.connect_timeout,
        ),
        max_retries=cfg.max_retries,
    )
    generate_dataset(plan, client, cfg)

    written = checkpoint_to_parquet(cfg)
    logger.info("Wrote %d records to %s", written, cfg.output_path)


if __name__ == "__main__":
    main()
