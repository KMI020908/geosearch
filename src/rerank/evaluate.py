"""Golden-set evaluation: the repeatable rerank-vs-baseline gate.

Replays the hand-curated ``golden_set`` through the pipeline twice — once
with the reranker off (retriever + population tiebreak) and once with it on — and
also computes the *oracle* ceiling (perfectly lifting the gold documents among
the retrieved candidates). Printing all three side by side answers the only
question that matters: **does the reranker beat doing nothing, and how far is it
from the achievable ceiling?**

Consolidates what lived across ``notebooks/retriever_metrics*.ipynb`` into one
command so every training change is measured identically. Runs in process by
default; set ``RERANK__SEARCH_URL`` to measure a live deployment instead
(:mod:`src.search.batch`). Run as::

    python -m src.rerank.evaluate
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import polars as pl
from ir_measures import RR, P, Recall, calc

from src.config import RerankConfig, settings
from src.search.batch import search_batch

logger = logging.getLogger(__name__)

_MEASURES = [RR, P @ 1, Recall @ 5, Recall @ 10, Recall @ 25, Recall @ 50]


def _run_from_order(predictions: list[dict]) -> dict[str, dict[str, float]]:
    """Score each candidate by its rank so ir_measures preserves the API order."""
    return {
        pred["query"]: {
            str(r["geonameid"]): float(len(pred["results"]) - i)
            for i, r in enumerate(pred["results"])
        }
        for pred in predictions
    }


def _oracle_run(
    predictions: list[dict], qrels: dict[str, dict[str, int]]
) -> dict[str, dict[str, float]]:
    """Ceiling: among the retrieved candidates, lift every gold document to the top."""
    run: dict[str, dict[str, float]] = {}
    for pred in predictions:
        gold = set(qrels[pred["query"]])
        run[pred["query"]] = {
            str(r["geonameid"]): 1.0 if str(r["geonameid"]) in gold else 0.0
            for r in pred["results"]
        }
    return run


def evaluate(cfg: RerankConfig) -> pl.DataFrame:
    """Return a metric table with baseline / rerank / ideal columns."""
    golden = pl.read_parquet(cfg.golden_set_path).filter(
        pl.col("geonameIds").list.len() > 0
    )
    qrels = {
        row["query"]: {str(gid): 1 for gid in row["geonameIds"]}
        for row in golden.iter_rows(named=True)
    }
    queries = list(qrels)

    logger.info("Baseline run (use_rerank=False) over %d queries…", len(queries))
    base_preds = search_batch(queries, cfg, use_rerank=False)
    logger.info("Reranker run (use_rerank=True)…")
    rerank_preds = search_batch(queries, cfg, use_rerank=True)

    runs = {
        "baseline": _run_from_order(base_preds),
        "rerank": _run_from_order(rerank_preds),
        "ideal": _oracle_run(base_preds, qrels),
    }

    table: dict[str, list] = {"metric": [str(m) for m in _MEASURES]}
    for name, run in runs.items():
        agg = cast(dict, calc(_MEASURES, qrels, run).aggregated)
        table[name] = [round(float(agg[m]), 4) for m in _MEASURES]
    return pl.DataFrame(table)


def write_metrics(metrics: pl.DataFrame, cfg: RerankConfig) -> Path:
    """Persist the comparison table as JSON.

    The counterpart to ``NerConfig.metrics_path``: small, git-tracked, and the
    only place the published model card is allowed to take its numbers from. A
    card that quotes a figure no file contains is one that keeps quoting it
    after it stops being true.
    """
    systems = [c for c in metrics.columns if c != "metric"]
    payload = {
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "golden_set": cfg.golden_set_path,
        "queries": pl.read_parquet(cfg.golden_set_path).height,
        "top_k": cfg.top_k,
        "use_gold_entities": cfg.use_gold_entities,
        "columns": metrics["metric"].to_list(),
        "rows": {name: metrics[name].to_list() for name in systems},
    }
    path = Path(cfg.metrics_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Wrote %s", path)
    return path


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    metrics = evaluate(settings.rerank)
    print(metrics)
    write_metrics(metrics, settings.rerank)

    # Hard gate: the reranker must not lose to the retriever baseline on the
    # top-heavy metrics, or it should not be served.
    gate = metrics.filter(pl.col("metric").is_in(["P@1", "RR"]))
    if (gate["rerank"] < gate["baseline"]).any():
        logger.warning(
            "Reranker is BELOW baseline on P@1 or RR — it degrades ranking. "
            "Do not ship it (keep the use_rerank fallback) until it clears baseline."
        )


if __name__ == "__main__":
    main()
