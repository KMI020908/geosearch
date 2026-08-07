"""Span-level NER metrics: the repeatable gate on the fine-tuned GLiNER model.

Scores the model **the way the engine calls it** — ``predict_entities`` on raw
query text (:meth:`src.search.engine.SearchEngine._extract_spans`), not on the
token indices the trainer consumes — so what is measured here is what retrieval
actually receives. A prediction counts only on an exact ``(start, end, label)``
match: downstream, :mod:`src.rerank.features` matches span text against candidate
names, so a span off by a word is not a partial success there either.

Micro-averaged over spans rather than macro over queries: a query naming three
cities feeds three names into retrieval and should weigh three times as much as
one naming a single city.

Two things this answers that nothing else does:

* **Did fine-tuning help, and where?** ``--baseline`` runs the zero-shot
  checkpoint (``cfg.base_model``) over the same split and prints base / tuned / Δ
  per label and per language. The zh row is the one to read first — the CJK
  splitter exists for it (:mod:`src.ner.tokenizer`).
* **Where should the decision threshold sit?** ``--sweep`` walks
  ``cfg.threshold_sweep``. ``settings.ner_threshold`` is a serving parameter, not
  a property of the model: retrieval is recall-hungry (a city span never
  extracted can never be retrieved, while a spurious span only adds a candidate
  the reranker can demote), so the served operating point belongs below GLiNER's
  own 0.5 default — but *how far* below is a measurement, and this is where it is
  taken.

The full report is written to ``cfg.metrics_path`` as JSON, so a number quoted
later can be traced to the run that produced it.

Run as::

    python -m src.ner.evaluate --baseline --sweep
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
from gliner import GLiNER

from src.config import NerConfig, settings
from src.ner.tokenizer import CjkAwareSplitter

logger = logging.getLogger(__name__)

# (start, end, label) — a span as both gold and prediction express it.
Span = tuple[int, int, str]


class Score:
    """Micro P/R/F1 for one bucket, carrying the counts it was computed from.

    The support is not decoration. The val split is ~47 queries, so a bucket like
    ``label:STATE`` rests on a handful of spans and swings by whole tenths of F1
    on a single one. A rate printed without its support invites reading noise as
    signal.
    """

    def __init__(self, tp: int, fp: int, fn: int) -> None:
        self.tp, self.fp, self.fn = tp, fp, fn
        self.precision = tp / (tp + fp) if tp + fp else 0.0
        self.recall = tp / (tp + fn) if tp + fn else 0.0
        denominator = self.precision + self.recall
        self.f1 = 2 * self.precision * self.recall / denominator if denominator else 0.0

    @property
    def gold(self) -> int:
        """Gold spans in this bucket."""
        return self.tp + self.fn

    @property
    def predicted(self) -> int:
        """Predicted spans in this bucket."""
        return self.tp + self.fp

    def as_dict(self) -> dict[str, float | int]:
        """JSON-serialisable form, rates rounded to the precision they support."""
        return {
            "gold": self.gold,
            "predicted": self.predicted,
            "tp": self.tp,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
        }


def load_model(path: str, *, cjk_splitter: bool = True) -> GLiNER:
    """Load a GLiNER checkpoint with the word splitter it expects.

    The splitter is **not** part of a saved checkpoint — ``words_splitter_type``
    in ``gliner_config.json`` can only name GLiNER's built-in kinds — so nothing
    but this reattaches it. Loading the fine-tuned model without it degrades zh
    silently, which is the whole reason this is a shared loader rather than a
    line repeated at each call site. See :mod:`src.ner.tokenizer`.
    """
    model = cast(GLiNER, GLiNER.from_pretrained(path))
    if cjk_splitter:
        model.data_processor.words_splitter = CjkAwareSplitter()  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return cast(GLiNER, model.to(device))


def predict_spans(
    model: GLiNER, texts: list[str], labels: list[str], threshold: float
) -> list[set[Span]]:
    """Predict a span set per text, one query at a time as the engine does."""
    model.eval()
    predictions: list[set[Span]] = []
    with torch.no_grad():
        for text in texts:
            entities = model.predict_entities(  # pyright: ignore[reportCallIssue]
                text, labels, threshold=threshold, flat_ner=True
            )
            predictions.append({(e["start"], e["end"], e["label"]) for e in entities})
    return predictions


def gold_spans(example: dict[str, Any]) -> set[Span]:
    """The reference spans of one exported example, as character offsets."""
    return {(e["start"], e["end"], e["label"]) for e in example["entities"]}


def score_predictions(
    data: list[dict[str, Any]], predictions: list[set[Span]], labels: list[str]
) -> dict[str, Score]:
    """Bucket predictions into ``overall`` / ``label:*`` / ``lang:*`` scores.

    Every label in ``labels`` gets a bucket even when the split holds none of it,
    so a report is comparable across runs instead of quietly changing shape when
    a rare label disappears from the holdout.
    """
    counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])  # tp, fp, fn
    for label in labels:
        _ = counts[f"label:{label}"]

    for example, predicted in zip(data, predictions, strict=True):
        gold = gold_spans(example)
        for bucket in ("overall", f"lang:{example['language']}"):
            counts[bucket][0] += len(gold & predicted)
            counts[bucket][1] += len(predicted - gold)
            counts[bucket][2] += len(gold - predicted)
        for label in labels:
            g = {s for s in gold if s[2] == label}
            p = {s for s in predicted if s[2] == label}
            counts[f"label:{label}"][0] += len(g & p)
            counts[f"label:{label}"][1] += len(p - g)
            counts[f"label:{label}"][2] += len(g - p)

    return {bucket: Score(*c) for bucket, c in counts.items()}


def evaluate(
    model: GLiNER, data: list[dict[str, Any]], labels: list[str], threshold: float
) -> dict[str, Score]:
    """Micro P/R/F1 per bucket at one decision threshold."""
    predictions = predict_spans(model, [ex["text"] for ex in data], labels, threshold)
    return score_predictions(data, predictions, labels)


def collect_errors(
    model: GLiNER, data: list[dict[str, Any]], labels: list[str], threshold: float
) -> list[dict[str, Any]]:
    """Queries whose predicted span set differs from gold, for error analysis.

    Spans are rendered as their surface text, not offsets: an offset pair says
    nothing about *why* the model was wrong, whereas seeing that it tagged
    "Ростов-на-Дону" as "Ростов" does.
    """
    predictions = predict_spans(model, [ex["text"] for ex in data], labels, threshold)
    errors = []
    for example, predicted in zip(data, predictions, strict=True):
        gold = gold_spans(example)
        if gold == predicted:
            continue
        text = example["text"]
        errors.append(
            {
                "language": example["language"],
                "text": text,
                "gold": [[text[s:e], label] for s, e, label in sorted(gold)],
                "predicted": [[text[s:e], label] for s, e, label in sorted(predicted)],
            }
        )
    return errors


def sweep_threshold(
    model: GLiNER,
    data: list[dict[str, Any]],
    labels: list[str],
    thresholds: list[float],
) -> dict[str, Score]:
    """Overall score at each threshold, keyed by the threshold as a string."""
    return {
        f"{threshold:.2f}": evaluate(model, data, labels, threshold)["overall"]
        for threshold in thresholds
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _print_report(report: dict[str, Score], title: str) -> None:
    """Print one model's buckets."""
    print(f"\n{title}")
    print(f"  {'bucket':<16}{'gold':>6}{'pred':>6}{'P':>8}{'R':>8}{'F1':>8}")
    for bucket, score in report.items():
        if score.gold + score.predicted == 0:
            rates = f"{'n/a':>8}{'n/a':>8}{'n/a':>8}"
        else:
            rates = f"{score.precision:>8.3f}{score.recall:>8.3f}{score.f1:>8.3f}"
        print(f"  {bucket:<16}{score.gold:>6}{score.predicted:>6}{rates}")


def _print_comparison(
    baseline: dict[str, Score], tuned: dict[str, Score], title: str
) -> None:
    """Print base / tuned / Δ F1 side by side, which is the question being asked."""
    print(f"\n{title}")
    header = (
        f"  {'bucket':<16}{'gold':>6}{'P base':>9}{'P tuned':>9}"
        f"{'R base':>9}{'R tuned':>9}{'F1 base':>9}{'F1 tuned':>9}{'ΔF1':>9}"
    )
    print(header)
    for bucket, t in tuned.items():
        b = baseline[bucket]
        print(
            f"  {bucket:<16}{t.gold:>6}{b.precision:>9.3f}{t.precision:>9.3f}"
            f"{b.recall:>9.3f}{t.recall:>9.3f}{b.f1:>9.3f}{t.f1:>9.3f}"
            f"{t.f1 - b.f1:>+9.3f}"
        )


def _print_sweep(sweep: dict[str, Score]) -> None:
    """Print the threshold sweep and name the F1-optimal point."""
    print("\nthreshold sweep (overall)")
    print(f"  {'threshold':>10}{'P':>8}{'R':>8}{'F1':>8}")
    for threshold, score in sweep.items():
        print(
            f"  {threshold:>10}{score.precision:>8.3f}"
            f"{score.recall:>8.3f}{score.f1:>8.3f}"
        )
    best = max(sweep.items(), key=lambda kv: kv[1].f1)
    print(
        f"  best F1 at threshold {best[0]} (F1 {best[1].f1:.3f}); "
        f"currently serving {settings.ner_threshold:.2f}. Retrieval is "
        "recall-hungry, so prefer the lowest threshold whose F1 is near this."
    )


def load_split(path: str) -> list[dict[str, Any]]:
    """Read one exported split (:mod:`src.ner.dataset`)."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg: NerConfig = settings.ner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=cfg.model_dir,
        help="checkpoint to evaluate (default: the fine-tuned model directory)",
    )
    parser.add_argument(
        "--baseline",
        action="store_true",
        help=f"also evaluate the zero-shot {cfg.base_model} and print the delta",
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="report the decision-threshold sweep over NER__THRESHOLD_SWEEP",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.ner_threshold,
        help="decision threshold for the main report (default: the served one)",
    )
    parser.add_argument(
        "--split",
        default=cfg.val_path,
        help="split to score (default: the held-out val split)",
    )
    args = parser.parse_args()

    data = load_split(args.split)
    labels = settings.ner_labels
    logger.info(
        "Scoring %d queries / %d gold spans from %s at threshold %.2f",
        len(data),
        sum(len(gold_spans(ex)) for ex in data),
        args.split,
        args.threshold,
    )

    logger.info("Loading %s…", args.model)
    model = load_model(args.model, cjk_splitter=settings.ner_cjk_splitter)
    report = evaluate(model, data, labels, args.threshold)

    result: dict[str, Any] = {
        "model": args.model,
        "split": args.split,
        "threshold": args.threshold,
        "queries": len(data),
        "labels": labels,
        "cjk_splitter": settings.ner_cjk_splitter,
        "evaluated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "scores": {bucket: score.as_dict() for bucket, score in report.items()},
    }

    if args.baseline:
        logger.info("Loading zero-shot baseline %s…", cfg.base_model)
        # The baseline is measured with the *same* splitter as the tuned model.
        # Scoring it under GLiNER's own whitespace splitter would compare two
        # different input segmentations, and the zh difference would be an
        # artefact of that rather than of fine-tuning.
        baseline_model = load_model(
            cfg.base_model, cjk_splitter=settings.ner_cjk_splitter
        )
        baseline_report = evaluate(baseline_model, data, labels, args.threshold)
        _print_comparison(baseline_report, report, f"{cfg.base_model} → {args.model}")
        result["baseline"] = {
            "model": cfg.base_model,
            "scores": {b: s.as_dict() for b, s in baseline_report.items()},
        }
    else:
        _print_report(report, f"{args.model} @ threshold {args.threshold:.2f}")

    if args.sweep:
        sweep = sweep_threshold(model, data, labels, cfg.threshold_sweep)
        _print_sweep(sweep)
        result["threshold_sweep"] = {t: s.as_dict() for t, s in sweep.items()}

    errors = collect_errors(model, data, labels, args.threshold)
    result["errors"] = errors
    print(f"\n{len(errors)}/{len(data)} queries not tagged exactly. First few:")
    for error in errors[:10]:
        print(f"  [{error['language']}] {error['text']}")
        print(f"      gold {error['gold']}")
        print(f"      pred {error['predicted']}")

    destination = Path(cfg.metrics_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %s", destination)

    # The val split is small enough that a reader will otherwise over-trust it.
    print(
        f"\nNote: {len(data)} val queries — one query is worth roughly "
        f"{100 / max(len(data), 1):.0f}% of a rate. Read these as directional."
    )


if __name__ == "__main__":
    main()
