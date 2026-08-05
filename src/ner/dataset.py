"""Export the synthetic query dataset as GLiNER fine-tuning data.

Turns ``data/query_dataset.parquet`` (queries + gold spans, see
:mod:`src.dataset.generate`) into the two JSON files GLiNER's trainer eats::

    {"tokenized_text": ["Необходимо", "найти", "рейсы", "в", "Москву", "."],
     "ner": [[4, 4, "CITY"]]}

``ner`` holds **inclusive** token indices, not character offsets — GLiNER scores
token spans — so every gold span has to land exactly on token boundaries of
:mod:`src.ner.tokenizer` (see that module for why the default whitespace splitter
cannot represent Chinese spans). A span that does not align is a span the model
could never predict; the row carrying it is dropped rather than snapped to the
enclosing tokens, and the count is logged.

Each record also keeps the raw ``text`` / ``entities`` (character offsets) and the
row's ``language`` / ``sample_source`` / ``name``. The trainer ignores the extra
keys; evaluation uses them to score ``predict_entities`` output the way the engine
actually calls it — on raw text, per language — instead of on token indices.

The split is **by name, within language**: every row whose ``name`` is the same
place goes to the same side, so a val query cannot be a paraphrase of a train
query about the same city. Splitting rows at random would leave the model
credited for having memorised "Москва is a CITY" rather than for tagging an
unseen name in context. Two caveats: ``multi_city`` rows name extra cities beyond
their anchor ``name``, so those extras can still cross the split; and the val
fraction is of *names*, so the row counts land near ``val_size``, not on it.

Run as::

    python -m src.ner.dataset
"""

from __future__ import annotations

import json
import logging
import random
from collections import Counter
from pathlib import Path
from typing import Any

import polars as pl

from src.config import NerConfig, settings
from src.ner.tokenizer import align_span, tokenize

logger = logging.getLogger(__name__)


def build_example(row: dict[str, Any]) -> dict[str, Any] | None:
    """Convert one dataset row into a GLiNER training record.

    Returns ``None`` if the row has no gold spans, if a span's offsets do not
    match the query text (a corrupt row), or if a span does not align to token
    boundaries — in every case the row cannot teach the model the span it claims
    to carry, so it is dropped whole rather than partially labelled: keeping a
    query that mentions a city with that city *unlabelled* teaches the opposite
    of the intended lesson.
    """
    text = str(row["query"])
    entities = list(row["entities"])
    if not entities:
        return None

    tokens = tokenize(text)
    ner: list[list[Any]] = []
    for entity in entities:
        start, end = int(entity["start"]), int(entity["end"])
        if text[start:end] != entity["text"]:
            logger.warning(
                "request_id=%s: span %r does not match text[%d:%d]=%r — row dropped",
                row["request_id"],
                entity["text"],
                start,
                end,
                text[start:end],
            )
            return None
        aligned = align_span(tokens, start, end)
        if aligned is None:
            logger.warning(
                "request_id=%s (%s): span %r is not on token boundaries — row dropped",
                row["request_id"],
                row["language"],
                entity["text"],
            )
            return None
        ner.append([aligned[0], aligned[1], entity["label"]])

    return {
        "tokenized_text": [token for token, _, _ in tokens],
        "ner": ner,
        # Kept for evaluation and error analysis; the trainer ignores them.
        "text": text,
        "entities": [
            {
                "text": e["text"],
                "label": e["label"],
                "start": int(e["start"]),
                "end": int(e["end"]),
            }
            for e in entities
        ],
        "language": row["language"],
        "sample_source": row["sample_source"],
        "name": row["name"],
        "request_id": int(row["request_id"]),
    }


def build_examples(df: pl.DataFrame) -> list[dict[str, Any]]:
    """Convert every row of the query dataset, dropping the unusable ones."""
    examples = [
        ex for row in df.iter_rows(named=True) if (ex := build_example(row)) is not None
    ]
    dropped = df.height - len(examples)
    if dropped:
        logger.warning("Dropped %d/%d rows (see warnings above)", dropped, df.height)
    return examples


def split_by_name(
    examples: list[dict[str, Any]], val_size: float, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Split into (train, val) holding every row of a ``name`` on one side.

    Names are shuffled and drawn into val language by language, so all four
    languages are represented in val at roughly ``val_size`` of their rows rather
    than one language dominating the holdout by chance. A name seen in two
    languages keeps the side it was first assigned — the no-leakage guarantee wins
    over the per-language target.
    """
    by_language: dict[str, list[dict[str, Any]]] = {}
    for example in examples:
        by_language.setdefault(example["language"], []).append(example)

    val_names: set[str] = set()
    train_names: set[str] = set()
    for language in sorted(by_language):
        rows = by_language[language]
        names = sorted({row["name"] for row in rows})
        random.Random(f"{seed}:{language}").shuffle(names)

        target = val_size * len(rows)
        taken = 0
        for name in names:
            if name in train_names or name in val_names:
                continue
            n_rows = sum(1 for row in rows if row["name"] == name)
            if taken + n_rows <= target:
                val_names.add(name)
                taken += n_rows
            else:
                train_names.add(name)

    train = [ex for ex in examples if ex["name"] not in val_names]
    val = [ex for ex in examples if ex["name"] in val_names]
    return train, val


def describe(examples: list[dict[str, Any]], title: str) -> None:
    """Log row counts, language mix and label counts for one split."""
    languages = Counter(ex["language"] for ex in examples)
    labels = Counter(label for ex in examples for *_, label in ex["ner"])
    logger.info(
        "%s: %d rows, %d spans | languages %s | labels %s",
        title,
        len(examples),
        sum(labels.values()),
        dict(sorted(languages.items())),
        dict(sorted(labels.items())),
    )


def write_json(examples: list[dict[str, Any]], path: str) -> None:
    """Write one split as a JSON array (UTF-8, unescaped)."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg: NerConfig = settings.ner

    df = pl.read_parquet(cfg.query_dataset_path)
    logger.info("Loaded %d rows from %s", df.height, cfg.query_dataset_path)

    examples = build_examples(df)
    train, val = split_by_name(examples, cfg.val_size, cfg.seed)
    describe(train, "train")
    describe(val, "val")

    write_json(train, cfg.train_path)
    write_json(val, cfg.val_path)
    logger.info("Wrote %s and %s", cfg.train_path, cfg.val_path)


if __name__ == "__main__":
    main()
