"""Fine-tune a cross-encoder reranker on the mined pairs from :mod:`src.rerank.dataset`.

The base model (``cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`` by default) is
fine-tuned pointwise with ``BinaryCrossEntropyLoss`` — each mined row is one
``(query_text, document)`` pair with a 0/1 label, no reshaping into ranking
groups required. ``top_k=50`` retrieval mines roughly one positive per query
against up to 50 negatives, so the loss is given a ``pos_weight`` computed from
the train split's own class ratio; without it, BCE collapses toward predicting
everything negative.

**Model selection is by P@1 on the held-out mined pairs, not by eval loss.**
``eval_loss`` is the pointwise training objective; the quantity that matters is
whether the top-scored candidate *within a query's group* is the gold one — the
same top-heavy quantity :mod:`src.rerank.evaluate`'s golden-set gate checks.
:class:`BestRankCallback` scores the test split's candidates grouped by query
after every epoch and saves whenever that P@1 improves, the same pattern
:class:`src.ner.train.BestSpanF1Callback` uses and for the same reason: the
live model is saved the moment it is measured, so there is no
``load_best_model_at_end`` reload to silently pick the wrong epoch.

Run as::

    python -m src.rerank.train
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import torch
import transformers
from datasets import Dataset
from sentence_transformers.cross_encoder import (
    CrossEncoder,
    CrossEncoderTrainer,
    CrossEncoderTrainingArguments,
)
from sentence_transformers.cross_encoder.losses import BinaryCrossEntropyLoss
from transformers import TrainerCallback, TrainerControl, TrainerState

from src.config import RerankConfig, settings

logger = logging.getLogger(__name__)

META_FILENAME = "rerank_meta.json"


def _positive_rate(df: pl.DataFrame) -> float:
    return 100 * int(df["label"].sum()) / df.height if df.height else 0.0


def load_splits(cfg: RerankConfig) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Load the mined train/test parquet written by ``make rerank-data``."""
    train_df = pl.read_parquet(cfg.train_path)
    test_df = pl.read_parquet(cfg.test_path)
    logger.info(
        "Loaded %d train / %d test pairs (%.1f%% / %.1f%% positive)",
        train_df.height,
        test_df.height,
        _positive_rate(train_df),
        _positive_rate(test_df),
    )
    return train_df, test_df


def compute_pos_weight(train_df: pl.DataFrame) -> float:
    """Ratio of negative to positive rows in the train split.

    ``top_k=50`` retrieval mines roughly one gold positive per query against up
    to 50 negatives, so this is normally large (tens) — passed to
    ``BCEWithLogitsLoss`` so a false negative costs as much as the many more
    frequent false positives would otherwise dominate.
    """
    n_pos = int(train_df["label"].sum())
    n_neg = train_df.height - n_pos
    if n_pos == 0:
        raise ValueError(
            "Train split has no positive rows — nothing to weight against."
        )
    return n_neg / n_pos


def to_dataset(df: pl.DataFrame) -> Dataset:
    """Select the model's input columns and hand them to 🤗 ``datasets``.

    Only ``query_text``/``document``/``label`` — ``BinaryCrossEntropyLoss``
    expects exactly two non-label columns, and ``query`` (the split/group key)
    is never a model input.
    """
    return Dataset.from_dict(
        df.select("query_text", "document", "label").to_dict(as_series=False)
    )


def build_model(cfg: RerankConfig, device: str) -> CrossEncoder:
    """Load the zero-shot base checkpoint to fine-tune."""
    model = CrossEncoder(
        cfg.base_model, num_labels=1, max_length=cfg.max_length, device=device
    )
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded %s (%.1fM parameters) on %s", cfg.base_model, n_params, device)
    return model


def build_training_args(
    cfg: RerankConfig, n_train: int
) -> CrossEncoderTrainingArguments:
    """Assemble ``CrossEncoderTrainingArguments`` for the fine-tune."""
    steps_per_epoch = max(1, n_train // cfg.batch_size)
    return CrossEncoderTrainingArguments(
        output_dir=cfg.checkpoint_dir,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        logging_steps=steps_per_epoch,
        # No Trainer-managed checkpoints or built-in eval loop: BestRankCallback
        # scores the held-out split itself and saves when P@1 improves — see the
        # module docstring for why the Trainer's own best-model reload is
        # avoided rather than merely unused.
        eval_strategy="no",
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=cfg.seed,
    )


def _group_p_at_1(model: CrossEncoder, df: pl.DataFrame) -> float:
    """Fraction of query groups where the top-scored candidate is gold.

    Groups with no positive candidate at all (retrieval missed the gold place
    entirely) are excluded — there is no correct answer within them for the
    model to be scored on, the same stance :mod:`src.rerank.evaluate` takes.
    """
    pairs = list(zip(df["query_text"].to_list(), df["document"].to_list(), strict=True))
    scores = model.predict(pairs, show_progress_bar=False)
    frame = df.with_columns(pl.Series("score", scores))

    hits = 0
    total = 0
    for (_query,), group in frame.group_by("query", maintain_order=True):
        if 1 not in group["label"].to_list():
            continue
        total += 1
        top = group["score"].arg_max()
        assert top is not None  # non-empty group, arg_max cannot be None
        hits += int(group["label"][top] == 1)
    return hits / total if total else 0.0


class BestRankCallback(TrainerCallback):
    """Score P@1 on the held-out test split each epoch, saving on improvement.

    Holds the metric history so the run can be summarised afterwards, and keeps
    ``best_p_at_1``/``best_epoch`` for ``rerank_meta.json``.
    """

    def __init__(
        self, model: CrossEncoder, test_df: pl.DataFrame, model_path: Path
    ) -> None:
        self._model = model
        self._test_df = test_df
        self._model_path = model_path
        self.best_p_at_1 = -1.0
        self.best_epoch = -1
        self.history: list[dict[str, float]] = []

    def on_epoch_end(
        self,
        args: transformers.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Evaluate, log, and save if this is the best epoch so far."""
        epoch = int(state.epoch or 0)
        p_at_1 = _group_p_at_1(self._model, self._test_df)
        self.history.append({"epoch": epoch, "p_at_1": round(p_at_1, 4)})
        improved = p_at_1 > self.best_p_at_1
        logger.info(
            "epoch %d — test P@1 %.3f%s",
            epoch,
            p_at_1,
            "  ← best, saving" if improved else "",
        )
        if improved:
            self.best_p_at_1 = p_at_1
            self.best_epoch = epoch
            self._model_path.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(self._model_path))
        # predict() puts the model in eval mode; training continues on the next
        # epoch and would otherwise run with dropout disabled.
        self._model.train()


def write_meta(
    cfg: RerankConfig, callback: BestRankCallback, pos_weight: float
) -> Path:
    """Record what the saved checkpoint is, for the model card and for `rerank-eval`."""
    meta = {
        "base_model": cfg.base_model,
        "best_epoch": callback.best_epoch,
        "best_p_at_1": round(callback.best_p_at_1, 4),
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": callback.history,
        "use_gold_entities": cfg.use_gold_entities,
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "warmup_ratio": cfg.warmup_ratio,
            "weight_decay": cfg.weight_decay,
            "max_length": cfg.max_length,
            "pos_weight": round(pos_weight, 4),
            "seed": cfg.seed,
        },
    }
    path = Path(cfg.model_path) / META_FILENAME
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def train(
    model: CrossEncoder,
    train_df: pl.DataFrame,
    test_df: pl.DataFrame,
    cfg: RerankConfig,
) -> tuple[BestRankCallback, float]:
    """Run the fine-tune, returning the callback and the ``pos_weight`` used."""
    pos_weight = (
        cfg.pos_weight if cfg.pos_weight is not None else compute_pos_weight(train_df)
    )
    logger.info("pos_weight = %.2f", pos_weight)
    loss = BinaryCrossEntropyLoss(model, pos_weight=torch.tensor(pos_weight))

    callback = BestRankCallback(model, test_df, Path(cfg.model_path))
    trainer = CrossEncoderTrainer(
        model=model,
        args=build_training_args(cfg, train_df.height),
        train_dataset=to_dataset(train_df),
        loss=loss,
        callbacks=[callback],
    )
    trainer.train()
    return callback, pos_weight


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg: RerankConfig = settings.rerank

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=cfg.epochs, help="override RERANK__EPOCHS"
    )
    args = parser.parse_args()
    cfg = cfg.model_copy(update={"epochs": args.epochs})

    if cfg.use_gold_entities:
        logger.warning(
            "RERANK__USE_GOLD_ENTITIES is true — training on the gold-entity "
            "ablation. This model is NOT servable; see src/rerank/dataset.py."
        )

    transformers.set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.info(
            "No CUDA device — training on CPU. mmarco-mMiniLMv2-L12-H384-v1 is "
            "small enough that this is fine, just slower than a GPU."
        )

    train_df, test_df = load_splits(cfg)
    model = build_model(cfg, device)
    callback, pos_weight = train(model, train_df, test_df, cfg)

    if callback.best_epoch < 0:
        raise RuntimeError("No epoch completed — nothing was saved")

    meta_path = write_meta(cfg, callback, pos_weight)
    logger.info(
        "Saved best model (epoch %d, test P@1 %.3f) to %s; wrote %s",
        callback.best_epoch,
        callback.best_p_at_1,
        cfg.model_path,
        meta_path,
    )

    if cfg.push_to_hub:
        # Imported here, not at module scope: publishing is optional and this
        # keeps `make rerank-train` from depending on the Hub client at import time.
        from src.hub.publish import push_reranker

        sha = push_reranker(settings, repo_id=settings.hf.reranker_repo)
        logger.info("Published to %s (%s)", settings.hf.reranker_repo, sha)
    logger.info(
        "Next: `make rerank-eval` for the golden-set rerank-vs-baseline report."
    )


if __name__ == "__main__":
    main()
