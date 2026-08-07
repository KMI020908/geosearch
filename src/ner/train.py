"""Fine-tune GLiNER on the exported query dataset (``make ner-train``).

Reads ``data/ner/{train,val}.json`` (:mod:`src.ner.dataset`) and writes the tuned
model to ``cfg.model_dir`` — the same path :data:`src.config.Settings.gliner_model`
serves from, so training publishes directly to what the engine loads.

**Model selection is by span F1, not by eval loss.** ``eval_loss`` is the model's
own training objective on the holdout: a reasonable checkpoint-picking signal, but
not the quantity anyone cares about. :class:`BestSpanF1Callback` scores the val
split after every epoch exactly the way the engine will call the model
(:mod:`src.ner.evaluate`) and saves whenever that improves.

Saving from the callback also sidesteps a failure that is silent by construction.
``load_best_model_at_end=True`` makes ``transformers.Trainer`` reload the best
checkpoint into the outer :class:`~gliner.GLiNER` wrapper, whose parameter names
carry a ``model.`` prefix — but ``GLiNER.save_pretrained`` writes the *inner*
module's keys, unprefixed. Every key mismatches, ``strict=False`` swallows it, and
the model left in memory is the **last** epoch rather than the selected one. The
checkpoint currently in ``data/gliner-geosearch/`` was produced that way. Here the
live model is saved at the moment it is measured, so there is no reload to fail.

A ``ner_meta.json`` is written next to the weights recording what the checkpoint
requires — above all the word splitter, which cannot be stored in the checkpoint
itself (``words_splitter_type`` only names GLiNER's built-in kinds) and whose
absence at serving time degrades zh silently.
:meth:`src.search.engine.SearchEngine.build` reads that file and refuses to start
on a mismatch.

Needs a GPU in practice: ~180 training rows × 15 epochs on mdeberta-v3-base is
minutes on a T4 and hours on CPU. It runs on CPU regardless, so a smoke test does
not need one.

Run as::

    python -m src.ner.train
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import torch
import transformers
from gliner import GLiNER
from gliner.training import Trainer, TrainingArguments
from transformers import TrainerCallback, TrainerControl, TrainerState

from src.config import NerConfig, settings
from src.ner.evaluate import Score, evaluate, load_split
from src.ner.tokenizer import CjkAwareSplitter

logger = logging.getLogger(__name__)

# Identifies the splitter contract in ner_meta.json. A checkpoint stamped with
# this must be served with `NER_CJK_SPLITTER=true`; see src/ner/tokenizer.py.
CJK_SPLITTER_ID = "src.ner.tokenizer.CjkAwareSplitter"
META_FILENAME = "ner_meta.json"


def _quiet_cosmetic_warnings() -> None:
    """Silence two warnings that fire every step and mean nothing here.

    Both were verified to be cosmetic before being suppressed, rather than muted
    to make the log look clean:

    * *"Was asked to gather along dimension 0"* — HF's ``Trainer`` gathers the
      scalar training loss across processes at every logging step, and
      ``torch.autograd`` upsizes 0-dim tensors before gathering. No effect on the
      loss values or the optimisation.
    * *"The tokenizer has new PAD/BOS/EOS tokens…"* — ``Trainer.__init__`` aligns
      ``model.config`` pad/bos/eos ids with the tokenizer, a safety check for
      generative models. ``GLiNERConfig`` has no such fields, so this bolts
      SentencePiece ids onto a config GLiNER never reads. A span classifier never
      generates text.
    """
    warnings.filterwarnings(
        "ignore",
        message=r".*Was asked to gather along dimension 0.*",
        category=UserWarning,
    )
    logging.getLogger("transformers.trainer").setLevel(logging.ERROR)


def load_splits(cfg: NerConfig) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Load train/val and verify the token indices were built by our splitter.

    The check is cheap and catches the one failure that destroys a fine-tune
    without producing an error: token indices built with a different splitter
    than inference uses. It re-tokenises each example's raw ``text`` and requires
    the result to equal the stored ``tokenized_text``.
    """
    train = load_split(cfg.train_path)
    val = load_split(cfg.val_path)
    for name, split in (("train", train), ("val", val)):
        _assert_tokenization(split, name)
    logger.info(
        "Loaded %d train / %d val queries (%d / %d spans)",
        len(train),
        len(val),
        sum(len(ex["ner"]) for ex in train),
        sum(len(ex["ner"]) for ex in val),
    )
    return train, val


def _assert_tokenization(split: list[dict[str, Any]], name: str) -> None:
    """Raise if any example's stored tokens disagree with :class:`CjkAwareSplitter`."""
    splitter = CjkAwareSplitter()
    for example in split:
        expected = [token for token, _, _ in splitter(example["text"])]
        if example["tokenized_text"] != expected:
            raise ValueError(
                f"{name} split: tokenized_text for {example['text']!r} was not "
                "produced by CjkAwareSplitter — re-run `make ner-data`. Training "
                "on indices from another splitter yields a model that cannot "
                "predict the spans it was trained on."
            )


def build_model(cfg: NerConfig, device: str) -> GLiNER:
    """Load the zero-shot checkpoint and attach the training-time splitter."""
    model = cast(GLiNER, GLiNER.from_pretrained(cfg.base_model))
    # Train/serve parity: the exported token indices were built with this.
    model.data_processor.words_splitter = CjkAwareSplitter()  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
    model = cast(GLiNER, model.to(device))
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info("Loaded %s (%.1fM parameters) on %s", cfg.base_model, n_params, device)
    return model


def build_training_args(cfg: NerConfig, n_train: int) -> TrainingArguments:
    """Assemble ``TrainingArguments`` for the fine-tune."""
    steps_per_epoch = max(1, n_train // cfg.batch_size)
    return TrainingArguments(
        output_dir=cfg.checkpoint_dir,
        learning_rate=cfg.learning_rate,
        others_lr=cfg.others_lr,
        weight_decay=cfg.weight_decay,
        others_weight_decay=cfg.weight_decay,
        lr_scheduler_type="cosine",
        # transformers>=5.1 merged `warmup_ratio` into `warmup_steps`: a value < 1
        # is read as a ratio, >= 1 as an absolute step count. Same behaviour, and
        # `warmup_ratio` is deprecated for removal in 5.2. The >=5.1 floor in
        # pyproject.toml is what makes passing a fraction here correct.
        warmup_steps=cfg.warmup_ratio,
        per_device_train_batch_size=cfg.batch_size,
        per_device_eval_batch_size=cfg.batch_size,
        num_train_epochs=cfg.epochs,
        logging_steps=steps_per_epoch,
        eval_strategy="epoch",
        # No Trainer-managed checkpoints: BestSpanF1Callback saves the model when
        # its span F1 improves, so per-epoch checkpoints would be ~1.1GB each of
        # something never loaded. See the module docstring for why the Trainer's
        # own best-model reload is avoided rather than merely unused.
        save_strategy="no",
        load_best_model_at_end=False,
        fp16=torch.cuda.is_available(),
        dataloader_num_workers=0,
        report_to="none",
        seed=cfg.seed,
    )


class BestSpanF1Callback(TrainerCallback):
    """Score span F1 on val each epoch and save the model when it improves.

    Holds the metric history so the run can be summarised afterwards, and keeps
    ``best_f1`` / ``best_epoch`` for ``ner_meta.json``.
    """

    def __init__(
        self,
        model: GLiNER,
        val_data: list[dict[str, Any]],
        labels: list[str],
        threshold: float,
        model_dir: Path,
    ) -> None:
        self._model = model
        self._val_data = val_data
        self._labels = labels
        self._threshold = threshold
        self._model_dir = model_dir
        self.best_f1 = -1.0
        self.best_epoch = -1
        self.history: list[dict[str, float]] = []

    def on_epoch_end(
        self,
        # The base hook is typed against transformers' TrainingArguments; GLiNER's
        # subclass is what actually arrives, and narrowing here would make the
        # override incompatible with the interface it implements.
        args: transformers.TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs: Any,
    ) -> None:
        """Evaluate, log, and save if this is the best epoch so far."""
        epoch = int(state.epoch or 0)
        report = evaluate(self._model, self._val_data, self._labels, self._threshold)
        overall: Score = report["overall"]
        self.history.append(
            {
                "epoch": epoch,
                "precision": round(overall.precision, 4),
                "recall": round(overall.recall, 4),
                "f1": round(overall.f1, 4),
            }
        )
        improved = overall.f1 > self.best_f1
        logger.info(
            "epoch %d — val span P %.3f / R %.3f / F1 %.3f%s",
            epoch,
            overall.precision,
            overall.recall,
            overall.f1,
            "  ← best, saving" if improved else "",
        )
        if improved:
            self.best_f1 = overall.f1
            self.best_epoch = epoch
            self._model_dir.mkdir(parents=True, exist_ok=True)
            self._model.save_pretrained(str(self._model_dir))
        # `predict_entities` puts the model in eval mode; training continues on
        # the next epoch and would otherwise run with dropout disabled.
        self._model.train()


def write_meta(
    cfg: NerConfig, callback: BestSpanF1Callback, labels: list[str], threshold: float
) -> Path:
    """Record what the saved checkpoint is and what serving it requires.

    ``words_splitter`` is the field that matters: it is the one requirement the
    checkpoint format cannot express, and the engine refuses to start when the
    configured splitter disagrees with it.
    """
    meta = {
        "base_model": cfg.base_model,
        "labels": labels,
        "words_splitter": CJK_SPLITTER_ID,
        "best_epoch": callback.best_epoch,
        "best_val_span_f1": round(callback.best_f1, 4),
        "eval_threshold": threshold,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "history": callback.history,
        "hyperparameters": {
            "epochs": cfg.epochs,
            "batch_size": cfg.batch_size,
            "learning_rate": cfg.learning_rate,
            "others_lr": cfg.others_lr,
            "weight_decay": cfg.weight_decay,
            "warmup_ratio": cfg.warmup_ratio,
            "seed": cfg.seed,
        },
    }
    path = Path(cfg.model_dir) / META_FILENAME
    path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def train(
    model: GLiNER,
    train_data: list[dict[str, Any]],
    val_data: list[dict[str, Any]],
    cfg: NerConfig,
    labels: list[str],
    threshold: float,
) -> BestSpanF1Callback:
    """Run the fine-tune, returning the callback that tracked the best epoch."""
    data_collator = model.data_collator_class(  # pyright: ignore[reportCallIssue]
        model.config,
        data_processor=model.data_processor,
        prepare_labels=True,
    )
    callback = BestSpanF1Callback(
        model, val_data, labels, threshold, Path(cfg.model_dir)
    )
    trainer = Trainer(
        model=model,
        args=build_training_args(cfg, len(train_data)),
        train_dataset=train_data,  # pyright: ignore[reportArgumentType]
        eval_dataset=val_data,  # pyright: ignore[reportArgumentType]
        data_collator=data_collator,
        # torch again: `data_processor` comes through nn.Module.__getattr__, typed
        # `Tensor | Module`, so nothing on it is visible to the type checker.
        processing_class=model.data_processor.transformer_tokenizer,  # pyright: ignore[reportArgumentType, reportAttributeAccessIssue]
        callbacks=[callback],
    )
    trainer.train()
    return callback


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    _quiet_cosmetic_warnings()
    cfg: NerConfig = settings.ner

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--epochs", type=int, default=cfg.epochs, help="override NER__EPOCHS"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=settings.ner_threshold,
        help="decision threshold used for per-epoch model selection",
    )
    args = parser.parse_args()
    cfg = cfg.model_copy(update={"epochs": args.epochs})

    transformers.set_seed(cfg.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        logger.warning(
            "No CUDA device — training on CPU. This works but takes hours where "
            "a GPU takes minutes."
        )

    train_data, val_data = load_splits(cfg)
    model = build_model(cfg, device)
    callback = train(
        model, train_data, val_data, cfg, settings.ner_labels, args.threshold
    )

    if callback.best_epoch < 0:
        raise RuntimeError("No epoch completed — nothing was saved")

    meta_path = write_meta(cfg, callback, settings.ner_labels, args.threshold)
    logger.info(
        "Saved best model (epoch %d, val span F1 %.3f) to %s; wrote %s",
        callback.best_epoch,
        callback.best_f1,
        cfg.model_dir,
        meta_path,
    )

    if cfg.push_to_hub:
        # Imported here, not at module scope: publishing is optional and this
        # keeps `make ner-train` from depending on the Hub client at import time.
        from src.hub.publish import push_ner_model

        sha = push_ner_model(cfg, settings, repo_id=settings.hf.ner_repo)
        logger.info("Published to %s (%s)", settings.hf.ner_repo, sha)
    logger.info(
        "Next: `make ner-eval` for the full report, then re-run `make rerank` — "
        "the reranker's text features are mined from live NER output, so its "
        "training distribution changed with this model."
    )


if __name__ == "__main__":
    main()
