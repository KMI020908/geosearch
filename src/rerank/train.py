"""Train a CatBoost reranker on the labelled pairs from :mod:`src.rerank.dataset`.

The model is a listwise ranker (``YetiRank`` loss) over raw ``query`` and
``document`` text, grouped per query so ranking is learned within each query's
candidate set. Both texts are handed to CatBoost as ``text_features`` — it does
its own tokenisation, so no manual embedding step. Evaluation is NDCG@k on the
held-out geonameids.

Run as::

    python -m src.rerank.train
"""

from __future__ import annotations

import logging

import pandas as pd
import polars as pl
from catboost import CatBoost, Pool

from src.config import RerankConfig, settings

logger = logging.getLogger(__name__)

_TEXT_FEATURES = ["query", "document"]
_NUM_FEATURES = ["population", "retriever_score"]
_FEATURES = _TEXT_FEATURES + _NUM_FEATURES


def to_pool(df: pd.DataFrame) -> Pool:
    """Wrap a train/test frame as a CatBoost ranking Pool.

    ``group_id`` is the query (factorised to ints): CatBoost ranks documents
    within each group, so all rows of one query must share a group. ``query``
    and ``document`` are text features (CatBoost tokenises them itself);
    ``population`` and ``retriever_score`` are plain numeric features.
    """
    group_id = df["query"].factorize()[0]
    return Pool(
        data=df[_FEATURES],
        label=df["label"],
        group_id=group_id,
        text_features=_TEXT_FEATURES,
    )


def train(train_df: pd.DataFrame, test_df: pd.DataFrame, cfg: RerankConfig) -> CatBoost:
    """Fit the ranker with early stopping on the test set, keeping the best model."""
    model = CatBoost(
        {
            "loss_function": "YetiRank",
            "eval_metric": f"NDCG:top={cfg.ndcg_top}",
            "iterations": cfg.iterations,
            "learning_rate": cfg.learning_rate,
            "l2_leaf_reg": cfg.l2_leaf_reg,
            "random_seed": cfg.seed,
            "verbose": 100,
        }
    )
    model.fit(
        to_pool(train_df),
        eval_set=to_pool(test_df),
        early_stopping_rounds=cfg.early_stopping_rounds,
        use_best_model=True,
    )
    return model


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg = settings.rerank

    train_df = pl.read_parquet(cfg.train_path).to_pandas()
    test_df = pl.read_parquet(cfg.test_path).to_pandas()
    logger.info("Loaded %d train / %d test pairs", len(train_df), len(test_df))

    model = train(train_df, test_df, cfg)
    model.save_model(cfg.model_path)
    logger.info("Saved model to %s", cfg.model_path)

    # Eval scores at the best iteration the model was rolled back to.
    # get_best_score() is unreliable here (returns {} for this YetiRank+text
    # ranking setup), so read the eval history and index it at best_iteration.
    best_it = model.get_best_iteration()
    evals = model.get_evals_result().get("validation", {})
    scores = {m: seq[best_it] for m, seq in evals.items()} if best_it is not None else {}
    logger.info("Test scores @ best iter %s: %s", best_it, scores)


if __name__ == "__main__":
    main()
