"""Inference wrapper around the trained CatBoost reranker.

Kept separate from :mod:`src.rerank.train` so the search engine can load and
score without pulling in the training entry point. The document text for each
candidate is the same ``build_descriptions`` string used at training time, so
the model sees identical features online and offline.
"""

from __future__ import annotations

import pandas as pd
from catboost import CatBoost, Pool

from src.rerank.features import FEATURE_COLUMNS, TEXT_FEATURES, build_row
from src.search.engine import GeonameMatch


class StaleRerankModelError(RuntimeError):
    """A model file on disk was trained on a different feature set."""


def load_validated_model(model_path: str) -> CatBoost:
    """Load a CatBoost model, rejecting one built against a different feature set.

    The same reason the BM25 index is versioned on disk: a model file outlives the
    code that produced it, and :data:`~src.rerank.features.FEATURE_COLUMNS` changes
    whenever the feature set does. Predicting anyway would fail deep inside CatBoost
    on every request, so the mismatch is caught here instead.

    Separate from :meth:`Reranker.load` so callers can pay this cheap check *before*
    the expensive part — the engine builds a description per in-scope place, which
    is wasted work if the model is then rejected.
    """
    model = CatBoost()
    model.load_model(model_path)
    trained_on = list(model.feature_names_ or [])
    if trained_on != FEATURE_COLUMNS:
        raise StaleRerankModelError(
            f"{model_path} was trained on {len(trained_on)} features {trained_on}, "
            f"but the current feature set has {len(FEATURE_COLUMNS)} "
            f"{FEATURE_COLUMNS}. Retrain it (make rerank-data && make rerank-train)."
        )
    return model


class Reranker:
    """Score ``(query, document)`` pairs and reorder candidate matches."""

    def __init__(self, model: CatBoost, descriptions: dict[int, str]) -> None:
        self._model = model
        self._descriptions = descriptions

    @classmethod
    def load(cls, model_path: str, descriptions: dict[int, str]) -> Reranker:
        """Load a validated model and pair it with the candidate descriptions."""
        return cls(load_validated_model(model_path), descriptions)

    def rerank(
        self, entity_buckets: dict[str, str], matches: list[GeonameMatch]
    ) -> list[tuple[GeonameMatch, float]]:
        """Return ``(match, score)`` pairs sorted by descending model score.

        ``entity_buckets`` is the NER spans split by type
        (:func:`src.rerank.features.bucket_entities`) — exactly what the engine
        passes and what :func:`src.rerank.dataset.build_pairs` mines from the API
        response, so train- and serve-time features are identical (via the shared
        :func:`src.rerank.features.build_row`). ``retriever_rank`` is the
        candidate's position in the retrieval-ordered ``matches`` list, matching
        the rank mined at training time. Candidates without a known description
        score against an empty document rather than being dropped, so the result
        set stays the same size.
        """
        if not matches:
            return []
        rows = [
            build_row(
                **entity_buckets,
                document=self._descriptions.get(m.geonameid, ""),
                population=m.population,
                retriever_score=m.retriever_score,
                retriever_rank=rank,
            )
            for rank, m in enumerate(matches)
        ]
        pool = Pool(
            data=pd.DataFrame(rows)[FEATURE_COLUMNS], text_features=TEXT_FEATURES
        )
        scores = self._model.predict(pool)
        order = sorted(range(len(matches)), key=lambda i: scores[i], reverse=True)
        return [(matches[i], float(scores[i])) for i in order]
