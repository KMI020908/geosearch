"""Inference wrapper around the trained CatBoost reranker.

Kept separate from :mod:`src.rerank.train` so the search engine can load and
score without pulling in the training entry point. The document text for each
candidate is the same ``build_descriptions`` string used at training time, so
the model sees identical features online and offline.
"""

from __future__ import annotations

import pandas as pd
from catboost import CatBoost, Pool

from src.search.engine import GeonameMatch

_TEXT_FEATURES = ["query", "document"]
_NUM_FEATURES = ["population", "retriever_score"]
_FEATURES = _TEXT_FEATURES + _NUM_FEATURES


class Reranker:
    """Score ``(query, document)`` pairs and reorder candidate matches."""

    def __init__(self, model: CatBoost, descriptions: dict[int, str]) -> None:
        self._model = model
        self._descriptions = descriptions

    @classmethod
    def load(cls, model_path: str, descriptions: dict[int, str]) -> "Reranker":
        model = CatBoost()
        model.load_model(model_path)
        return cls(model, descriptions)

    def rerank(
        self, query_text: str, matches: list[GeonameMatch]
    ) -> list[tuple[GeonameMatch, float]]:
        """Return ``(match, score)`` pairs sorted by descending model score.

        Candidates without a known description score against an empty document
        rather than being dropped, so the result set stays the same size. The
        ``population`` and ``retriever_score`` features come straight off each
        match, mirroring what :func:`src.rerank.dataset.build_pairs` mines.
        """
        if not matches:
            return []
        frame = pd.DataFrame(
            {
                "query": [query_text] * len(matches),
                "document": [self._descriptions.get(m.geonameid, "") for m in matches],
                "population": [m.population for m in matches],
                "retriever_score": [m.retriever_score for m in matches],
            }
        )
        pool = Pool(data=frame[_FEATURES], text_features=_TEXT_FEATURES)
        scores = self._model.predict(pool)
        order = sorted(range(len(matches)), key=lambda i: scores[i], reverse=True)
        return [(matches[i], float(scores[i])) for i in order]
