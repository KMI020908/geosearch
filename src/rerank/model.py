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
        self, entities: str, matches: list[GeonameMatch]
    ) -> list[tuple[GeonameMatch, float]]:
        """Return ``(match, score)`` pairs sorted by descending model score.

        ``entities`` is the NER spans joined into one string — exactly what the
        engine passes and what :func:`src.rerank.dataset.build_pairs` mines as the
        text feature, so train- and serve-time features are identical (via the
        shared :func:`src.rerank.features.build_row`). ``retriever_rank`` is the
        candidate's position in the retrieval-ordered ``matches`` list, matching
        the rank mined at training time. Candidates without a known description
        score against an empty document rather than being dropped, so the result
        set stays the same size.
        """
        if not matches:
            return []
        rows = [
            build_row(
                entities=entities,
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
