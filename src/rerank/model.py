"""Inference wrapper around the fine-tuned cross-encoder reranker.

Kept separate from :mod:`src.rerank.train` so the search engine can load and
score without pulling in the training entry point. The document text for each
candidate is the same ``build_descriptions`` string used at training time, so
the model sees an identical document at both stages.
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder

from src.search.engine import GeonameMatch


class Reranker:
    """Score ``(query_text, document)`` pairs with a fine-tuned cross-encoder.

    ``query_text`` is the same string BM25 retrieval scores against
    (``" ".join(entities)`` in :meth:`~src.search.engine.SearchEngine.search`)
    — the flat NER span texts, not the typed ``entity_buckets`` split used only
    for display. The model attends jointly over both texts, so no hand-computed
    overlap features are needed the way the previous CatBoost model needed them.

    ``predict()`` returns an unbounded raw logit per pair (verified against a
    trained checkpoint — ``BinaryCrossEntropyLoss`` trains on raw logits and
    ``CrossEncoderTrainer`` persists that as the model's default predict-time
    activation, so no sigmoid is applied on load), not a bounded probability.
    Only relative order matters here: sort order and the population tie-break
    downstream in ``engine.py`` work on any monotonic score.
    """

    def __init__(self, model: CrossEncoder, descriptions: dict[int, str]) -> None:
        self._model = model
        self._descriptions = descriptions

    @classmethod
    def load(cls, model_path: str, descriptions: dict[int, str]) -> Reranker:
        """Load the checkpoint directory and pair it with the candidate descriptions."""
        return cls(CrossEncoder(model_path), descriptions)

    def rerank(
        self, query_text: str, matches: list[GeonameMatch]
    ) -> list[tuple[GeonameMatch, float]]:
        """Return ``(match, score)`` pairs sorted by descending model score.

        Candidates without a known description score against an empty document
        rather than being dropped, so the result set stays the same size.
        """
        if not matches:
            return []
        pairs = [(query_text, self._descriptions.get(m.geonameid, "")) for m in matches]
        scores = self._model.predict(pairs)
        order = sorted(range(len(matches)), key=lambda i: scores[i], reverse=True)
        return [(matches[i], float(scores[i])) for i in order]
