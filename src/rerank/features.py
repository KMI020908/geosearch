"""Shared reranker feature builder — the single source of truth for both the
offline training set (:mod:`src.rerank.dataset`) and online scoring
(:mod:`src.rerank.model`).

Keeping one builder guarantees the model sees *identical* features at train and
serve time. This matters concretely here: the engine feeds the reranker the
NER **entities** (``" ".join(entities)``), not the raw query — so the training
set must featurise the same entities, not the full query text.

To change the feature set, edit the lists below and :func:`build_row`.
"""

from __future__ import annotations

import math

TEXT_FEATURES = ["entities", "document"]
NUMERIC_FEATURES = ["log_population", "retriever_score", "retriever_rank"]
FEATURE_COLUMNS = TEXT_FEATURES + NUMERIC_FEATURES


def build_row(
    entities: str,
    document: str,
    population: int,
    retriever_score: float,
    retriever_rank: int,
) -> dict[str, object]:
    """Build the feature dict for one (query, candidate) pair.

    * ``entities`` — the NER spans joined into one string (constant across a
      query's candidates; the ``document`` is what varies within a group).
    * ``document`` — the candidate's ``build_descriptions`` text.
    * ``log_population`` — ``log1p(population)``; population is the dominant
      tiebreak signal, log-scaled so trees split it cleanly.
    * ``retriever_score`` / ``retriever_rank`` — the BM25 score and 0-based
      position in the retrieved list.
    """
    return {
        "entities": entities,
        "document": document,
        "log_population": math.log1p(max(population, 0)),
        "retriever_score": float(retriever_score),
        "retriever_rank": float(retriever_rank),
    }
