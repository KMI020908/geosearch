"""Unit test for the reranker's sort-by-score contract — no real model load."""

from src.rerank.model import Reranker
from src.search.engine import GeonameMatch


class _StubCrossEncoder:
    """A fake with a ``.predict()`` returning fixed scores, in call order."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def predict(self, pairs: list[tuple[str, str]]):
        assert len(pairs) == len(self._scores)
        return self._scores


def _match(geonameid: int) -> GeonameMatch:
    return GeonameMatch(
        geonameid=geonameid,
        asciiname=f"place-{geonameid}",
        country_code="XX",
        population=0,
        feature_code=None,
        latitude=None,
        longitude=None,
        retriever_score=0.0,
    )


def test_rerank_sorts_by_descending_model_score():
    matches = [_match(1), _match(2), _match(3)]
    model = _StubCrossEncoder(scores=[0.1, 0.9, 0.5])
    reranker = Reranker(model, descriptions={1: "doc-1", 2: "doc-2", 3: "doc-3"})

    scored = reranker.rerank("query", matches)

    assert [m.geonameid for m, _ in scored] == [2, 3, 1]
    assert [round(s, 2) for _, s in scored] == [0.9, 0.5, 0.1]


def test_rerank_scores_missing_descriptions_as_empty_string():
    matches = [_match(1)]
    model = _StubCrossEncoder(scores=[0.7])
    reranker = Reranker(model, descriptions={})

    scored = reranker.rerank("query", matches)

    assert len(scored) == 1
    assert scored[0][0].geonameid == 1


def test_rerank_returns_empty_list_for_no_matches():
    reranker = Reranker(_StubCrossEncoder(scores=[]), descriptions={})
    assert reranker.rerank("query", []) == []
