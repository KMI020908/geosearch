"""City detection pipeline: NER → normalisation → BM25 retrieval."""
from gliner import GLiNER

from .bm25_index import GeoSearchIndex

_NER_LABELS = ["CITY", "COUNTRY", "STATE", "REGION"]


class CityDetector:
    """
    Three-stage city detection pipeline.

    Stage 1 — NER
        GLiNER extracts location spans (city, country, state, region) from the
        raw query together with per-span confidence scores.

    Stage 2 — Normalisation
        Each span is lowercased.  The normalised spans are joined with spaces
        to form the BM25 query string.

    Stage 3 — BM25 retrieval
        The query is searched against a :class:`GeoSearchIndex` that stores one
        document per city containing all multilingual name variants plus country
        and region names.  Results are returned ranked by BM25 score.
    """

    def __init__(self, gliner_model: str, index: GeoSearchIndex) -> None:
        self._gliner = GLiNER.from_pretrained(gliner_model)
        self._index = index

    def detect(self, query: str, top_k: int = 20) -> list[dict]:
        """
        Run the full pipeline and return up to *top_k* matching cities.

        Parameters
        ----------
        query:  Raw text in English, Turkish, or Russian.
        top_k:  Maximum number of results to return.

        Returns
        -------
        List of city dicts as produced by :meth:`GeoSearchIndex.search`.
        """
        spans = self._gliner.predict_entities(query, _NER_LABELS)
        if not spans:
            return []

        bm25_query = " ".join(span["text"].lower() for span in spans)
        return self._index.search(bm25_query, top_k=top_k)
