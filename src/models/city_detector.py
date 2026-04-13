"""City detection pipeline: NER → normalisation → BM25 retrieval."""
import logging

from gliner import GLiNER

from .bm25_index import GeoSearchIndex

_NER_LABELS = ["CITY", "COUNTRY", "STATE", "REGION"]

logger = logging.getLogger(__name__)


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

    def detect(
        self,
        query: str,
        top_k: int = 20,
        context_id: str | None = None,
    ) -> list[dict]:
        """
        Run the full pipeline and return up to *top_k* matching cities.

        Parameters
        ----------
        query:      Raw text in English, Turkish, or Russian.
        top_k:      Maximum number of results to return.
        context_id: Optional trace ID propagated from the request.
        """
        extra = {"context_id": context_id}

        # Stage 1 — NER
        spans = self._gliner.predict_entities(query, _NER_LABELS)
        logger.info(
            "NER stage complete",
            extra={
                **extra,
                "stage": "ner",
                "query": query,
                "spans": [
                    {
                        "text": s["text"],
                        "label": s["label"],
                        "score": round(s["score"], 4),
                        "start": s["start"],
                        "end": s["end"],
                    }
                    for s in spans
                ],
            },
        )

        if not spans:
            return []

        # Stage 2 — Normalisation
        # Deduplicate by character position: same (start, end) = same surface text,
        # possibly tagged with multiple labels. Keep the highest-scoring span per position.
        seen_positions: dict[tuple[int, int], dict] = {}
        for span in spans:
            pos = (span["start"], span["end"])
            if pos not in seen_positions or span["score"] > seen_positions[pos]["score"]:
                seen_positions[pos] = span

        # Preserve original left-to-right order
        unique_spans = sorted(seen_positions.values(), key=lambda s: s["start"])

        dropped = len(spans) - len(unique_spans)
        if dropped:
            logger.info(
                "Dropped duplicate spans",
                extra={
                    **extra,
                    "stage": "normalise",
                    "dropped_count": dropped,
                    "kept_spans": [
                        {"text": s["text"], "label": s["label"], "start": s["start"], "end": s["end"]}
                        for s in unique_spans
                    ],
                },
            )

        bm25_query = " ".join(s["text"].lower() for s in unique_spans)
        logger.info(
            "Normalisation stage complete",
            extra={**extra, "stage": "normalise", "bm25_query": bm25_query},
        )

        # Stage 3 — BM25 retrieval
        results = self._index.search(bm25_query, top_k=top_k, context_id=context_id)
        logger.info(
            "BM25 stage complete",
            extra={
                **extra,
                "stage": "bm25",
                "bm25_query": bm25_query,
                "total_results": len(results),
                "top_hits": [
                    {"name": r["ascii_name"], "score": round(r["score"], 4)}
                    for r in results[:5]
                ],
            },
        )

        return results
