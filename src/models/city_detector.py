"""City detection pipeline: NER → normalisation → hybrid retrieval (BM25 + dense)."""
import logging
from typing import TYPE_CHECKING

from gliner import GLiNER

from .bm25_index import BM25Index

if TYPE_CHECKING:
    from .biencoder_index import BiEncoderIndex

_NER_LABELS = ["CITY", "COUNTRY", "STATE", "REGION"]
_RRF_K = 60  # standard RRF constant

logger = logging.getLogger(__name__)


def _rrf_fusion(
    bm25_results: list[dict],
    biencoder_results: list[dict],
    top_k: int,
) -> list[dict]:
    """Reciprocal Rank Fusion of two ranked lists."""
    rrf_scores: dict[int, float] = {}
    all_cities: dict[int, dict] = {}

    for rank, r in enumerate(bm25_results):
        gid = r["geoname_id"]
        rrf_scores[gid] = rrf_scores.get(gid, 0.0) + 1.0 / (_RRF_K + rank)
        all_cities[gid] = r

    for rank, r in enumerate(biencoder_results):
        gid = r["geoname_id"]
        rrf_scores[gid] = rrf_scores.get(gid, 0.0) + 1.0 / (_RRF_K + rank)
        all_cities.setdefault(gid, r)

    fused = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_k]
    return [{**all_cities[gid], "score": round(score, 6)} for gid, score in fused]


class CityDetector:
    """
    City detection pipeline with optional hybrid retrieval.

    Stage 1 — NER
        GLiNER extracts location spans (city, country, state, region) from the
        raw query together with per-span confidence scores.

    Stage 2 — Normalisation
        Each span is lowercased.  The normalised spans are joined with spaces
        to form the retrieval query string.

    Stage 3 — Hybrid retrieval
    """

    def __init__(
        self,
        gliner_model: str,
        index_bm25: BM25Index,
        index_biencoder: "BiEncoderIndex",
    ) -> None:
        self._gliner = GLiNER.from_pretrained(gliner_model)
        self._index_bm25 = index_bm25
        self._index_biencoder = index_biencoder

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

        # Stage 3 — Hybrid retrieval
        # Fetch more candidates than needed so RRF has enough material to rerank
        n_candidates = top_k * 3

        bm25_results = self._index_bm25.search(bm25_query, top_k=n_candidates, context_id=context_id)
        logger.info(
            "BM25 stage complete",
            extra={
                **extra,
                "stage": "bm25",
                "bm25_query": bm25_query,
                "total_results": len(bm25_results),
                "top_hits": [
                    {"name": r["ascii_name"], "score": round(r["score"], 4)}
                    for r in bm25_results[:5]
                ],
            },
        )

        print(bm25_query)

        biencoder_results = self._index_biencoder.search(
            bm25_query, top_k=n_candidates, context_id=context_id
        )
        logger.info(
            "BiEncoder stage complete",
            extra={
                **extra,
                "stage": "biencoder",
                "total_results": len(biencoder_results),
                "top_hits": [
                    {"name": r["ascii_name"], "score": round(float(r["score"]), 4)}
                    for r in biencoder_results[:5]
                ],
            },
        )

        results = _rrf_fusion(bm25_results, biencoder_results, top_k=top_k)
        logger.info(
            "RRF fusion complete",
            extra={
                **extra,
                "stage": "rrf",
                "top_hits": [
                    {"name": r["ascii_name"], "score": r["score"]}
                    for r in results[:5]
                ],
            },
        )

        return results
