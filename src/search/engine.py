import asyncio
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from gliner import GLiNER
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings
from src.db.models import AlternateName, Geoname
from src.search.bm25 import BM25Index

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GeonameMatch:
    """A single search result."""

    geonameid: int
    asciiname: str
    country_code: str
    population: int
    feature_code: str | None
    latitude: float | None
    longitude: float | None


class SearchEngine:
    """NER + BM25 search over GeoNames populated places.

    Build once at startup via :meth:`build`, then call :meth:`search` per
    request.  Both the GLiNER model and the BM25 index are CPU-bound and
    read-only after construction, so they are safe to share across concurrent
    requests.
    """

    def __init__(
        self,
        ner: GLiNER,
        index: BM25Index,
        corpus: list[list[int]],
        settings: Settings,
    ) -> None:
        self._ner = ner
        self._index = index
        self._corpus = corpus
        self._settings = settings

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def build(
        cls, session_factory: async_sessionmaker, settings: Settings
    ) -> "SearchEngine":
        """Load corpus from the database, build the BM25 index, load NER model."""
        t_build = time.perf_counter()
        index_path = Path(settings.index_path)
        index = None
        if index_path.exists() and settings.index_warm_start:
            logger.info("Loading BM25 index from %s…", index_path)
            t0 = time.perf_counter()
            corpus_ids, index = await asyncio.to_thread(
                lambda: pickle.loads(index_path.read_bytes())
            )
            logger.info("BM25 index loaded: %d documents (%.3fs)", len(corpus_ids), time.perf_counter() - t0)
        else:
            logger.info("Loading corpus from database…")
            t0 = time.perf_counter()
            corpus_ids, documents = await cls._load_corpus(
                session_factory, settings
            )
            logger.info("Corpus loaded: %d documents (%.3fs)", len(corpus_ids), time.perf_counter() - t0)

            logger.info("Building BM25 index…")
            t0 = time.perf_counter()
            index: BM25Index = await asyncio.to_thread(BM25Index, documents)
            logger.info("BM25 index built (%.3fs) — saving to %s…", time.perf_counter() - t0, index_path)
            t0 = time.perf_counter()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                index_path.write_bytes, pickle.dumps((corpus_ids, index))
            )
            logger.info("BM25 index saved (%.3fs)", time.perf_counter() - t0)

        logger.info("Loading NER model %s…", settings.gliner_model)
        t0 = time.perf_counter()
        ner: GLiNER = await asyncio.to_thread(
            GLiNER.from_pretrained, settings.gliner_model
        )
        logger.info("NER model loaded (%.3fs)", time.perf_counter() - t0)

        logger.info("Engine built in %.3fs total", time.perf_counter() - t_build)
        return cls(ner, index, corpus_ids, settings)

    @staticmethod
    async def _load_corpus(
        session_factory: async_sessionmaker,
        settings: Settings,
    ) -> tuple[list[list[int]], list[str]]:
        """Group geonameids by name variant — one BM25 document per name.

        Each unique name (asciiname or alternate name) maps to every geonameid
        that carries it, mirroring the notebook's group_by(name).agg(geonameid).
        Tokenization into character n-grams happens inside :class:`BM25Index`,
        so documents stay as plain name strings here.
        """
        async with session_factory() as session:
            stmt = (
                select(
                    Geoname.geonameid,
                    Geoname.asciiname,
                    AlternateName.alternate_name,
                )
                .where(
                    Geoname.feature_code.not_in(settings.excluded_feature_codes)
                )
                .outerjoin(
                    AlternateName, Geoname.geonameid == AlternateName.geonameid
                )
            )
            result = await session.execute(stmt)
            rows = result.all()

        groups: dict[str, list[int]] = {}
        seen: set[tuple[str, int]] = set()

        def _add(name: str, geonameid: int) -> None:
            key = (name, geonameid)
            if key in seen:
                return
            seen.add(key)
            groups.setdefault(name, []).append(geonameid)

        for geonameid, asciiname, alternate_name in rows:
            if alternate_name:
                _add(alternate_name, geonameid)
            _add(asciiname, geonameid)

        documents = list(groups.keys())
        corpus_ids = list(groups.values())
        return corpus_ids, documents

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _extract_entities(self, text: str) -> list[str]:
        entities = self._ner.predict_entities(text, self._settings.ner_labels)
        return [e["text"] for e in entities]

    async def search(
        self,
        text: str,
        top_k: int,
        session: AsyncSession,
    ) -> tuple[list[str], list[GeonameMatch]]:
        """Run the full pipeline and return (entities, ranked matches)."""
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        entities: list[str] = await asyncio.to_thread(self._extract_entities, text)
        logger.info("NER: %.3fs → %s", time.perf_counter() - t0, entities)

        query_text = " ".join(entities)
        if not query_text.strip():
            logger.info("Total: %.3fs (no entities found)", time.perf_counter() - t_total)
            return entities, []

        t0 = time.perf_counter()
        raw_ids: list[int] = await asyncio.to_thread(
            self._index.get_top_n, query_text, self._corpus, top_k
        )
        logger.info("Retrieval: %.3fs → %d candidates", time.perf_counter() - t0, len(raw_ids))

        unique_ids = _deduplicate(raw_ids)
        if not unique_ids:
            logger.info("Total: %.3fs (no matches)", time.perf_counter() - t_total)
            return entities, []

        t0 = time.perf_counter()
        stmt = select(Geoname).where(Geoname.geonameid.in_(unique_ids))
        db_result = await session.execute(stmt)
        geonames_by_id = {g.geonameid: g for g in db_result.scalars()}
        logger.info("DB fetch: %.3fs", time.perf_counter() - t0)

        matches = [
            GeonameMatch(
                geonameid=g.geonameid,
                asciiname=g.asciiname,
                country_code=g.country_code,
                population=g.population,
                feature_code=g.feature_code,
                latitude=g.latitude,
                longitude=g.longitude,
            )
            for gid in unique_ids
            if (g := geonames_by_id.get(gid)) is not None
        ]

        t0 = time.perf_counter()
        matches = sorted(matches, key=lambda m: m.population, reverse=True)[:top_k]
        logger.info("Reranker: %.3fs", time.perf_counter() - t0)

        logger.info("Total: %.3fs → %d results", time.perf_counter() - t_total, len(matches))
        return entities, matches


def _deduplicate(ids: list[int]) -> list[int]:
    """Return unique values from *ids* in first-seen order."""
    seen: set[int] = set()
    result: list[int] = []
    for gid in ids:
        if gid not in seen:
            seen.add(gid)
            result.append(gid)
    return result
