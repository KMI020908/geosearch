import asyncio
import logging
import pickle
import time
from dataclasses import dataclass
from pathlib import Path

from gliner import GLiNER
from sqlalchemy import select, union
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from src.config import Settings
from src.db.models import AlternateName, Geoname
from src.search.bm25 import BM25Index

logger = logging.getLogger(__name__)

# Bumped when the pickled index layout changes, so a stale file on disk is
# rebuilt instead of failing at query time.
_INDEX_FORMAT = 5

# One BM25 document: a name and every (geonameid, population) it resolves to.
# Population rides along so each geonameid's own population is available for
# tie-breaking later; BM25 relevance itself is per-query (computed by
# BM25Index.get_top_n at search time), so no score is stored here.
NameGroup = tuple[tuple[int, int], ...]


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
    retriever_score: float


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
        corpus: list[NameGroup],
        settings: Settings,
        reranker: "Reranker | None" = None,
    ) -> None:
        self._ner = ner
        self._index = index
        self._corpus = corpus
        self._settings = settings
        # None until a trained model is saved; the endpoint's use_rerank flag
        # falls back to plain retriever order while it is absent (e.g. during the
        # very first dataset-generation run, before any reranker exists).
        self._reranker = reranker

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
        cached = (
            await cls._load_cached_index(index_path)
            if index_path.exists() and settings.index_warm_start
            else None
        )
        if cached is not None:
            corpus, index = cached
        else:
            logger.info("Loading corpus from database…")
            t0 = time.perf_counter()
            corpus, documents = await cls._load_corpus(session_factory, settings)
            logger.info("Corpus loaded: %d documents (%.3fs)", len(corpus), time.perf_counter() - t0)

            logger.info("Building BM25 index…")
            t0 = time.perf_counter()
            index: BM25Index = await asyncio.to_thread(BM25Index, documents)
            logger.info("BM25 index built (%.3fs) — saving to %s…", time.perf_counter() - t0, index_path)
            t0 = time.perf_counter()
            index_path.parent.mkdir(parents=True, exist_ok=True)
            await asyncio.to_thread(
                index_path.write_bytes, pickle.dumps((_INDEX_FORMAT, corpus, index))
            )
            logger.info("BM25 index saved (%.3fs)", time.perf_counter() - t0)

        logger.info("Loading NER model %s…", settings.gliner_model)
        t0 = time.perf_counter()
        ner: GLiNER = await asyncio.to_thread(
            GLiNER.from_pretrained, settings.gliner_model
        )
        logger.info("NER model loaded (%.3fs)", time.perf_counter() - t0)

        reranker = await cls._load_reranker(session_factory, settings)

        logger.info("Engine built in %.3fs total", time.perf_counter() - t_build)
        return cls(ner, index, corpus, settings, reranker)

    @staticmethod
    async def _load_cached_index(
        index_path: Path,
    ) -> tuple[list[NameGroup], BM25Index] | None:
        """Return the pickled (corpus, index), or None if it predates _INDEX_FORMAT.

        A payload written under an older corpus layout isn't guaranteed to match
        the current NameGroup shape, so it's rebuilt rather than trusted.
        """
        logger.info("Loading BM25 index from %s…", index_path)
        t0 = time.perf_counter()
        payload = await asyncio.to_thread(lambda: pickle.loads(index_path.read_bytes()))
        if not (isinstance(payload, tuple) and payload[:1] == (_INDEX_FORMAT,)):
            logger.info("Index at %s is stale — rebuilding", index_path)
            return None
        _, corpus, index = payload
        logger.info(
            "BM25 index loaded: %d documents (%.3fs)", len(corpus), time.perf_counter() - t0
        )
        return corpus, index

    @staticmethod
    async def _load_reranker(
        session_factory: async_sessionmaker, settings: Settings
    ) -> "Reranker | None":
        """Load the trained reranker if its model file exists, else return None.

        Descriptions are the same documents the reranker was trained on
        (:func:`src.rerank.dataset.build_descriptions`). Absent model = the very
        first run, before any reranker is trained, so we fall back to population
        sort. Imported lazily to avoid an engine↔rerank import cycle.
        """
        model_path = Path(settings.rerank.model_path)
        if not model_path.exists():
            logger.info("No reranker at %s — using population sort", model_path)
            return None

        from src.dataset.sampling import load_admin1_names, load_name_rows
        from src.rerank.dataset import build_descriptions
        from src.rerank.model import Reranker

        logger.info("Loading reranker from %s…", model_path)
        t0 = time.perf_counter()
        name_rows = await load_name_rows(session_factory, settings)
        admin1_names = load_admin1_names(settings.geonames_data_dir, settings.countries)
        descriptions = build_descriptions(name_rows, admin1_names)
        reranker = await asyncio.to_thread(
            Reranker.load, str(model_path), descriptions
        )
        logger.info("Reranker loaded (%.3fs)", time.perf_counter() - t0)
        return reranker

    @staticmethod
    async def _load_corpus(
        session_factory: async_sessionmaker,
        settings: Settings,
    ) -> tuple[list[NameGroup], list[str]]:
        """Group geonameids by name variant — one BM25 document per name.

        Each unique name (name, asciiname or alternate name) maps to every
        geonameid that carries it, mirroring the notebook's
        group_by(name).agg(geonameid).  Population rides along on the same union
        so each geonameid's own population is available for tie-breaking later;
        the group's BM25 score itself is computed per query, not here.
        Tokenization into character n-grams happens inside :class:`BM25Index`,
        so documents stay as plain name strings.
        """

        async with session_factory() as session:
            g_filt = [
                Geoname.country_code.in_(settings.countries),
                Geoname.feature_code.not_in(settings.excluded_feature_codes)
            ]

            q1 = select(
                Geoname.geonameid, Geoname.name.label('name'), Geoname.population
            ).where(*g_filt)

            q2 = select(
                Geoname.geonameid, Geoname.asciiname.label('name'), Geoname.population
            ).where(*g_filt)

            q3 = (
                select(
                    Geoname.geonameid,
                    AlternateName.alternate_name.label('name'),
                    Geoname.population,
                )
                .where(*g_filt, AlternateName.isolanguage.in_(settings.languages))
                .outerjoin(AlternateName, Geoname.geonameid == AlternateName.geonameid)
            )

            stmt = union(q1, q2, q3)

            result = await session.execute(stmt)
            rows = result.all()

        groups: dict[str, list[tuple[int, int]]] = {}
        for geonameid, name, population in rows:
            groups.setdefault(name, []).append((geonameid, population or 0))

        documents = list(groups)
        corpus = [tuple(ids) for ids in groups.values()]
        return corpus, documents

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
        use_rerank: bool = True,
    ) -> tuple[list[str], list[tuple[GeonameMatch, float]]]:
        """Run the full pipeline and return (entities, ranked (match, score) pairs).

        BM25 scores the best matching *names* for this query; every geonameid
        under a name takes that name's BM25 score as its ``retriever_score``,
        and only the top-*k* names are hydrated from the database. Retrieval
        then *ends* by sorting the full candidate set by ``retriever_score``
        (population breaking homonym ties) and cutting to top-*k* — so retrieval,
        not the reranker, fixes which candidates survive. The reranker is a pure
        reordering of those exactly-top-*k* survivors: with ``use_rerank`` and a
        loaded reranker the final order is the model's score, otherwise it stays
        ``retriever_score`` order; population breaks ties on the final score.
        The returned score is the reranker score when reranking, else the
        ``retriever_score``. Dataset generation calls with ``use_rerank=False``
        to mine candidates from raw retrieval, independent of any reranker.
        """
        t_total = time.perf_counter()

        t0 = time.perf_counter()
        entities: list[str] = await asyncio.to_thread(self._extract_entities, text)
        logger.info("NER: %.3fs → %s", time.perf_counter() - t0, entities)

        query_text = " ".join(entities)
        if not query_text.strip():
            logger.info("Total: %.3fs (no entities found)", time.perf_counter() - t_total)
            return entities, []

        t0 = time.perf_counter()
        scored_groups: list[tuple[NameGroup, float]] = await asyncio.to_thread(
            self._index.get_top_n, query_text, self._corpus, top_k
        )
        top_ids = _rank_candidates(scored_groups)
        logger.info(
            "Retrieval: %.3fs → %d names, %d candidates",
            time.perf_counter() - t0, len(scored_groups), len(top_ids),
        )
        if not top_ids:
            logger.info("Total: %.3fs (no matches)", time.perf_counter() - t_total)
            return entities, []

        t0 = time.perf_counter()
        stmt = select(Geoname).where(Geoname.geonameid.in_(list(top_ids)))
        db_result = await session.execute(stmt)
        geonames_by_id = {g.geonameid: g for g in db_result.scalars()}
        logger.info("DB fetch: %.3fs → %d rows", time.perf_counter() - t0, len(geonames_by_id))

        matches = [
            GeonameMatch(
                geonameid=g.geonameid,
                asciiname=g.asciiname,
                country_code=g.country_code,
                population=g.population,
                feature_code=g.feature_code,
                latitude=g.latitude,
                longitude=g.longitude,
                retriever_score=score,
            )
            for gid, score in top_ids.items()
            if (g := geonames_by_id.get(gid)) is not None
        ]

        # --- Retrieval stage ends here ---------------------------------------
        # Rank the full candidate set by retriever_score, break homonym ties by
        # population, then cut to top_k. Only these survivors reach the reranker,
        # so retrieval — not the reranker — fixes which candidates make top_k.
        matches.sort(key=lambda m: (m.retriever_score, m.population), reverse=True)
        matches = matches[:top_k]

        # --- Rerank stage: pure reordering of the retrieved top_k ------------
        t0 = time.perf_counter()
        if use_rerank and self._reranker is not None:
            scored = await asyncio.to_thread(
                self._reranker.rerank, query_text, matches
            )
            logger.info("Reranker: %.3fs (model)", time.perf_counter() - t0)
        else:
            # No trained reranker (or use_rerank=False): rank by the raw BM25
            # retriever_score itself, so the exposed score is exactly that.
            scored = [(m, m.retriever_score) for m in matches]
            logger.info("Reranker: %.3fs (retriever-score order)", time.perf_counter() - t0)

        # Population still breaks ties on the final score — model ties are rare,
        # but retriever_score ties among homonyms are common on the no-rerank path.
        scored.sort(key=lambda pair: (pair[1], pair[0].population), reverse=True)

        logger.info("Total: %.3fs → %d results", time.perf_counter() - t_total, len(scored))
        return entities, scored


def _rank_candidates(scored_groups: list[tuple[NameGroup, float]]) -> dict[int, float]:
    """Return every geonameid reachable from the retrieved names, with its BM25 retriever score.

    Every geonameid under a retrieved name takes that name's BM25 score, so
    homonyms tie and rank together; a geonameid reachable under several names
    (e.g. both "Moscow" and "Moskva") keeps the best of them.  This does *not*
    truncate to top-k: the retrieved name-groups are already limited to top-k
    by BM25 (:meth:`BM25Index.get_top_n`), but each can hold more than one
    geonameid, so the candidate set can exceed top-k here — it's
    :meth:`SearchEngine.search` that decides which top-k survive, by
    ``retriever_score`` then population, *before* the reranker only reorders
    those survivors. Truncating here would let BM25's uniform-IDF TF score — a
    fuzzy-matching signal, not a disambiguator — silently drop the right homonym
    before population ever gets a say.
    """
    scores: dict[int, float] = {}
    for group, score in scored_groups:
        for gid, _population in group:
            current = scores.get(gid)
            if current is None or score > current:
                scores[gid] = score
    return scores


if __name__ == "__main__":
    # the user's Moscow example — homonyms tie on the group's BM25 score
    # (supplied per query, not stored on the group); which one wins is decided
    # by population in SearchEngine.search, not by _rank_candidates.
    moscow: NameGroup = ((2, 200), (1, 1000))
    hamlet: NameGroup = ((3, 5),)
    scored_groups = [(hamlet, 0.5), (moscow, 2.0)]
    scores = _rank_candidates(scored_groups)
    assert scores == {2: 2.0, 1: 2.0, 3: 0.5}, scores
    print("ok", scores)
