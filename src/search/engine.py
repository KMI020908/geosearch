import asyncio
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

from gliner import GLiNER

from src.config import Settings
from src.ner.tokenizer import CjkAwareSplitter
from src.rerank.features import bucket_entities
from src.search.bm25 import BM25Index
from src.search.places import ParquetPlaceStore, PlaceStore

if TYPE_CHECKING:
    # Type-only: the runtime import lives inside `_load_reranker`, which is what
    # breaks the engine↔rerank import cycle. Without this the quoted annotation
    # below names nothing at all.
    from src.rerank.model import Reranker

logger = logging.getLogger(__name__)

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


@dataclass(frozen=True)
class EntitySpan:
    """One NER span, with the character offsets GLiNER reported.

    The offsets are dropped from ``entities`` (which is just the texts) but kept
    here, because a UI that highlights spans in the original text needs them and
    cannot recover them by searching — a span's surface form can occur more than
    once, and the second occurrence is not the one the model tagged.
    """

    text: str
    label: str
    start: int
    end: int


@dataclass(frozen=True)
class SearchResult:
    """Everything one :meth:`SearchEngine.search` call produced.

    A dataclass rather than a tuple because the field count grew past what a
    positional unpack reads clearly, and because ``spans`` is additive: callers
    that only want the ranked results are unaffected by its presence.
    """

    entities: list[str]
    entity_buckets: dict[str, str]
    spans: list[EntitySpan]
    ranked: list[tuple[GeonameMatch, float]]


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
        places: PlaceStore,
        settings: Settings,
        reranker: "Reranker | None" = None,
    ) -> None:
        self._ner = ner
        self._index = index
        self._corpus = corpus
        # How retrieved geonameids become displayable results.
        self._places = places
        self._settings = settings
        # None until a trained model is saved; the endpoint's use_rerank flag
        # falls back to plain retriever order while it is absent (e.g. during the
        # very first dataset-generation run, before any reranker exists).
        self._reranker = reranker

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    async def build(cls, settings: Settings) -> "SearchEngine":
        """Build the engine from the prebuilt artifacts.

        Every check on the way in *raises* rather than falling back to a
        rebuild: there is no database to rebuild from, so the honest outcome of
        a stale or half-updated directory is an actionable startup failure, not
        a silently degraded index that returns wrong results all day.
        """
        t_build = time.perf_counter()
        corpus, index, places = await cls._load_artifacts(settings)

        logger.info("Loading NER model %s…", settings.gliner_model)
        cls._check_ner_meta(settings)
        t0 = time.perf_counter()
        # `from_pretrained` is typed as returning the union of every GLiNER
        # architecture rather than the base class it dispatches from; the cast
        # names the one interface this module uses.
        ner = cast(
            GLiNER,
            await asyncio.to_thread(GLiNER.from_pretrained, settings.gliner_model),
        )
        if settings.ner_cjk_splitter:
            # Train/serve parity: the fine-tuned checkpoint was trained on this
            # tokenization and the splitter is not stored in the checkpoint, so
            # nothing but this line enforces it. See src/ner/tokenizer.py.
            # The ignore is torch's: data_processor is reached through
            # nn.Module.__getattr__, typed Tensor | Module, so any attribute on it
            # is invisible to the type checker.
            ner.data_processor.words_splitter = CjkAwareSplitter()  # pyright: ignore[reportAttributeAccessIssue, reportArgumentType]
        # Log which splitter and threshold are live: both are silent failure modes
        # (a mismatched splitter only degrades zh), so startup states them.
        logger.info(
            "NER model loaded (%.3fs) — %s splitter, threshold %.2f",
            time.perf_counter() - t0,
            "CJK-aware" if settings.ner_cjk_splitter else "GLiNER whitespace",
            settings.ner_threshold,
        )

        reranker = await cls._load_reranker(settings)

        logger.info("Engine built in %.3fs total", time.perf_counter() - t_build)
        return cls(ner, index, corpus, places, settings, reranker)

    @classmethod
    async def _load_artifacts(
        cls, settings: Settings
    ) -> tuple[list[NameGroup], BM25Index, PlaceStore]:
        """Load the corpus, index and place table from ``artifacts_dir``.

        Every check here *raises* rather than falling back to a rebuild: there
        is no database in this mode to rebuild from, so the honest outcome of a
        stale or half-updated directory is an actionable startup failure, not a
        silently degraded index that returns wrong results all day.
        """
        from src.search.artifacts import (
            artifact_set,
            check_build_ids,
            check_manifest,
            load_index,
            load_manifest,
        )

        directory = Path(settings.artifacts_dir)
        paths = artifact_set(directory)
        logger.info("Loading serving artifacts from %s…", paths.source)

        manifest = load_manifest(paths.manifest)
        check_manifest(manifest, settings)
        check_build_ids(paths, manifest)

        t0 = time.perf_counter()
        index, corpus = await asyncio.to_thread(load_index, paths.index)
        places = await asyncio.to_thread(ParquetPlaceStore.from_parquet, paths.places)
        logger.info(
            "Artifacts loaded: %d names, %d places, build %s (%.3fs)",
            len(corpus),
            len(places),
            manifest.build_id,
            time.perf_counter() - t0,
        )
        return corpus, index, places

    @classmethod
    def _check_ner_meta(cls, settings: Settings) -> None:
        """Refuse to serve a checkpoint whose splitter contract we would violate.

        ``ner_meta.json`` is written beside the weights by :mod:`src.ner.train`
        and records which word splitter the model was trained with. That is the
        one requirement a GLiNER checkpoint cannot carry itself —
        ``words_splitter_type`` in ``gliner_config.json`` names only the built-in
        kinds — and getting it wrong does not raise: the model simply tokenises
        Chinese differently than it was trained and every zh span quietly
        disappears (:mod:`src.ner.tokenizer`). Since nothing downstream notices,
        the mismatch is turned into a startup failure here.

        A checkpoint with no meta file (a zero-shot hub model, or one trained
        before this existed) is left alone — absence is not evidence of either
        setting.
        """
        from src.ner.train import CJK_SPLITTER_ID

        meta = cls._load_ner_meta(settings)
        if meta is None:
            return
        wants_cjk = meta.get("words_splitter") == CJK_SPLITTER_ID
        if wants_cjk != settings.ner_cjk_splitter:
            raise RuntimeError(
                f"{settings.gliner_model} was trained with "
                f"words_splitter={meta.get('words_splitter')!r}, but "
                f"NER_CJK_SPLITTER={settings.ner_cjk_splitter}. Serving this "
                f"combination degrades Chinese silently — set NER_CJK_SPLITTER="
                f"{str(wants_cjk).lower()} or point GLINER_MODEL at a matching "
                "checkpoint."
            )

    @staticmethod
    def _load_ner_meta(settings: Settings) -> dict | None:
        """Read ``ner_meta.json`` for the configured model, or None if there is none.

        ``gliner_model`` is either a local directory or a hub id, and the guard
        above has to work for both — a model published to the Hub
        (:mod:`src.hub.publish`) is exactly the case where nobody can glance at
        the directory to check. The hub read costs one ~600-byte cached file
        against the gigabyte of weights being fetched anyway.

        Any failure to *retrieve* the file (absent, offline, private without a
        token) is treated as "no meta", not as an error: an unavailable record is
        the pre-existing state, and refusing to boot over it would turn a missing
        cross-check into an outage.
        """
        from src.ner.train import META_FILENAME

        local = Path(settings.gliner_model) / META_FILENAME
        if local.exists():
            return json.loads(local.read_text(encoding="utf-8"))
        if Path(settings.gliner_model).is_dir():
            return None

        from huggingface_hub import hf_hub_download

        try:
            downloaded = hf_hub_download(
                repo_id=settings.gliner_model,
                filename=META_FILENAME,
                token=settings.hf_token,
            )
        except Exception as exc:  # noqa: BLE001 — any retrieval failure is "no meta"
            logger.info(
                "No %s for %s (%s) — skipping the splitter cross-check",
                META_FILENAME,
                settings.gliner_model,
                type(exc).__name__,
            )
            return None
        return json.loads(Path(downloaded).read_text(encoding="utf-8"))

    @staticmethod
    async def _load_reranker(settings: Settings) -> "Reranker | None":
        """Load the trained reranker if its checkpoint directory exists, else None.

        Descriptions are the same documents the reranker was trained on
        (:func:`src.rerank.dataset.build_descriptions`), read from
        ``descriptions.parquet`` — the same file ``make rerank-data`` mines
        against, so the documents a model is scored with are the ones it was
        fitted on. Absent model = the very first run, before any
        reranker is trained, so we fall back to population sort. Imported lazily
        to avoid an engine↔rerank import cycle.
        """
        model_path = Path(settings.rerank.model_path)
        if not model_path.exists():
            logger.info("No reranker at %s — using population sort", model_path)
            return None

        from sentence_transformers import CrossEncoder

        from src.rerank.model import Reranker

        logger.info("Loading reranker from %s…", model_path)
        t0 = time.perf_counter()
        # Any load failure (a directory that isn't a valid checkpoint, a partial
        # download, an incompatible sentence-transformers version) degrades to
        # retriever order rather than failing every request.
        try:
            model = await asyncio.to_thread(CrossEncoder, str(model_path))
        except Exception as exc:  # noqa: BLE001 — any load failure degrades, not crashes
            logger.warning("Could not load reranker — using retriever order (%s)", exc)
            return None

        from src.search.artifacts import read_descriptions

        descriptions = await read_descriptions(settings)
        logger.info(
            "Reranker loaded: %d descriptions (%.3fs)",
            len(descriptions),
            time.perf_counter() - t0,
        )
        return Reranker(model, descriptions)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def extract_spans(self, text: str) -> list[EntitySpan]:
        """Run GLiNER and return each span with its label and character offsets.

        The label (one of ``settings.ner_labels``) is kept — not dropped — so the
        reranker can bucket spans by type (:func:`src.rerank.features.bucket_entities`).

        The decision threshold comes from ``settings.ner_threshold`` (a serving
        parameter, see :class:`src.config.Settings`); ``flat_ner=True`` is GLiNER's
        default, stated explicitly because it is what :mod:`src.ner.evaluate`
        measures. The word splitter was set once in :meth:`build`.
        """
        # The ignore is torch's, as on the splitter assignment in `build`: methods
        # on an nn.Module subclass are reached through `__getattr__`, typed
        # `Tensor | Module`, so the checker cannot see `predict_entities`.
        entities = self._ner.predict_entities(  # pyright: ignore[reportCallIssue]
            text,
            self._settings.ner_labels,
            threshold=self._settings.ner_threshold,
            flat_ner=True,
        )
        return [
            EntitySpan(text=e["text"], label=e["label"], start=e["start"], end=e["end"])
            for e in entities
        ]

    async def search(
        self,
        text: str,
        top_k: int,
        use_rerank: bool = True,
    ) -> SearchResult:
        """Run the full pipeline, returning a :class:`SearchResult`.

        ``entities`` is the flat list of NER span texts, joined into
        ``query_text`` — what both BM25 retrieval and the cross-encoder reranker
        score against, and what the dataset builder mines for train/serve
        parity. ``entity_buckets`` is the same spans split by type
        (city/country/admin1); it is a display-only field on the API response,
        not a reranker input.

        Takes no database session: hydration goes through :attr:`_places`, which
        reads the prebuilt places table. That is what lets the whole pipeline
        run where no database exists.

        BM25 scores the best matching *names* for this query; every geonameid
        under a name takes that name's BM25 score as its ``retriever_score``,
        and only the top-*k* names are hydrated. Retrieval
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
        spans = await asyncio.to_thread(self.extract_spans, text)
        entities = [span.text for span in spans]
        # bucket_entities takes (text, label) pairs — the offsets are for display
        # only and are not part of what the reranker sees.
        entity_buckets = bucket_entities([(s.text, s.label) for s in spans])
        logger.info("NER: %.3fs → %s", time.perf_counter() - t0, spans)

        query_text = " ".join(entities)
        if not query_text.strip():
            logger.info(
                "Total: %.3fs (no entities found)", time.perf_counter() - t_total
            )
            return SearchResult(entities, entity_buckets, spans, [])

        t0 = time.perf_counter()
        scored_groups: list[tuple[NameGroup, float]] = await asyncio.to_thread(
            self._index.get_top_n, query_text, self._corpus, top_k
        )
        top_ids = _rank_candidates(scored_groups)
        logger.info(
            "Retrieval: %.3fs → %d names, %d candidates",
            time.perf_counter() - t0,
            len(scored_groups),
            len(top_ids),
        )
        if not top_ids:
            logger.info("Total: %.3fs (no matches)", time.perf_counter() - t_total)
            return SearchResult(entities, entity_buckets, spans, [])

        t0 = time.perf_counter()
        places = await self._places.fetch(list(top_ids))
        logger.info("Hydration: %.3fs → %d rows", time.perf_counter() - t0, len(places))

        # An id the store cannot resolve is dropped, not defaulted — the store
        # contract (src/search/places.py). `_check_coverage` in the artifact
        # builder is what keeps that from silently losing results.
        matches = [
            GeonameMatch(
                geonameid=p.geonameid,
                asciiname=p.asciiname,
                country_code=p.country_code,
                population=p.population,
                feature_code=p.feature_code,
                latitude=p.latitude,
                longitude=p.longitude,
                retriever_score=score,
            )
            for gid, score in top_ids.items()
            if (p := places.get(gid)) is not None
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
            scored = await asyncio.to_thread(self._reranker.rerank, query_text, matches)
            logger.info("Reranker: %.3fs (model)", time.perf_counter() - t0)
        else:
            # No trained reranker (or use_rerank=False): rank by the raw BM25
            # retriever_score itself, so the exposed score is exactly that.
            scored = [(m, m.retriever_score) for m in matches]
            logger.info(
                "Reranker: %.3fs (retriever-score order)", time.perf_counter() - t0
            )

        # Population still breaks ties on the final score — model ties are rare,
        # but retriever_score ties among homonyms are common on the no-rerank path.
        scored.sort(key=lambda pair: (pair[1], pair[0].population), reverse=True)

        logger.info(
            "Total: %.3fs → %d results", time.perf_counter() - t_total, len(scored)
        )
        return SearchResult(entities, entity_buckets, spans, scored)


def _rank_candidates(scored_groups: list[tuple[NameGroup, float]]) -> dict[int, float]:
    """Return every geonameid reachable from the retrieved names, with its score.

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
