"""Build the serving artifacts, so the engine does not need a database.

``SearchEngine`` used to reach a database twice per startup and once per
*request*: the BM25 corpus came from a SQL union, the reranker's candidate
documents from another, and every result was hydrated with
``select(Geoname).where(...)`` because the index stores only
``(geonameid, population)`` pairs. That is fine on a machine running the
database, and impossible on a HuggingFace Space, which has none.

This module is the seam. It runs those projections once, offline, and writes
what serving actually reads:

``bm25_index.npz``
    The retrieval index (:mod:`src.search.bm25`) plus the corpus — the
    ``NameGroup`` per document — flattened into plain arrays.
``places.parquet``
    The hydration table: one row per in-scope place, holding exactly the fields
    :class:`~src.search.engine.GeonameMatch` needs.
``descriptions.parquet``
    ``build_descriptions`` output, the documents the reranker scores against.
``manifest.json``
    What the other three are, and whether they belong together.

**The manifest's ``build_id`` is the load-bearing part.** The three files are
only meaningful as a set: a places table missing a geonameid the corpus
references does not error, it silently drops that place from every result it
should have won. ``_INDEX_FORMAT`` already guards the *layout* of one file
against drift; ``build_id`` generalises exactly that stance to the set, and
:func:`check_manifest` refuses to start on a mismatch rather than serving
quietly-wrong results.

The staging corpus (:mod:`src.corpus`) is the authoring path: ``make etl``
fills it, and ``make artifacts`` compiles it into the files below.

Run as::

    python -m src.search.artifacts
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import time
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING

import polars as pl
from pydantic import BaseModel

from src.config import Settings, settings
from src.search.bm25 import INDEX_FORMAT, BM25Index

if TYPE_CHECKING:
    # Type-only. This module is imported by the engine on the `artifacts` path,
    # so a module-scope SQLAlchemy import here would put the dependency straight
    # back onto the serving path — which is exactly what the Space cannot have.
    # `build()` needs it at runtime, but only inside its own body.
    pass

logger = logging.getLogger(__name__)

# Bumped when the *set* of files, or the schema of one of them, changes in a way
# that a previous build would not satisfy. Distinct from `INDEX_FORMAT`, which
# versions only the index file's array layout.
ARTIFACT_FORMAT = 1

INDEX_FILE = "bm25_index.npz"
PLACES_FILE = "places.parquet"
DESCRIPTIONS_FILE = "descriptions.parquet"
MANIFEST_FILE = "manifest.json"

SERVING_FILES = (INDEX_FILE, PLACES_FILE, DESCRIPTIONS_FILE, MANIFEST_FILE)

# Exactly the fields `GeonameMatch` carries, so hydration is a rename and not a
# join against anything else.
PLACES_SCHEMA: dict[str, pl.DataType] = {
    "geonameid": pl.Int64(),
    "asciiname": pl.String(),
    "country_code": pl.String(),
    "population": pl.Int64(),
    "feature_code": pl.String(),
    "latitude": pl.Float64(),
    "longitude": pl.Float64(),
}

DESCRIPTIONS_SCHEMA: dict[str, pl.DataType] = {
    "geonameid": pl.Int64(),
    "description": pl.String(),
}


class StaleArtifactsError(RuntimeError):
    """The artifacts on disk cannot be served by this build of the code."""


class Manifest(BaseModel):
    """What one ``make artifacts`` run produced, and under what scope.

    ``countries`` / ``languages`` / ``excluded_feature_codes`` are recorded
    because they decide *what is in* the corpus, not merely how it is stored: an
    index built for RU+US served under a config naming RU+US+TR+CN silently
    cannot return a Turkish city, and nothing downstream would notice.
    """

    artifact_format: int
    index_format: int
    build_id: str
    built_at: str

    countries: list[str]
    languages: list[str]
    excluded_feature_codes: list[str]

    n_documents: int
    n_places: int
    n_descriptions: int

    sha256: dict[str, str]


@dataclass(frozen=True)
class ArtifactSet:
    """Resolved paths to one coherent set of serving artifacts."""

    index: Path
    places: Path
    descriptions: Path
    manifest: Path
    source: str  # "local:<dir>" or "hub:<repo>@<sha>"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_set(directory: Path, *, source: str | None = None) -> ArtifactSet:
    """Name the four files inside *directory* without checking they exist."""
    return ArtifactSet(
        index=directory / INDEX_FILE,
        places=directory / PLACES_FILE,
        descriptions=directory / DESCRIPTIONS_FILE,
        manifest=directory / MANIFEST_FILE,
        source=source or f"local:{directory}",
    )


def load_manifest(path: Path) -> Manifest:
    """Read and validate a manifest written by :func:`build`."""
    if not path.exists():
        raise StaleArtifactsError(
            f"{path} is missing — build the artifacts with `make artifacts`, "
            "or fetch them with `make hub-pull`."
        )
    return Manifest.model_validate_json(path.read_text(encoding="utf-8"))


def check_manifest(manifest: Manifest, settings: Settings) -> None:
    """Refuse to serve artifacts this build cannot correctly interpret.

    Raises rather than rebuilding: in ``artifacts`` mode there is no database to
    rebuild *from*, so the honest outcome is an actionable failure at startup
    instead of a silently degraded index at query time.
    """
    if manifest.artifact_format != ARTIFACT_FORMAT:
        raise StaleArtifactsError(
            f"Artifacts are format {manifest.artifact_format}, this build expects "
            f"{ARTIFACT_FORMAT} — rebuild with `make artifacts`."
        )
    if manifest.index_format != INDEX_FORMAT:
        raise StaleArtifactsError(
            f"Index inside the artifacts is format {manifest.index_format}, this "
            f"build expects {INDEX_FORMAT} — rebuild with `make artifacts`."
        )

    scope = {
        "countries": (sorted(manifest.countries), sorted(settings.countries)),
        "languages": (sorted(manifest.languages), sorted(settings.languages)),
        "excluded_feature_codes": (
            sorted(manifest.excluded_feature_codes),
            sorted(settings.excluded_feature_codes),
        ),
    }
    for field, (built, configured) in scope.items():
        if built != configured:
            raise StaleArtifactsError(
                f"Artifacts were built with {field}={built}, but this process is "
                f"configured for {configured}. The corpus does not contain what "
                "the configuration claims — rebuild with `make artifacts`."
            )


# ---------------------------------------------------------------------------
# Building
# ---------------------------------------------------------------------------


async def build(settings: Settings, directory: Path | None = None) -> ArtifactSet:
    """Compile the staging corpus into the files serving reads.

    The three projections come from :mod:`src.search.corpus_queries`, which is
    also what the dataset sampler reads, so the corpus the index is built over
    and the corpus queries are generated from cannot diverge.
    """
    # Imported here, not at module scope: the staging corpus exists only on a
    # machine that has run `make etl`, while this module is imported by every
    # process that *serves* the artifacts and must not require it.
    from src.search import corpus_queries

    directory = directory or Path(settings.artifacts_dir)
    directory.mkdir(parents=True, exist_ok=True)
    build_id = str(uuid.uuid4())
    t_build = time.perf_counter()

    logger.info("Loading corpus…")
    t0 = time.perf_counter()
    corpus, documents = await asyncio.to_thread(corpus_queries.bm25_corpus, settings)
    logger.info(
        "Corpus loaded: %d names (%.1fs)", len(documents), time.perf_counter() - t0
    )

    logger.info("Building BM25 index over %d names…", len(documents))
    t0 = time.perf_counter()
    index = await asyncio.to_thread(BM25Index, documents)
    logger.info("BM25 index built (%.1fs)", time.perf_counter() - t0)

    logger.info("Loading places…")
    places = await asyncio.to_thread(corpus_queries.places, settings)

    logger.info("Building candidate descriptions…")
    descriptions = await asyncio.to_thread(corpus_queries.descriptions, settings)

    _check_coverage(corpus, places)

    paths = artifact_set(directory)
    logger.info("Writing artifacts to %s…", directory)
    await asyncio.to_thread(_write_index, index, corpus, paths.index)
    _write_table(places, paths.places, build_id)
    _write_table(descriptions, paths.descriptions, build_id)

    manifest = Manifest(
        artifact_format=ARTIFACT_FORMAT,
        index_format=INDEX_FORMAT,
        build_id=build_id,
        built_at=datetime.now(UTC).isoformat(timespec="seconds"),
        countries=list(settings.countries),
        languages=list(settings.languages),
        excluded_feature_codes=list(settings.excluded_feature_codes),
        n_documents=len(documents),
        n_places=places.height,
        n_descriptions=descriptions.height,
        sha256={
            name: _sha256(directory / name)
            for name in (INDEX_FILE, PLACES_FILE, DESCRIPTIONS_FILE)
        },
    )
    paths.manifest.write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    total_mb = sum((directory / n).stat().st_size for n in SERVING_FILES) / 1e6
    logger.info(
        "Artifacts built in %.1fs — %d names, %d places, %d descriptions, %.1f MB",
        time.perf_counter() - t_build,
        len(documents),
        places.height,
        descriptions.height,
        total_mb,
    )
    logger.info("build_id %s — publish with `make hub-push-index`", build_id)
    return paths


def _check_coverage(
    corpus: list[tuple[tuple[int, int], ...]], places: pl.DataFrame
) -> None:
    """Every geonameid the corpus can retrieve must be hydratable.

    A missing row does not raise at query time — the match is simply dropped
    (``engine.search`` skips ids with no row) — so the place would vanish from
    results it should have won, with nothing logged. Checked here, once, where
    it is still cheap to fix.
    """
    referenced = {geonameid for group in corpus for geonameid, _ in group}
    known = set(places["geonameid"].to_list())
    missing = referenced - known
    if missing:
        sample = sorted(missing)[:5]
        raise RuntimeError(
            f"{len(missing)} geonameid(s) are retrievable but absent from "
            f"{PLACES_FILE} (e.g. {sample}). The two queries disagree on scope; "
            "results for these places would be silently dropped."
        )


def _write_index(
    index: BM25Index, corpus: list[tuple[tuple[int, int], ...]], path: Path
) -> None:
    """Save the index, then append the corpus to the same ``.npz``.

    The corpus is a ragged ``list[NameGroup]``, which numpy cannot hold as one
    array without ``dtype=object`` — and an object array is pickle again, which
    is the whole thing this format exists to avoid. So it is flattened CSR-style
    into offsets plus two value arrays, exactly as the postings are.
    """
    import numpy as np

    index.save(path)

    offsets = np.zeros(len(corpus) + 1, dtype=np.int64)
    geonameids: list[int] = []
    populations: list[int] = []
    for i, group in enumerate(corpus):
        offsets[i + 1] = offsets[i] + len(group)
        geonameids.extend(geonameid for geonameid, _ in group)
        populations.extend(population for _, population in group)

    with np.load(path, allow_pickle=False) as payload:
        fields = dict(payload)
    fields["group_offsets"] = offsets
    fields["group_geonameids"] = np.asarray(geonameids, dtype=np.int64)
    fields["group_populations"] = np.asarray(populations, dtype=np.int64)
    np.savez_compressed(path, **fields)


def load_index(path: Path) -> tuple[BM25Index, list[tuple[tuple[int, int], ...]]]:
    """Read back what :func:`_write_index` wrote: the index and its corpus."""
    import numpy as np

    index = BM25Index.load(path)
    with np.load(path, allow_pickle=False) as payload:
        offsets = payload["group_offsets"]
        geonameids = payload["group_geonameids"]
        populations = payload["group_populations"]

    corpus = [
        tuple(
            zip(
                geonameids[start:end].tolist(),
                populations[start:end].tolist(),
                strict=True,
            )
        )
        for start, end in zip(offsets[:-1], offsets[1:], strict=True)
    ]
    return index, corpus


def _write_table(frame: pl.DataFrame, path: Path, build_id: str) -> None:
    """Write *frame* to Parquet, stamping the build it belongs to.

    The stamp rides in Parquet's key-value metadata rather than as a column: it
    describes the file, not any row, and a constant column over a million rows
    is a waste even after compression.
    """
    frame.write_parquet(path, compression="zstd", metadata={"build_id": build_id})


async def read_descriptions(settings: Settings) -> dict[int, str]:
    """The reranker's candidate documents, read from the artifact.

    Shared by the engine and by ``make rerank-data`` so the documents a model is
    *mined against* and the ones it is *served* with are the same file, not two
    derivations that happen to agree.
    """
    path = artifact_set(Path(settings.artifacts_dir)).descriptions
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing, so there are no documents to score — run "
            "`make artifacts` or `make hub-pull`."
        )
    frame = await asyncio.to_thread(pl.read_parquet, path)
    return dict(
        zip(frame["geonameid"].to_list(), frame["description"].to_list(), strict=True)
    )


def read_build_id(path: Path) -> str | None:
    """Return the ``build_id`` stamped into a Parquet file, if any.

    Read from the *file* metadata, not the schema's. Polars' ``metadata=``
    argument writes to the footer's key-value block, which ``read_schema`` does
    not surface — it reports only the Arrow schema's own metadata, which stays
    empty. Reading the wrong one yields ``None`` for a correctly stamped file,
    i.e. a spurious "these are not one set" failure.
    """
    import pyarrow.parquet as pq

    metadata = pq.read_metadata(path).metadata or {}
    value = metadata.get(b"build_id")
    return value.decode() if value else None


def check_build_ids(paths: ArtifactSet, manifest: Manifest) -> None:
    """Every table must come from the run the manifest describes.

    This is the check that catches a half-updated directory — a fresh index
    beside yesterday's places table, which is exactly the failure that produces
    silently-missing results rather than an error.
    """
    for path in (paths.places, paths.descriptions):
        stamped = read_build_id(path)
        if stamped != manifest.build_id:
            raise StaleArtifactsError(
                f"{path.name} is from build {stamped}, but the manifest describes "
                f"{manifest.build_id}. These files are not one set — rebuild with "
                "`make artifacts` or re-fetch with `make hub-pull`."
            )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s — %(message)s"
    )
    asyncio.run(build(settings))


if __name__ == "__main__":
    main()
