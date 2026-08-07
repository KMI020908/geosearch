"""Artifacts are only valid as a *set*, and the checks must say so.

Every failure mode here is silent if unchecked: a half-updated directory does
not error, it drops results. These tests pin the checks that turn each one into
a startup failure instead.
"""

import json

import numpy as np
import polars as pl
import pytest

from src.config import Settings
from src.search.artifacts import (
    ARTIFACT_FORMAT,
    DESCRIPTIONS_SCHEMA,
    PLACES_SCHEMA,
    Manifest,
    StaleArtifactsError,
    _write_table,
    artifact_set,
    check_build_ids,
    check_manifest,
    load_index,
    load_manifest,
    read_build_id,
)
from src.search.bm25 import INDEX_FORMAT, BM25Index

BUILD_ID = "11111111-2222-3333-4444-555555555555"
DOCUMENTS = ["Moscow", "Moskva", "Kazan"]
CORPUS = [((524901, 10381222), (5601538, 302)), ((524901, 10381222),), ((551487, 1),)]


def _settings(**overrides) -> Settings:
    return Settings(
        countries=["RU", "US", "TR", "CN"],
        languages=["ru", "en", "tr", "zh"],
        excluded_feature_codes=["PPLH", "PPLQ", "PPLW", "PPLX"],
        **overrides,
    )


def _manifest(**overrides) -> Manifest:
    base = dict(
        artifact_format=ARTIFACT_FORMAT,
        index_format=INDEX_FORMAT,
        build_id=BUILD_ID,
        built_at="2026-01-01T00:00:00+00:00",
        countries=["RU", "US", "TR", "CN"],
        languages=["ru", "en", "tr", "zh"],
        excluded_feature_codes=["PPLH", "PPLQ", "PPLW", "PPLX"],
        n_documents=len(DOCUMENTS),
        n_places=3,
        n_descriptions=3,
        sha256={},
    )
    return Manifest(**{**base, **overrides})


@pytest.fixture
def directory(tmp_path):
    """A complete, self-consistent artifact directory."""
    from src.search.artifacts import _write_index

    paths = artifact_set(tmp_path)
    _write_index(BM25Index(DOCUMENTS), CORPUS, paths.index)
    _write_table(
        pl.DataFrame(
            [(524901, "Moscow", "RU", 10381222, "PPLC", 55.7, 37.6)],
            schema=PLACES_SCHEMA,
            orient="row",
        ),
        paths.places,
        BUILD_ID,
    )
    _write_table(
        pl.DataFrame(
            [(524901, "Moscow | Москва\nRussia\nMoscow")],
            schema=DESCRIPTIONS_SCHEMA,
            orient="row",
        ),
        paths.descriptions,
        BUILD_ID,
    )
    paths.manifest.write_text(_manifest().model_dump_json(indent=2), encoding="utf-8")
    return tmp_path


# --- the corpus round-trip -------------------------------------------------


def test_corpus_round_trips_without_pickle(directory) -> None:
    """The ragged NameGroup list survives being flattened into plain arrays."""
    paths = artifact_set(directory)
    index, corpus = load_index(paths.index)
    assert corpus == CORPUS
    assert index.corpus_size == len(DOCUMENTS)
    # The whole point of the format: no pickle anywhere in it.
    with np.load(paths.index, allow_pickle=False) as payload:
        assert "group_offsets" in payload


# --- manifest checks -------------------------------------------------------


def test_complete_directory_passes(directory) -> None:
    paths = artifact_set(directory)
    manifest = load_manifest(paths.manifest)
    check_manifest(manifest, _settings())
    check_build_ids(paths, manifest)


def test_missing_manifest_names_the_fix(tmp_path) -> None:
    with pytest.raises(StaleArtifactsError, match="make artifacts"):
        load_manifest(artifact_set(tmp_path).manifest)


@pytest.mark.parametrize(
    "field,value",
    [("artifact_format", ARTIFACT_FORMAT + 1), ("index_format", INDEX_FORMAT + 1)],
)
def test_stale_format_is_rejected(field: str, value: int) -> None:
    with pytest.raises(StaleArtifactsError, match="make artifacts"):
        check_manifest(_manifest(**{field: value}), _settings())


@pytest.mark.parametrize(
    "field,value",
    [
        ("countries", ["RU", "US"]),
        ("languages", ["ru", "en"]),
        ("excluded_feature_codes", []),
    ],
)
def test_scope_drift_is_rejected(field: str, value: list[str]) -> None:
    """Config claiming a scope the corpus does not contain must not start.

    An index built for RU+US, served under a config naming four countries,
    cannot return a Turkish city — and nothing downstream would notice.
    """
    with pytest.raises(StaleArtifactsError, match=field):
        check_manifest(_manifest(**{field: value}), _settings())


def test_scope_comparison_ignores_ordering() -> None:
    """The scope is a set; a reordered env var is not a rebuild."""
    check_manifest(_manifest(countries=["CN", "TR", "US", "RU"]), _settings())


# --- build_id checks -------------------------------------------------------


def test_build_id_is_read_from_file_metadata(directory) -> None:
    """Polars writes it to the footer, not the Arrow schema — see read_build_id."""
    assert read_build_id(artifact_set(directory).places) == BUILD_ID


def test_mismatched_build_id_is_rejected(directory) -> None:
    """The half-updated-directory case: a fresh index, yesterday's places."""
    paths = artifact_set(directory)
    _write_table(
        pl.read_parquet(paths.places),
        paths.places,
        "99999999-0000-0000-0000-000000000000",
    )
    with pytest.raises(StaleArtifactsError, match="not one set"):
        check_build_ids(paths, load_manifest(paths.manifest))


def test_unstamped_table_is_rejected(directory) -> None:
    """A table written without a stamp cannot be proven to belong to the set."""
    paths = artifact_set(directory)
    pl.read_parquet(paths.places).write_parquet(paths.places)
    with pytest.raises(StaleArtifactsError, match="not one set"):
        check_build_ids(paths, load_manifest(paths.manifest))


# --- coverage --------------------------------------------------------------


def test_coverage_gap_is_caught_at_build_time() -> None:
    """A retrievable geonameid with no place row would vanish from results."""
    from src.search.artifacts import _check_coverage

    places = pl.DataFrame([(524901,)], schema={"geonameid": pl.Int64}, orient="row")
    with pytest.raises(RuntimeError, match="retrievable but absent"):
        _check_coverage(CORPUS, places)


def test_manifest_json_is_readable(directory) -> None:
    """The manifest is a record a human reads, so it stays plain JSON."""
    payload = json.loads(artifact_set(directory).manifest.read_text())
    assert payload["build_id"] == BUILD_ID
    assert payload["countries"] == ["RU", "US", "TR", "CN"]
