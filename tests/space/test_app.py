"""The Gradio demo must stay callable against the current `src/`.

It ships a *copy* of the package rather than importing an installed one, so a
signature change in `src/` cannot break it at import — only at the moment a
visitor clicks Search, on a Space that takes minutes to boot. That is precisely
what happened once: `SearchEngine.build` lost its `session_factory` parameter,
`src/api/main.py` was updated, and `space/app.py` was not.

These tests are cheap and offline. They do not start the engine — building it
needs a 1.3GB artifact set — but they do exercise every piece of glue that the
type checker cannot see through, which is where that class of breakage lives.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest

APP_PATH = Path(__file__).resolve().parents[2] / "space" / "app.py"


@pytest.fixture(scope="module")
def app():
    """Import `space/app.py` as a module.

    Safe because the background loader starts under ``__main__``, not at
    import — the whole reason it lives there.
    """
    spec = importlib.util.spec_from_file_location("space_app", APP_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["space_app"] = module
    spec.loader.exec_module(module)
    return module


def test_imports(app) -> None:
    """Catches import errors and anything evaluated at module scope."""
    assert app.demo is not None


async def _stub_build(settings):
    """Stands in for `SearchEngine.build`, with the signature it must have."""
    return "engine"


def test_stub_matches_the_real_build_signature() -> None:
    """Guards the guard: if `src/` changes, the stub below must change too.

    Without this the next test would keep passing against a stale stub while the
    real call broke — the exact failure mode it exists to prevent.
    """
    from src.search.engine import SearchEngine

    def shape(fn):
        # Names and kinds, not annotations: what decides whether a call binds.
        return [(p.name, p.kind) for p in inspect.signature(fn).parameters.values()]

    assert shape(_stub_build) == shape(SearchEngine.build)


def test_load_actually_calls_build(app, monkeypatch, tmp_path) -> None:
    """The regression this file exists for.

    Checking the *signature* is not enough — that is what a first attempt at
    this test did, and it passed happily while `_load` still called
    `build(None, settings)`. The call site has to be executed. Everything
    expensive is stubbed; what is left is the glue.
    """
    monkeypatch.setattr("src.search.engine.SearchEngine.build", _stub_build)
    # Pretend the artifacts are already present, so nothing reaches the network.
    (tmp_path / "manifest.json").write_text("{}")
    monkeypatch.setattr("src.config.settings.artifacts_dir", str(tmp_path))

    app._error = None
    app._ready.clear()
    app._load()

    assert app._error is None, f"_load() failed: {app._error!r}"
    assert app._engine == "engine"
    assert app._ready.is_set()


def test_highlight_uses_offsets_not_search(app) -> None:
    """Spans are placed by character offset, so a repeated surface form works.

    Searching for the span text would highlight the *first* occurrence, which is
    not necessarily the one the model tagged.
    """
    from src.search.engine import EntitySpan

    text = "Moscow to Moscow"
    spans = [EntitySpan(text="Moscow", label="CITY", start=10, end=16)]

    segments = app._highlight(text, spans)

    assert segments == [("Moscow to ", None), ("Moscow", "CITY")]


def test_highlight_handles_no_spans(app) -> None:
    assert app._highlight("nothing here", []) == [("nothing here", None)]


def test_highlight_orders_by_position(app) -> None:
    """Spans arrive in model order, which is not necessarily left to right."""
    from src.search.engine import EntitySpan

    text = "Kazan and Moscow"
    spans = [
        EntitySpan(text="Moscow", label="CITY", start=10, end=16),
        EntitySpan(text="Kazan", label="CITY", start=0, end=5),
    ]
    assert app._highlight(text, spans) == [
        ("Kazan", "CITY"),
        (" and ", None),
        ("Moscow", "CITY"),
    ]


def test_results_table_shape(app) -> None:
    """The dataframe the UI renders, built from engine output."""
    from src.search.engine import GeonameMatch

    match = GeonameMatch(
        geonameid=524901,
        asciiname="Moscow",
        country_code="RU",
        population=10381222,
        feature_code="PPLC",
        latitude=55.75,
        longitude=37.61,
        retriever_score=4.2,
    )
    frame = app._table([(match, 9.9)])

    assert list(frame.columns) == [
        "#",
        "name",
        "country",
        "population",
        "code",
        "score",
        "bm25",
    ]
    assert frame.iloc[0]["name"] == "Moscow"
    assert frame.iloc[0]["score"] == 9.9


def test_map_skips_places_without_coordinates(app) -> None:
    """Coordinates are nullable in GeoNames; a null must not plot at (0, 0)."""
    from src.search.engine import GeonameMatch

    def place(lat, lon):
        return GeonameMatch(
            geonameid=1,
            asciiname="X",
            country_code="RU",
            population=1,
            feature_code=None,
            latitude=lat,
            longitude=lon,
            retriever_score=1.0,
        )

    assert app._map([(place(None, None), 1.0)]) is None
    assert app._map([(place(55.75, 37.61), 1.0)]) is not None


def test_requirements_cover_what_the_app_imports(app) -> None:
    """The Space installs its own requirements.txt, not pyproject.toml.

    A dependency added to `src/` but not to that file breaks the Space and
    nothing else, so the omission is invisible until deploy.
    """
    requirements = (APP_PATH.parent / "requirements.txt").read_text().lower()
    for package in (
        "gradio",
        "gliner",
        "polars",
        "sentence-transformers",
        "plotly",
        "pandas",
    ):
        assert package in requirements, f"{package} missing from requirements.txt"
