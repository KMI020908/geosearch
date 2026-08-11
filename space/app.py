"""Gradio demo for the multilingual toponym search pipeline.

Runs the *same* `src/` code the API and the evaluation harness run — the Space
push ships a copy of the package rather than a reimplementation, so what is
demoed here cannot quietly diverge from what is measured.

Two things shape this file.

**There is no database.** The engine reads prebuilt artifacts pulled from the
Hub — the same files `make artifacts` compiles on the authoring machine.

**Cold start is minutes, not seconds.** ~1.3 GB has to come down (1.15 GB of
GLiNER weights, ~90 MB index, ~45 MB of tables) and then load. A free Space has
no persistent storage, so every cold boot pays it again. Rather than block
import — which would leave the port unbound and show visitors a build error —
loading runs on a daemon thread and the handlers wait on an Event behind an
honest message.
"""

from __future__ import annotations

import json
import logging
import os
import sys
import threading
from pathlib import Path

import gradio as gr
import pandas as pd

# `make hub-push-space` uploads `src/` beside this file, so on the Space they are
# siblings and the import just works. Stated explicitly anyway, so running the
# demo locally (`python space/app.py` from the repo root) resolves the same
# package rather than failing on sys.path.
for candidate in (Path(__file__).resolve().parent, Path.cwd()):
    if (candidate / "src").is_dir() and str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)

# Set before importing src.config so pydantic-settings sees it: the Space has no
# .env to read one from.
os.environ.setdefault("ARTIFACTS_DIR", "data/artifacts")

_engine = None
_ready = threading.Event()
_error: BaseException | None = None


def _load() -> None:
    """Fetch artifacts and build the engine. Runs once, on a background thread."""
    global _engine, _error
    try:
        import asyncio

        from src.config import settings
        from src.hub.pull import pull_index, pull_reranker
        from src.search.engine import SearchEngine

        if not (Path(settings.artifacts_dir) / "manifest.json").exists():
            logger.info("No local artifacts — pulling from the Hub…")
            pull_index(settings)
            try:
                pull_reranker(settings)
            except (Exception, SystemExit) as exc:  # noqa: BLE001
                # A missing reranker is a documented degradation, not an outage:
                # search falls back to retriever order + population tiebreak.
                # `pull_reranker` raises SystemExit (like `pull_index`) when the
                # Hub fetch fails — not caught by a bare `except Exception`.
                logger.warning("No reranker (%s) — using retriever order", exc)

        _engine = asyncio.run(SearchEngine.build(settings))
        logger.info("Engine ready")
    except BaseException as exc:  # noqa: BLE001 — surfaced in the UI below
        logger.exception("Failed to build the engine")
        _error = exc
    finally:
        _ready.set()


def _require_engine():
    _ready.wait()
    if _error is not None:
        raise gr.Error(f"The demo failed to start: {_error}")
    return _engine


def _highlight(text: str, spans) -> list[tuple[str, str | None]]:
    """Build gr.HighlightedText segments from character offsets.

    Offsets, not a search for the span text: a surface form can occur more than
    once in a query and only one occurrence was the tagged one.
    """
    segments: list[tuple[str, str | None]] = []
    cursor = 0
    for span in sorted(spans, key=lambda s: s.start):
        if span.start > cursor:
            segments.append((text[cursor : span.start], None))
        segments.append((text[span.start : span.end], span.label))
        cursor = span.end
    if cursor < len(text):
        segments.append((text[cursor:], None))
    return segments or [(text, None)]


def _table(ranked) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "#": i + 1,
                "name": m.asciiname,
                "country": m.country_code,
                "population": m.population,
                "code": m.feature_code,
                "score": round(score, 4),
                "bm25": round(m.retriever_score, 4),
            }
            for i, (m, score) in enumerate(ranked)
        ]
    )


def _map(ranked):
    """Scatter the results geographically. No tiles, no API key."""
    import plotly.express as px

    points = [
        {
            "name": m.asciiname,
            "lat": m.latitude,
            "lon": m.longitude,
            "population": max(m.population, 1),
            "rank": i + 1,
        }
        for i, (m, _) in enumerate(ranked)
        if m.latitude is not None and m.longitude is not None
    ]
    if not points:
        return None
    figure = px.scatter_geo(
        pd.DataFrame(points),
        lat="lat",
        lon="lon",
        size=[len(points) - d["rank"] + 1 for d in points],
        hover_name="name",
        hover_data={"rank": True, "population": True, "lat": False, "lon": False},
        projection="natural earth",
    )
    figure.update_layout(margin={"r": 0, "t": 0, "l": 0, "b": 0}, height=380)
    return figure


def search(text: str, top_k: int):
    """Run the pipeline twice — with and without the reranker — and show both.

    Side by side because the comparison *is* the interesting part: on a homonym
    query it shows exactly what the reranker contributes over BM25 score plus a
    population tiebreak, live, on input the viewer chose.
    """
    import asyncio

    engine = _require_engine()
    if not text.strip():
        raise gr.Error("Type something with a place name in it.")

    reranked = asyncio.run(engine.search(text, top_k=top_k, use_rerank=True))
    baseline = asyncio.run(engine.search(text, top_k=top_k, use_rerank=False))

    if not reranked.spans:
        gr.Warning(
            "NER found no place names, so there was nothing to retrieve. "
            "Try naming a city explicitly."
        )

    buckets = {k: v for k, v in reranked.entity_buckets.items() if v}
    return (
        _highlight(text, reranked.spans),
        json.dumps(buckets, ensure_ascii=False, indent=2) if buckets else "{}",
        _table(reranked.ranked),
        _table(baseline.ranked),
        _map(reranked.ranked),
    )


EXAMPLES = [
    ["Хочу поехать в Москву и Казань на выходных", 20],
    ["cheap flights to Istanbul next week", 20],
    ["莫斯科新闻", 20],
    ["İzmir ve Ankara arasında tren var mı", 20],
    ["Springfield, Illinois", 20],
]

CSS = """
.gradio-container {max-width: 1200px !important}
"""

with gr.Blocks(title="GeoSearch — multilingual toponym search") as demo:
    gr.Markdown(
        """
        # GeoSearch

        Multilingual toponym recognition and matching over
        [GeoNames](https://www.geonames.org/). Type free text in **Russian,
        English, Turkish or Chinese**; the pipeline finds the place names in it
        and returns ranked GeoNames entries.

        ```
        text -> GLiNER NER -> char-n-gram BM25 -> cross-encoder reranker -> places
        ```

        No database: this Space serves prebuilt artifacts pulled from the Hub.
        **First request after a cold start takes a few minutes** while ~1.3 GB
        of model and index download.
        """
    )

    with gr.Row():
        query = gr.Textbox(
            label="Query",
            placeholder="Хочу поехать в Москву и Казань…",
            scale=4,
            autofocus=True,
        )
        top_k = gr.Slider(5, 100, value=20, step=5, label="top-k", scale=1)
    run = gr.Button("Search", variant="primary")

    with gr.Row():
        spans_out = gr.HighlightedText(
            label="NER spans", combine_adjacent=True, scale=3
        )
        buckets_out = gr.Code(
            label="Entity buckets (NER spans by type, for display)",
            language="json",
            scale=2,
        )

    gr.Markdown(
        "### Reranked vs retriever order\n"
        "Left is the full pipeline. Right is retrieval alone — BM25 score with "
        "population breaking ties. Both rank the *same* candidate set, so the "
        "difference is exactly the reranker's contribution."
    )
    with gr.Row():
        reranked_out = gr.Dataframe(label="With reranker", scale=1)
        baseline_out = gr.Dataframe(label="Retriever order", scale=1)

    map_out = gr.Plot(label="Results on the map")

    gr.Examples(examples=EXAMPLES, inputs=[query, top_k])

    gr.Markdown(
        """
        ---
        Place data from [GeoNames](https://www.geonames.org/), licensed
        [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); filtered to
        populated places in RU/US/TR/CN with name variants in ru/en/tr/zh.

        Artifacts: [geosearch-ner](https://huggingface.co/mki0809/geosearch-ner)
        · [geosearch-reranker](https://huggingface.co/mki0809/geosearch-reranker)
        · [geosearch-index](https://huggingface.co/datasets/mki0809/geosearch-index)
        · [geosearch-queries](https://huggingface.co/datasets/mki0809/geosearch-queries)

        """
    )

    outputs = [spans_out, buckets_out, reranked_out, baseline_out, map_out]
    run.click(search, inputs=[query, top_k], outputs=outputs)
    query.submit(search, inputs=[query, top_k], outputs=outputs)


if __name__ == "__main__":
    # Start loading *before* launch so the port binds immediately and visitors
    # get the UI (and an honest wait) rather than a Space that looks broken for
    # two minutes. Under `__main__` rather than at import, so importing this
    # module — which is how it is tested — does not kick off a 1.3GB download.
    threading.Thread(target=_load, daemon=True).start()

    # css belongs to launch() as of Gradio 6. The port honours
    # GRADIO_SERVER_PORT — passing a literal here would override the env var and
    # make a second instance impossible to start, which is exactly what happens
    # when you want to check a fix against an already-running one. 7860 is the
    # default Spaces expects. `0.0.0.0` so the container is reachable from
    # outside it.
    demo.queue().launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", 7860)),
        css=CSS,
    )
