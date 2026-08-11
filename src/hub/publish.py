"""Publish each artifact to its own Hub repo, with a card describing it.

Publishing is always an explicit command, never a side effect of building: a
push sends data outside this machine, and a network failure or an expired token
should not cost a fifteen-epoch fine-tune. ``make ner-train`` reaches
:func:`push_ner_model` only when ``NER__PUSH_TO_HUB`` is set.

Five repos, because they version independently — re-training NER forces a
reranker re-train (it changes the spans the reranker is fed) but not the
reverse, and the serving index is rebuilt on every GeoNames refresh while the
query dataset is not.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import NerConfig, Settings
from src.hub import cards
from src.hub.client import ensure_repo, push_files, push_folder, require_token

logger = logging.getLogger(__name__)


def _write_card(text: str, directory: Path) -> Path:
    path = directory / "README.md"
    path.write_text(text, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def push_ner_model(cfg: NerConfig, settings: Settings, *, repo_id: str) -> str:
    """Upload the fine-tuned GLiNER directory.

    Uploads the **directory**, not the in-memory model: ``GLiNER.push_to_hub``
    re-serialises from memory and would drop ``ner_meta.json`` — the file
    recording which word splitter this checkpoint requires, which is precisely
    what must never get separated from the weights.

    Refuses a directory without that file. A checkpoint whose splitter
    requirement is unrecorded degrades Chinese silently wherever it is later
    loaded, and the Hub is exactly where it stops being obvious which training
    run produced it.
    """
    from src.ner.train import META_FILENAME

    require_token(settings)
    model_dir = Path(cfg.model_dir)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"{model_dir} does not exist — run `make ner-train` before publishing."
        )
    meta_path = model_dir / META_FILENAME
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} is missing, so the upload would not record which word "
            "splitter this checkpoint needs. Re-run `make ner-train`."
        )

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    metrics = cards._read_json(cfg.metrics_path)
    if metrics is None:
        logger.warning(
            "%s is absent — the card will say 'not measured'. Run `make ner-eval` "
            "first if you want the numbers on it.",
            cfg.metrics_path,
        )

    ensure_repo(repo_id, "model", private=settings.hf.private, settings=settings)
    _write_card(cards.render_ner_card(meta, metrics, settings), model_dir)
    if metrics is not None:
        (model_dir / "ner_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return push_folder(
        model_dir,
        repo_id,
        "model",
        message=(
            f"Fine-tuned {meta.get('base_model')} — epoch {meta.get('best_epoch')}, "
            f"val span F1 {meta.get('best_val_span_f1')}"
        ),
        settings=settings,
    )


def push_reranker(settings: Settings, *, repo_id: str) -> str:
    """Upload the fine-tuned cross-encoder checkpoint directory and its card.

    Uploads the **directory** `make rerank-train` wrote — config.json, weights,
    tokenizer files, `rerank_meta.json` — the same shape `push_ner_model` uses,
    since a sentence-transformers checkpoint is a directory, not a single file.
    """
    require_token(settings)
    cfg = settings.rerank
    model_dir = Path(cfg.model_path)
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"{model_dir} does not exist — run `make rerank-train` first."
        )
    # The one fact about this model that must not be lost: a model trained on
    # gold entities is not servable, and nothing in the checkpoint itself
    # records which it is. Checked before any Hub I/O, so a bad flag fails fast.
    if cfg.use_gold_entities:
        raise SystemExit(
            "RERANK__USE_GOLD_ENTITIES is true, so the trained model is the "
            "gold-entity ablation — it breaks train/serve parity by design and "
            "must not be published as servable. Re-train with it false."
        )

    meta_path = model_dir / "rerank_meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(
            f"{meta_path} is missing — re-run `make rerank-train` (it writes "
            "this alongside the weights)."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    metrics = cards._read_json(cfg.metrics_path)
    if metrics is None:
        logger.warning(
            "%s is absent — the card will say 'not measured'. Run `make rerank-eval`.",
            cfg.metrics_path,
        )

    ensure_repo(repo_id, "model", private=settings.hf.private, settings=settings)
    _write_card(cards.render_reranker_card(meta, metrics, settings), model_dir)
    if metrics is not None:
        (model_dir / "rerank_metrics.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    return push_folder(
        model_dir,
        repo_id,
        "model",
        message=(
            f"Fine-tuned {meta.get('base_model')} — epoch {meta.get('best_epoch')}, "
            f"P@1 {meta.get('best_p_at_1')}"
        ),
        settings=settings,
    )


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------


def push_index(settings: Settings, *, repo_id: str) -> str:
    """Upload the four serving artifacts as one commit.

    One commit, not four uploads: the files are only valid as a set, and a
    puller must never be able to observe a half-updated repo — the exact state
    ``build_id`` exists to detect.
    """
    from src.search.artifacts import SERVING_FILES, artifact_set, load_manifest

    require_token(settings)
    directory = Path(settings.artifacts_dir)
    paths = artifact_set(directory)
    manifest = load_manifest(paths.manifest)

    ensure_repo(repo_id, "dataset", private=settings.hf.private, settings=settings)
    with TemporaryDirectory() as tmp:
        files = {name: directory / name for name in SERVING_FILES}
        files["README.md"] = _write_card(
            cards.render_index_card(manifest, settings), Path(tmp)
        )
        return push_files(
            files,
            repo_id,
            "dataset",
            message=f"Serving artifacts, build {manifest.build_id}",
            settings=settings,
        )


def push_datasets(settings: Settings, *, repo_id: str) -> str:
    """Upload the query corpus and derived training sets.

    Parquet is normalised on the way out (:mod:`src.hub.arrow`) so the Dataset
    Viewer renders it; the local files stay exactly as polars wrote them.
    """
    import polars as pl

    from src.hub.arrow import normalize_parquet

    require_token(settings)
    ds, rr, ner = settings.dataset, settings.rerank, settings.ner

    # path-in-repo -> local source. The layout mirrors the `configs:` block in
    # the card, so a rename here without one there breaks the viewer.
    sources: dict[str, Path] = {
        "queries/query_dataset.parquet": Path(ds.output_path),
        "queries/query_plan.parquet": Path(ds.plan_path),
        "rerank/train.parquet": Path(rr.train_path),
        "rerank/test.parquet": Path(rr.test_path),
        "eval/golden_set.parquet": Path(rr.golden_set_path),
    }
    plain: dict[str, Path] = {
        "ner/train.json": Path(ner.train_path),
        "ner/val.json": Path(ner.val_path),
        "raw/query_dataset.checkpoint.jsonl": Path(ds.checkpoint_path),
    }

    missing = [n for n, p in sources.items() if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Cannot publish, these are not built: " + ", ".join(missing)
        )

    def _rows(path: Path) -> int | str:
        return pl.read_parquet(path).height if path.exists() else "?"

    stats = {
        "queries": _rows(Path(ds.output_path)),
        "plan": _rows(Path(ds.plan_path)),
        "rerank_train": _rows(Path(rr.train_path)),
        "rerank_test": _rows(Path(rr.test_path)),
        "golden": _rows(Path(rr.golden_set_path)),
        "model": ds.deepseek_model,
    }

    ensure_repo(repo_id, "dataset", private=settings.hf.private, settings=settings)
    with TemporaryDirectory() as tmp:
        staging = Path(tmp)
        files = {
            name: normalize_parquet(source, staging / name)
            for name, source in sources.items()
        }
        files.update({n: p for n, p in plain.items() if p.exists()})
        files["README.md"] = _write_card(
            cards.render_dataset_card(stats, settings), staging
        )
        for name, path in plain.items():
            if not path.exists():
                logger.info("Skipping %s — not built", name)
        return push_files(
            files,
            repo_id,
            "dataset",
            message="Publish query dataset and derived training sets",
            settings=settings,
        )


def push_space(settings: Settings, *, repo_id: str) -> str:
    """Upload the Gradio app plus a copy of ``src/``.

    The app runs the *same* pipeline code as everything else rather than a
    forked copy, so the demo cannot silently diverge from what is measured. What
    it does not carry is enforcement: nothing re-pushes on a ``src/`` change, so
    the Space README is stamped with the source git sha, making drift visible.
    """
    import shutil
    import subprocess

    require_token(settings)
    space_dir = Path(settings.hf.space_dir)
    if not space_dir.is_dir():
        raise FileNotFoundError(f"{space_dir} does not exist.")

    try:
        sha = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:  # noqa: BLE001 — a missing git is not a reason not to ship
        sha = "unknown"

    ensure_repo(
        repo_id,
        "space",
        private=settings.hf.private,
        settings=settings,
        space_sdk="gradio",
    )
    with TemporaryDirectory() as tmp:
        staging = Path(tmp)
        shutil.copytree(space_dir, staging, dirs_exist_ok=True)
        shutil.copytree(
            Path("src"),
            staging / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
        readme = staging / "README.md"
        readme.write_text(
            readme.read_text(encoding="utf-8").replace("{{GIT_SHA}}", sha),
            encoding="utf-8",
        )
        return push_folder(
            staging,
            repo_id,
            "space",
            message=f"Deploy Space from geosearch@{sha}",
            settings=settings,
        )
