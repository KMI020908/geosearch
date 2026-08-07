"""``python -m src.hub.pull`` — fetch what a fresh clone needs to run.

The repository no longer carries its data, so this is the first command after
cloning: it brings down the serving artifacts (and optionally the models and
datasets) into the paths ``src/config.py`` already points at.

Each fetch logs the commit it resolved, which is the record of what a given run
actually served. Set ``HF__INDEX_REVISION`` (etc.) to a 40-char sha to freeze it.
"""

from __future__ import annotations

import argparse
import logging
import shutil
from pathlib import Path

from src.config import Settings, settings
from src.hub.client import fetch_folder

logger = logging.getLogger(__name__)


def pull_index(settings: Settings) -> Path:
    """Fetch the serving artifacts into ``settings.artifacts_dir``."""
    from src.search.artifacts import (
        SERVING_FILES,
        artifact_set,
        check_build_ids,
        check_manifest,
        load_manifest,
    )

    destination = Path(settings.artifacts_dir)
    destination.mkdir(parents=True, exist_ok=True)
    try:
        cached = fetch_folder(
            settings.hf.index_repo,
            repo_type="dataset",
            revision=settings.hf.index_revision,
            settings=settings,
            allow_patterns=list(SERVING_FILES),
        )
    except Exception as exc:  # noqa: BLE001 — the raw 401 is actively misleading
        # The Hub answers 401 for a repo you cannot see *and* for one that does
        # not exist, so the default message sends people to check a token when
        # the real cause is usually that nothing has been published yet.
        raise SystemExit(
            f"Could not fetch {settings.hf.index_repo} ({type(exc).__name__}).\n"
            "  - If it has not been published yet: run `make artifacts` then "
            "`make hub-push-index` (needs a write HF_TOKEN).\n"
            "  - If it is private: set HF_TOKEN in .env.\n"
            "  - If you meant a different repo: set HF__INDEX_REPO.\n"
            "Note the Hub returns 401 for both 'no access' and 'does not exist'."
        ) from exc
    for name in SERVING_FILES:
        source = cached / name
        if not source.exists():
            raise FileNotFoundError(f"{settings.hf.index_repo} has no {name}.")
        shutil.copyfile(source, destination / name)

    # Validate what was fetched rather than trusting it: a repo can be
    # half-updated by a failed push exactly as a local directory can.
    paths = artifact_set(destination)
    manifest = load_manifest(paths.manifest)
    check_manifest(manifest, settings)
    check_build_ids(paths, manifest)
    logger.info(
        "Artifacts ready in %s — build %s (%d names, %d places)",
        destination,
        manifest.build_id,
        manifest.n_documents,
        manifest.n_places,
    )
    return destination


def pull_reranker(settings: Settings) -> Path:
    """Fetch ``rerank_model.cbm`` to ``settings.rerank.model_path``."""
    destination = Path(settings.rerank.model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    cached = fetch_folder(
        settings.hf.reranker_repo,
        repo_type="model",
        revision=settings.hf.reranker_revision,
        settings=settings,
        allow_patterns=[destination.name, "rerank_meta.json", "rerank_metrics.json"],
    )
    shutil.copyfile(cached / destination.name, destination)
    logger.info("Reranker ready at %s", destination)
    return destination


def pull_datasets(settings: Settings) -> Path:
    """Fetch the query corpus and derived training sets into ``data/``."""
    ds, rr, ner = settings.dataset, settings.rerank, settings.ner
    wanted = {
        "queries/query_dataset.parquet": Path(ds.output_path),
        "queries/query_plan.parquet": Path(ds.plan_path),
        "rerank/train.parquet": Path(rr.train_path),
        "rerank/test.parquet": Path(rr.test_path),
        "eval/golden_set.parquet": Path(rr.golden_set_path),
        "ner/train.json": Path(ner.train_path),
        "ner/val.json": Path(ner.val_path),
        "raw/query_dataset.checkpoint.jsonl": Path(ds.checkpoint_path),
    }
    cached = fetch_folder(
        settings.hf.datasets_repo,
        repo_type="dataset",
        revision="main",
        settings=settings,
        allow_patterns=list(wanted),
    )
    for name, destination in wanted.items():
        source = cached / name
        if not source.exists():
            logger.info("%s not in the repo — skipping", name)
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
    logger.info("Datasets ready under data/")
    return cached


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--what",
        default="serving",
        choices=["serving", "index", "rerank", "data", "all"],
        help=(
            "serving (default) = index + reranker, everything the API needs; "
            "data = the training/eval corpora; all = both"
        ),
    )
    args = parser.parse_args()

    if args.what in ("serving", "index", "all"):
        pull_index(settings)
    if args.what in ("serving", "rerank", "all"):
        pull_reranker(settings)
    if args.what in ("data", "all"):
        pull_datasets(settings)

    # The NER weights are not copied anywhere: `gliner_model` can name the hub
    # repo directly and `GLiNER.from_pretrained` will cache it itself.
    logger.info(
        "For the NER model, set GLINER_MODEL=%s (no download step needed).",
        settings.hf.ner_repo,
    )


if __name__ == "__main__":
    main()
