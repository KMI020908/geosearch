"""``python -m src.hub.push --what {ner,rerank,index,data,space,all}``

One entry point per artifact rather than a single "publish everything", because
they are rebuilt on different schedules and pushing an unchanged 1.1 GB
checkpoint alongside a 12 KB Parquet is not a thing to do by accident.
"""

from __future__ import annotations

import argparse
import logging

from src.config import settings
from src.hub import publish

logger = logging.getLogger(__name__)

TARGETS = ("ner", "rerank", "index", "data", "space")


def run(what: str) -> str:
    hf = settings.hf
    if what == "ner":
        return publish.push_ner_model(settings.ner, settings, repo_id=hf.ner_repo)
    if what == "rerank":
        return publish.push_reranker(settings, repo_id=hf.reranker_repo)
    if what == "index":
        return publish.push_index(settings, repo_id=hf.index_repo)
    if what == "data":
        return publish.push_datasets(settings, repo_id=hf.datasets_repo)
    if what == "space":
        return publish.push_space(settings, repo_id=hf.space_repo)
    raise SystemExit(f"Unknown target {what!r}; expected one of {', '.join(TARGETS)}.")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--what", required=True, choices=[*TARGETS, "all"])
    args = parser.parse_args()

    targets = TARGETS if args.what == "all" else (args.what,)
    for target in targets:
        sha = run(target)
        logger.info("%s → %s", target, sha or "(no sha reported)")


if __name__ == "__main__":
    main()
