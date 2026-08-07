"""Thin wrappers over ``huggingface_hub``, and the only module that calls it.

Two reasons for the indirection. Practically, ``huggingface_hub`` 1.x is a
recent breaking major, and keeping every call behind one file makes the next one
a single place to fix. Substantively, every fetch here **logs the commit it
resolved** and every push **returns and prints the commit to pin**.

That is the whole reproducibility mechanism. ``main`` is the right default while
developing, but a number reported in a thesis has to name the artifact it was
measured on, and "the model as of some Tuesday" is not a citation. With this,
the startup log of any run records the exact revision it served, and
``HF__INDEX_REVISION=<sha>`` freezes it.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Literal

from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from src.config import Settings

logger = logging.getLogger(__name__)

RepoType = Literal["model", "dataset", "space"]


def api(settings: Settings) -> HfApi:
    """An :class:`HfApi` carrying the configured token, if there is one."""
    return HfApi(token=settings.hf_token)


def require_token(settings: Settings) -> str:
    """Return the write token, or explain how to get one.

    Pushing without a token fails as a 401 from deep inside the client; failing
    here instead names the variable and the page that issues it.
    """
    if not settings.hf_token:
        raise SystemExit(
            "HF_TOKEN is not set. Create a *write* token at "
            "https://huggingface.co/settings/tokens and put it in .env "
            "(not .env.example — that file is committed)."
        )
    return settings.hf_token


def ensure_repo(
    repo_id: str,
    repo_type: RepoType,
    *,
    private: bool,
    settings: Settings,
    space_sdk: str | None = None,
) -> None:
    """Create *repo_id* if it does not exist.

    ``exist_ok=True``, so re-publishing is the same command as publishing. Note
    that this does **not** change an existing repo's visibility: ``private`` is
    honoured only at creation, so flipping a published repo public is a
    deliberate action on the website, not a side effect of a re-push.

    A Space additionally needs its SDK named at creation — the Hub has no
    default and rejects the call without one — so it is required here rather
    than left to fail deep in the client.
    """
    if repo_type == "space" and not space_sdk:
        raise ValueError("Creating a Space requires space_sdk (e.g. 'gradio').")
    api(settings).create_repo(
        repo_id=repo_id,
        repo_type=repo_type,
        private=private,
        exist_ok=True,
        space_sdk=space_sdk,
    )


def push_folder(
    folder: Path,
    repo_id: str,
    repo_type: RepoType,
    *,
    message: str,
    settings: Settings,
    allow_patterns: list[str] | None = None,
    ignore_patterns: list[str] | None = None,
    path_in_repo: str = "",
) -> str:
    """Upload a directory, returning the commit sha and logging how to pin it."""
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} does not exist — nothing to upload.")
    logger.info("Uploading %s → %s (%s)…", folder, repo_id, repo_type)
    commit = api(settings).upload_folder(
        folder_path=str(folder),
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=message,
        allow_patterns=allow_patterns,
        ignore_patterns=ignore_patterns,
        path_in_repo=path_in_repo,
    )
    return _report(commit, repo_id, repo_type)


def push_files(
    files: Mapping[str, Path],
    repo_id: str,
    repo_type: RepoType,
    *,
    message: str,
    settings: Settings,
) -> str:
    """Upload named files (``path-in-repo -> local path``) as one commit.

    One commit rather than a loop of single-file uploads: the files of an
    artifact set are only valid together, and a partial upload that a puller
    could observe mid-way is exactly the half-updated state ``build_id`` exists
    to detect.
    """
    from huggingface_hub import CommitOperationAdd

    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Cannot upload, missing: {', '.join(missing)}")

    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=str(path))
        for name, path in sorted(files.items())
    ]
    logger.info("Uploading %d file(s) → %s (%s)…", len(operations), repo_id, repo_type)
    commit = api(settings).create_commit(
        repo_id=repo_id,
        repo_type=repo_type,
        operations=operations,
        commit_message=message,
    )
    return _report(commit, repo_id, repo_type)


def _report(commit, repo_id: str, repo_type: RepoType) -> str:
    """Log the commit and the env var that pins it; return the sha."""
    sha = getattr(commit, "oid", None) or ""
    logger.info("Pushed %s → %s", repo_id, getattr(commit, "commit_url", sha))
    if sha:
        logger.info("Pin this exact version with revision=%s", sha)
    return sha


def resolve_revision(
    repo_id: str, repo_type: RepoType, revision: str, settings: Settings
) -> str:
    """Resolve a branch name to the 40-char commit it currently points at."""
    info = api(settings).repo_info(
        repo_id=repo_id, repo_type=repo_type, revision=revision
    )
    return info.sha or revision


def fetch_file(
    repo_id: str,
    filename: str,
    *,
    repo_type: RepoType,
    revision: str,
    settings: Settings,
) -> Path:
    """Download one file, returning its local cache path."""
    return Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            repo_type=repo_type,
            revision=revision,
            token=settings.hf_token,
        )
    )


def fetch_folder(
    repo_id: str,
    *,
    repo_type: RepoType,
    revision: str,
    settings: Settings,
    allow_patterns: list[str] | None = None,
    local_dir: Path | None = None,
) -> Path:
    """Download a repo (or part of it), logging the revision actually served.

    The log line is the point: a run that reports metrics should leave behind a
    record of which artifact produced them, without anyone having to remember to
    write it down.
    """
    logger.info("Fetching %s@%s (%s)…", repo_id, revision, repo_type)
    path = Path(
        snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=revision,
            token=settings.hf_token,
            allow_patterns=allow_patterns,
            local_dir=str(local_dir) if local_dir else None,
        )
    )
    try:
        sha = resolve_revision(repo_id, repo_type, revision, settings)
        logger.info("%s resolved to %s", repo_id, sha)
    except Exception as exc:  # noqa: BLE001 — reporting must not break fetching
        logger.info("Could not resolve %s revision (%s)", repo_id, type(exc).__name__)
    return path
