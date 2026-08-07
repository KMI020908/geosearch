# The search API as a container image: no database, no sidecar, no external
# service. `./data/artifacts` must be mounted — build it with `make artifacts`
# or fetch it with `make hub-pull`.
#
#     docker build -t geosearch .
#     docker run -p 8000:8000 -v "$PWD/data:/app/data" geosearch
#
# Nothing in this project depends on the image; `uv run uvicorn src.api.main:app`
# is the ordinary way to run the API. It exists for deploying somewhere that
# takes a container and not a Python environment.
#
# KNOWN ISSUE: the image is ~9.4GB, of which 2.7GB is nvidia/* CUDA runtime and
# 698MB is Triton — pulled in because uv.lock pins the default (CUDA) torch
# wheel, which is what `make ner-train` wants on a dev machine. Serving is
# CPU-only and uses none of it. Fixing it means installing
# `torch==2.12.0+cpu` from https://download.pytorch.org/whl/cpu in the *same*
# RUN as `uv sync` (Docker layers are additive, so deleting the CUDA packages
# later does not shrink the image) with `--reinstall-package torch` (uv
# otherwise considers the already-installed version satisfied and skips the CPU
# wheel, leaving a CUDA build whose libraries have been removed — which fails at
# import). Left undone here because the 183MB download from that index proved
# unreliable to fetch during a build.
FROM python:3.12-slim

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

WORKDIR /app

# Dependencies before source, so this layer is cached across code changes.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

COPY src/ ./src/

EXPOSE 8000
CMD ["/app/.venv/bin/uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
