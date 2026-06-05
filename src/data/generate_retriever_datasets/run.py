#!/usr/bin/env python3
"""Generate train and validation retriever datasets from template queries.

Splits by template_index per source: for each source, indices up to the
train threshold go to train, the rest to val. Queries are deduplicated
by query text.

Output schema: query, geoname_ids.
"""
import os
from pathlib import Path

import click
import polars as pl

from ..config import PROCESSED_DIR, LANGUAGES


@click.command()
@click.option(
    "--train-size", "-t", type=float, default=0.95, show_default=True,
    help="Fraction of template indices per source assigned to train.",
)
@click.option(
    "--seed", "-s", type=int, default=42, show_default=True,
    help="Random seed for shuffle.",
)
@click.option(
    "--languages", "-l", multiple=True, default=None,
    help="Language codes (e.g. en ru tr). Defaults to all configured languages.",
)
@click.option(
    "--out-dir", "-o", type=click.Path(path_type=os.PathLike), default=None,
    help="Output directory. Default: data/processed/retriever_datasets",
)
def main(
    train_size: float,
    seed: int,
    languages: tuple[str, ...] | None,
    out_dir: os.PathLike | None,
) -> None:
    """Generate train and val retriever datasets (query, geoname_ids)."""
    languages_list = list(languages) if languages else LANGUAGES
    out = Path(out_dir) if out_dir else PROCESSED_DIR / "retriever_datasets"
    os.makedirs(out, exist_ok=True)

    click.echo(f"Languages: {languages_list}, train_size: {train_size}, seed: {seed}")

    df = pl.read_parquet(PROCESSED_DIR / "template_queries")
    click.echo(f"Loaded {df.shape[0]:,} queries")

    train_parts: list[pl.DataFrame] = []
    val_parts: list[pl.DataFrame] = []

    for _, group in df.group_by("source"):
        max_ti = group["template_index"].max()
        if max_ti is None or max_ti == 0:
            threshold = 0
        else:
            n_train = max(1, int(train_size * (max_ti + 1)))
            threshold = n_train - 1

        train_parts.append(group.filter(pl.col("template_index") <= threshold))
        val_part = group.filter(pl.col("template_index") > threshold)
        if not val_part.is_empty():
            val_parts.append(val_part)

    train_df = pl.concat(train_parts, how="vertical")
    val_df = pl.concat(val_parts, how="vertical") if val_parts else pl.DataFrame()

    out_cols = ["query", "geoname_ids"]
    train_df = train_df.unique(subset=["query"], keep="first").select(out_cols)
    if not val_df.is_empty():
        val_df = val_df.unique(subset=["query"], keep="first").select(out_cols)

    train_path = out / "train.parquet"
    val_path = out / "val.parquet"
    train_df.sort("query").write_parquet(train_path)
    val_df.sort("query").write_parquet(val_path)

    click.echo(f"Saved train: {train_df.shape[0]:,} -> {train_path}")
    click.echo(f"Saved val:   {val_df.shape[0]:,} -> {val_path}")


if __name__ == "__main__":
    main()
