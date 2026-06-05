#!/usr/bin/env python3
"""CLI entry point for generating template-based training queries.

Reads city parquet files from ``data/processed/cities/`` (all countries),
applies query templates for each language via the query engine, and writes
output to ``data/processed/template_queries/{lang}.parquet``.

Each output row contains:

- ``query``          -- the generated search query text
- ``geoname_ids``    -- list of matching city geoname IDs (ground truth)
- ``language``       -- ISO language code
- ``source``         -- which template category produced the query (single- or two-city)
- ``template_index`` -- variant index within the template category
"""
import os

import click
import polars as pl

from  ..config import PROCESSED_DIR, QUERY_SCHEMA, LANGUAGES
from ..queries import generate_for_language
from .utils import load_cities
from ..utils import make_region_resolver


@click.command()
@click.option(
    "--languages", "-l", multiple=True, default=None,
    help="Language codes (e.g. en ru tr). Defaults to config.",
)
@click.option(
    "--max-pairs", "-m", type=int, default=None,
    help="Max city pairs per two-city generator (default: engine default 10 000).",
)
@click.option(
    "--templates-per-city",
    type=int,
    default=15,
    help=(
        "Randomly sample at most this many template variants per single-city "
        "template family (per city). Defaults to None = use all templates."
    ),
)
@click.option(
    "--template-seed",
    type=int,
    default=67,
    show_default=True,
    help="Random seed for per-city template sampling.",
)
def main(
    languages: tuple[str, ...] | None,
    max_pairs: int | None,
    templates_per_city: int | None,
    template_seed: int,
) -> None:
    """Generate template-based training queries partitioned by language.

    Reads city records from ``data/processed/cities/`` (all countries),
    applies single-city and two-city query templates for each language, and
    writes the results to ``data/processed/template_queries/{lang}.parquet``.
    """
    languages_list = list(languages) if languages else LANGUAGES

    click.echo(f"Languages: {languages_list}")

    cities = load_cities(languages_list)
    if cities.is_empty():
        click.echo("No cities found.")
        return

    click.echo(f"Loaded {cities.shape[0]} cities.")

    get_region_name = make_region_resolver()
    out_dir = PROCESSED_DIR / "template_queries"
    os.makedirs(out_dir, exist_ok=True)
    total_queries = 0

    for lang in languages_list:
        click.echo(f"\nGenerating [{lang}] queries...")

        kwargs: dict = {}
        if max_pairs is not None:
            kwargs["max_pairs"] = max_pairs
        if templates_per_city is not None and templates_per_city > 0:
            kwargs["template_sample"] = templates_per_city
            kwargs["template_seed"] = template_seed
        data = generate_for_language(cities, lang, get_region_name, **kwargs)

        if not data:
            click.echo(f"  [{lang}] -- no queries generated.")
            continue

        df = (
            pl.DataFrame(data, schema=QUERY_SCHEMA, orient="row")
            .sort("source", "template_index", "query")
        )
        out_path = out_dir / f"{lang}.parquet"
        df.write_parquet(out_path)

        count = df.shape[0]
        total_queries += count
        click.echo(f"  [{lang}] -- saved {count} queries to {out_path}")

    click.echo(f"\nDone. Total queries generated: {total_queries}")


if __name__ == "__main__":
    main()
