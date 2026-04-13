"""CLI entry point: run the full region name preprocessing pipeline."""
import argparse
import asyncio
import os

from ..config import COUNTRIES, LANGUAGES, PROCESSED_DIR, RAW_DIR
from .pipeline import build_regions


async def main(warm_start: bool = False) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Building multilingual region names...")
    df = await build_regions(
        languages=LANGUAGES,
        countries=COUNTRIES,
        max_concurrent=32,
        warm_start=warm_start,
        raw_dir=RAW_DIR,
    )
    print(f"{df.height} region-language rows built.")

    out_path = PROCESSED_DIR / "regions.parquet"
    print(f"Saving -> {out_path}")
    df.write_parquet(out_path)
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess admin1 region names.")
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Resume from cached translations instead of starting over.",
    )
    args = parser.parse_args()
    asyncio.run(main(warm_start=args.warm_start))
