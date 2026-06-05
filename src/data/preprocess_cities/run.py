"""CLI entry point: run the full city preprocessing pipeline."""
import argparse
import asyncio
import os

from ..config import COUNTRIES, LANGUAGES, PROCESSED_DIR, RAW_DIR
from .loader import build_dataset_lf
from .pipeline import fill_missing_names


async def main(warm_start: bool = False) -> None:
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Step 1/3  Building city dataset...")
    df = build_dataset_lf(RAW_DIR, COUNTRIES, LANGUAGES).collect()
    print(f"{df.height} unique city records loaded.")

    print("Step 2/3  Filling missing multilingual names...")
    df = await fill_missing_names(df, max_concurrent=32, warm_start=warm_start)

    out_path = PROCESSED_DIR / "cities.parquet"
    print(f"Step 3/3  Saving -> {out_path}")
    df.write_parquet(out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Resume from cached translations instead of starting over.",
    )
    args = parser.parse_args()
    asyncio.run(main(warm_start=args.warm_start))
