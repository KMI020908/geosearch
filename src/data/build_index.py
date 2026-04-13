"""Build and save the BM25 search index from the processed cities Parquet file."""
import sys
from src.data.config import INDEX_PATH, PROCESSED_DIR
from src.models.bm25_index import GeoSearchIndex


def main() -> None:
    parquet_path = PROCESSED_DIR / "cities.parquet"

    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found. Run preprocessing first:")
        print("python -m src.data.preprocess_cities.run")
        sys.exit(1)

    print(f"Building BM25 index from {parquet_path} ...")
    index = GeoSearchIndex.from_parquet(parquet_path)

    print(f"Saving index to {INDEX_PATH} ...")
    index.save(INDEX_PATH)

    size_mb = INDEX_PATH.stat().st_size / 1024 / 1024
    print(f"Done. Index size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
