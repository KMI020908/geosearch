"""Build and save the BM25 search index from the processed cities Parquet file.

Usage
-----
python -m src.data.build_index
"""
import sys
from src.models.bm25_index import BM25Index
from src.models.biencoder_index import BiEncoderIndex
from src.data.config import BM25_INDEX_PATH, BIENCODER_INDEX_PATH, PROCESSED_DIR, BIENCODER_MODEL


def main() -> None:
    parquet_path = PROCESSED_DIR / "cities.parquet"
    if not parquet_path.exists():
        print(f"ERROR: {parquet_path} not found. Run preprocessing first:")
        print("  python -m src.data.preprocess_cities.run")
        sys.exit(1)

    print(f"Building BM25 index from {parquet_path} ...")
    index = BM25Index.from_parquet(parquet_path)
    print(f"Saving BM25 index to {BM25_INDEX_PATH} ...")
    index.save(BM25_INDEX_PATH)
    size_mb = BM25_INDEX_PATH.stat().st_size / 1024 / 1024
    print(f"Done. BM25 index size: {size_mb:.1f} MB")

    print(f"Building BiEncoder index from {parquet_path} ...")
    index = BiEncoderIndex.from_parquet(parquet_path, BIENCODER_MODEL)
    print(f"Saving BiEncoder index to {BIENCODER_INDEX_PATH} ...")
    index.save(BIENCODER_INDEX_PATH)


if __name__ == "__main__":
    main()
