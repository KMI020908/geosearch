"""Dense retrieval index backed by a fine-tuned sentence-transformers model."""
import logging
import pickle
from pathlib import Path

import faiss
import numpy as np
import polars as pl
from country_list import countries_for_language
from sentence_transformers import SentenceTransformer
from src.data.languages import get_language
from src.data.utils import make_region_resolver
from src.data.config import PRIMARY_LANGUAGES

logger = logging.getLogger(__name__)


class BiEncoderIndex:
    """
    Cosine-similarity retrieval index over multilingual city name documents.

    The index reuses the same corpus (and doc→city lookup tables) as
    :class:`GeoSearchIndex`, so a single call to
    ``BiEncoderIndex.from_bm25_index`` is all that is needed after the BM25
    index has been built.

    Persistence
    -----------
    ``save`` / ``load`` serialise everything *except* the model weights.  The
    model is reloaded from ``model_path`` on ``load``, so the model directory
    must still be present at load time.
    """

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Path,
        model_path: str | Path,
        batch_size: int = 512,
    ) -> "BiEncoderIndex":
        """Build the dense index from a cities Parquet file.

        Documents are language-aware descriptions (one per city) built from
        the city's primary language, e.g. "Город Москва, страна Россия, Москва".

        Parameters
        ----------
        parquet_path:  Path to the processed cities Parquet file.
        model_path:    Path to the local sentence-transformers model directory.
        batch_size:    Encoding batch size.
        """

        df = pl.read_parquet(parquet_path)
        region_resolver = make_region_resolver()
        
        city_descriptions = []
        for row in df.iter_rows(named=True):
            iso_code = PRIMARY_LANGUAGES.get(row["country_code"], "en")
            country_code = row["country_code"]
            country_name = dict(countries_for_language(iso_code))[country_code]
            admin1_name = region_resolver(country_code, row["admin1_code"], iso_code)
            for i in row["info"]:
                if i["language"] == iso_code:
                    info = i
                    break
                elif i["language"] == "en":
                    en_info = i
            else:
                info = en_info
            name = info["names"][0]
            other_names = info["names"][1:]
            city_descriptions.append((
                get_language(iso_code).city_description(name, country_name, admin1_name, other_names),
                row["geoname_id"]
            ))

        city_descriptions = (
            pl.DataFrame(city_descriptions, schema={"description": pl.String(), "geoname_id": pl.Int64()}, orient="row")
            .with_columns(pl.col("description").str.to_lowercase())
        )
        corpus = dict(zip(city_descriptions["geoname_id"].cast(pl.String()).to_list(), city_descriptions["description"].to_list()))

        instance = cls.__new__(cls)
        instance._model_path = str(model_path)
        instance._docid_to_geoid = city_descriptions["geoname_id"].to_list()
        instance._corpus = corpus
        instance._cities = {
            row["geoname_id"]: {
                "geoname_id":   row["geoname_id"],
                "ascii_name":   row["ascii_name"],
                "country_code": row["country_code"],
                "admin1_code":  row["admin1_code"],
                "latitude":     row["latitude"],
                "longitude":    row["longitude"],
                "population":   row["population"],
            }
            for row in df.unique("geoname_id").iter_rows(named=True)
        }

        logger.info("Loading retriever model", extra={"path": str(model_path)})
        model = SentenceTransformer(str(model_path))

        logger.info("Encoding corpus", extra={"n_docs": len(corpus)})
        instance._corpus_embeddings = model.encode(
            list(corpus.values()),
            batch_size=batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(np.float32)
        instance._model = model

        instance._index = faiss.IndexHNSWFlat(instance._corpus_embeddings.shape[1], 32, faiss.METRIC_INNER_PRODUCT)
        instance._index.add(instance._corpus_embeddings)
        instance._index.hnsw.efSearch = 64

        return instance

    @classmethod
    def load(cls, index_path: Path) -> "BiEncoderIndex":
        """Load a saved index and reinitialise the sentence-transformers model."""
        with open(index_path, "rb") as f:
            instance = pickle.load(f)
        logger.info("Loading retriever model", extra={"path": instance._model_path})
        instance._model = SentenceTransformer(instance._model_path)
        return instance

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, index_path: Path) -> None:
        """Serialise the index (embeddings + lookup tables) to disk.

        The model weights are *not* stored — only ``_model_path`` is kept so
        the model can be reloaded on the next ``load`` call.
        """
        index_path.parent.mkdir(parents=True, exist_ok=True)
        model, self._model = self._model, None
        try:
            with open(index_path, "wb") as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        finally:
            self._model = model

    # ── public interface ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        context_id: str | None = None,
    ) -> list[dict]:
        """Return up to *top_k* cities ranked by cosine similarity."""
        logger.info(
            "BiEncoder tokenized query",
            extra={"context_id": context_id, "stage": "biencoder"},
        )
        query_embeddings = self._model.encode(
            [query],
            convert_to_numpy=True,
            show_progress_bar=True,
            normalize_embeddings=True,
            batch_size=32
        )
        scores, indices = self._index.search(query_embeddings, top_k)
        gids = [self._docid_to_geoid[i] for i in indices.flat]
        results = []
        for gid, score in zip(gids, scores.flat):
            result = self._cities[gid]
            result.update({"score": score})
            results.append(result)
        return results
