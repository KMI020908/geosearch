"""BM25-based geo-search index backed by a GeoNames Parquet file."""
import logging
import pickle
from pathlib import Path

import polars as pl
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class _BM25Uniform(BM25Okapi):
    """BM25Okapi with IDF fixed to 1 — pure TF-based n-gram matching."""

    def _calc_idf(self, nd: dict) -> None:
        self.idf = {term: 1.0 for term in nd}


def _char_ngrams(text: str, n: int = 3) -> list[str]:
    text = text.lower().strip()
    words = text.split()
    ngrams = []
    for word in words:
        ngrams.extend([word[i:i+n] for i in range(len(word) - n + 1)])
    return ngrams


class GeoSearchIndex:
    """
    In-memory BM25 index over multilingual city names.

    Loading
    -------
    The Parquet file produced by the preprocessing pipeline is exploded so that
    each (name, city) pair becomes its own row.  Names are then grouped back into
    a corpus where each document represents one unique name string — so the BM25
    score reflects how well the query matches the name, not the city.

    Retrieval
    ---------
    1. Tokenise the query into character 3-grams (same as indexing).
    2. Score and retrieve the top-k *name* documents via BM25Okapi.
    3. Collect all geoname_ids behind those names, then re-rank by population
       and deduplicate, returning up to top_k unique cities.

    Persistence
    -----------
    Use ``GeoSearchIndex.from_parquet(path)`` to build, then ``index.save(path)``
    to serialise.  On subsequent starts use ``GeoSearchIndex.load(path)`` — orders
    of magnitude faster than rebuilding from scratch.
    """

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(cls, parquet_path: Path) -> "GeoSearchIndex":
        """Build the index from a processed cities Parquet file."""
        instance = cls.__new__(cls)
        instance._build(parquet_path)
        return instance

    @classmethod
    def load(cls, index_path: Path) -> "GeoSearchIndex":
        """Load a previously saved index from disk (pickle)."""
        with open(index_path, "rb") as f:
            return pickle.load(f)

    # ── persistence ──────────────────────────────────────────────────────────

    def save(self, index_path: Path) -> None:
        """Serialise the index to disk (pickle)."""
        index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(index_path, "wb") as f:
            pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

    # ── public interface ──────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = 20,
        context_id: str | None = None,
    ) -> list[dict]:
        """Return up to top_k cities matching *query*, sorted by population."""
        tokens = _char_ngrams(query)
        logger.info(
            "BM25 tokenized query",
            extra={"context_id": context_id, "stage": "bm25", "tokens": tokens},
        )
        raw_scores = self._bm25.get_scores(tokens)
        name_scores: dict[str, float] = dict(zip(self._corpus, raw_scores))
        top_names: list[str] = self._bm25.get_top_n(tokens, self._corpus, n=top_k)

        # Expand names → (population, geoname_id) candidates; track best BM25 score per city
        candidates: list[tuple[int, int]] = []
        gid_score: dict[int, float] = {}
        for name in top_names:
            score = name_scores[name]
            for gid, pop in zip(
                self._name_to_gids[name], self._name_to_populations[name]
            ):
                candidates.append((pop, gid))
                if gid not in gid_score or score > gid_score[gid]:
                    gid_score[gid] = score

        # Deduplicate and sort by score desc, then population desc
        candidates.sort(key=lambda x: (-gid_score.get(x[1], 0.0), -x[0]))
        seen: set[int] = set()
        results: list[dict] = []
        for _, gid in candidates:
            if gid not in seen:
                seen.add(gid)
                if gid in self._cities:
                    results.append({**self._cities[gid], "score": gid_score.get(gid, 0.0)})
            if len(results) >= top_k:
                break

        return results

    # ── internal ──────────────────────────────────────────────────────────────

    def _build(self, path: Path) -> None:
        flat = (
            pl.read_parquet(path)
            .explode("info")
            .unnest("info")
            .explode("names", "is_preferred")
            .rename({"names": "name"})
            .select(
                "name",
                "geoname_id",
                "ascii_name",
                "country_code",
                "admin1_code",
                "latitude",
                "longitude",
                "population",
            )
            .unique()
            .sort("population", descending=True)
        )

        # City lookup: geoname_id → response dict (one row per unique city)
        self._cities: dict[int, dict] = {
            row["geoname_id"]: {
                "geoname_id":   row["geoname_id"],
                "ascii_name":   row["ascii_name"],
                "country_code": row["country_code"],
                "admin1_code":  row["admin1_code"],
                "latitude":     row["latitude"],
                "longitude":    row["longitude"],
                "population":   row["population"],
            }
            for row in flat.unique("geoname_id").iter_rows(named=True)
        }

        # Name-level corpus for BM25 (each document = one unique name string)
        grouped = flat.group_by("name").agg("*")
        self._corpus: list[str] = grouped["name"].to_list()
        self._bm25 = _BM25Uniform([_char_ngrams(n) for n in self._corpus])

        # Lookup tables used during retrieval re-ranking
        self._name_to_gids: dict[str, list[int]] = {
            d["name"]: d["geoname_id"]
            for d in grouped.select("name", "geoname_id").to_dicts()
        }
        self._name_to_populations: dict[str, list[int]] = {
            d["name"]: d["population"]
            for d in grouped.select("name", "population").to_dicts()
        }
