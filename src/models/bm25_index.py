"""BM25-based geo-search index backed by a GeoNames Parquet file."""
import logging
import pickle
from pathlib import Path

import polars as pl
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


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
        self._bm25 = BM25Okapi([_char_ngrams(n) for n in self._corpus])

        # Lookup tables used during retrieval re-ranking
        self._name_to_gids: dict[str, list[int]] = {
            d["name"]: d["geoname_id"]
            for d in grouped.select("name", "geoname_id").to_dicts()
        }
        self._name_to_populations: dict[str, list[int]] = {
            d["name"]: d["population"]
            for d in grouped.select("name", "population").to_dicts()
        }



# """BM25-based geo-search index backed by GeoNames Parquet files."""
# import pickle
# from pathlib import Path

# import numpy as np
# import polars as pl
# from country_list import countries_for_language
# from rank_bm25 import BM25Okapi


# def _country_names_for_languages(languages: list[str]) -> dict[str, dict[str, str]]:
#     """Return {lang: {country_code: name}}."""
#     result: dict[str, dict[str, str]] = {}
#     for lang in languages:
#         result[lang] = dict(countries_for_language(lang))
#     return result


# def _char_ngrams(text: str, n: int = 3) -> list[str]:
#     text = text.lower().strip()
#     return [text[i : i + n] for i in range(len(text) - n + 1)]


# class _BM25Uniform(BM25Okapi):
#     """BM25Okapi with uniform IDF = 1.0 for every term.

#     Standard BM25 penalises high-frequency terms — a city name like "Moscow"
#     that appears in many documents gets a near-zero IDF and is effectively
#     invisible to the scorer.  Setting all IDF values to 1.0 means scores
#     reflect only term frequency and document-length normalisation, which is
#     the correct behaviour for city-name retrieval.
#     """

#     def _calc_idf(self, nd: dict[str, int]) -> None:
#         for word in nd:
#             self.idf[word] = 1.0
#         self.average_idf = 1.0


# class GeoSearchIndex:
#     """
#     In-memory BM25 index over exploded multilingual city name documents.

#     Document structure
#     ------------------
#     Each city is exploded so that every individual name variant becomes its
#     own BM25 document.  Each document contains only same-language context:

#         {name}  {country_name_in_same_language}  {region_name_in_same_language}

#     For example, the Russian name "Москва" is paired with "Россия" and
#     "Москва" (oblast), while the English name "Moscow" is paired with
#     "Russia" and "Moscow Oblast".

#     Documents are tokenised into character 3-grams (same as the original
#     implementation), which provides fuzzy matching across transliterations
#     and minor spelling variations.

#     Retrieval
#     ---------
#     1. Tokenise the query into character 3-grams.
#     2. Score all name documents with :class:`_BM25Uniform` (uniform IDF).
#     3. Collect the best score per geoname_id across all its name documents.
#     4. Return the top-k cities sorted by score descending.

#     Persistence
#     -----------
#     Build with :meth:`from_parquet`, then :meth:`save`.  Reload with
#     :meth:`load` — much faster than rebuilding from scratch.
#     """

#     _LANGUAGES = ["en", "tr", "ru"]

#     # ── constructors ─────────────────────────────────────────────────────────

#     @classmethod
#     def from_parquet(
#         cls,
#         cities_path: Path,
#         regions_path: Path,
#         k1: float = 1.5,
#         b: float = 0.75,
#     ) -> "GeoSearchIndex":
#         """Build the index from pre-processed cities and regions Parquet files."""
#         instance = cls.__new__(cls)
#         instance._build(cities_path, regions_path, k1=k1, b=b)
#         return instance

#     @classmethod
#     def load(cls, index_path: Path) -> "GeoSearchIndex":
#         """Load a previously saved index from disk."""
#         with open(index_path, "rb") as f:
#             return pickle.load(f)

#     # ── persistence ──────────────────────────────────────────────────────────

#     def save(self, index_path: Path) -> None:
#         """Serialise the index to disk."""
#         index_path.parent.mkdir(parents=True, exist_ok=True)
#         with open(index_path, "wb") as f:
#             pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

#     # ── public interface ──────────────────────────────────────────────────────

#     def search(self, query: str, top_k: int = 20) -> list[dict]:
#         """Return up to *top_k* city dicts ranked by BM25 score."""
#         tokens = _char_ngrams(query)
#         if not tokens:
#             return []

#         scores: np.ndarray = self._bm25.get_scores(tokens)

#         # Best score per geoname_id across all its name documents
#         gid_score: dict[int, float] = {}
#         for doc_idx, score in enumerate(scores):
#             if score <= 0:
#                 continue
#             gid = self._doc_gids[doc_idx]
#             if score > gid_score.get(gid, 0.0):
#                 gid_score[gid] = score

#         sorted_gids = sorted(gid_score, key=gid_score.__getitem__, reverse=True)[:top_k]
#         return [
#             {**self._cities[gid], "score": float(gid_score[gid])}
#             for gid in sorted_gids
#         ]

#     # ── internal ──────────────────────────────────────────────────────────────

#     def _build(
#         self, cities_path: Path, regions_path: Path, k1: float, b: float
#     ) -> None:
#         cities = pl.read_parquet(cities_path)

#         # (country_code, admin1_code, language) → region name
#         region_lookup: dict[tuple[str, str, str], str] = {}
#         for row in pl.read_parquet(regions_path).iter_rows(named=True):
#             key = (row["country_code"], row["admin1_code"], row["language"])
#             region_lookup[key] = row["name"]

#         # {lang: {country_code: name}}
#         country_lookup = _country_names_for_languages(self._LANGUAGES)

#         self._cities: dict[int, dict] = {}
#         self._doc_gids: list[int] = []
#         corpus: list[list[str]] = []

#         for row in cities.iter_rows(named=True):
#             gid = row["geoname_id"]
#             country_code = row["country_code"]
#             admin1_code = row["admin1_code"]

#             self._cities[gid] = {
#                 "geoname_id":   gid,
#                 "ascii_name":   row["ascii_name"],
#                 "country_code": country_code,
#                 "admin1_code":  admin1_code,
#                 "latitude":     row["latitude"],
#                 "longitude":    row["longitude"],
#                 "population":   row["population"],
#             }

#             seen_names: set[str] = set()
#             for lang_entry in row["info"]:
#                 lang = lang_entry["language"]
#                 country_name = country_lookup.get(lang, {}).get(country_code, "")
#                 region_name = region_lookup.get((country_code, admin1_code, lang), "")

#                 for name in lang_entry["names"]:
#                     if name in seen_names:
#                         continue
#                     seen_names.add(name)

#                     doc = " ".join(filter(None, [name, country_name, region_name]))
#                     corpus.append(_char_ngrams(doc))
#                     self._doc_gids.append(gid)

#         self._bm25 = _BM25Uniform(corpus, k1=k1, b=b)
