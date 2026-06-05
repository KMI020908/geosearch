"""BM25-based geo-search index backed by a GeoNames Parquet file."""
import logging
import pickle
from pathlib import Path

import polars as pl
from country_list import countries_for_language
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


def _build_country_df(languages: list[str]) -> pl.DataFrame:
    """Return a DataFrame with columns (country_code, language, country_name)."""
    rows = []
    for lang in languages:
        try:
            lang_map = dict(countries_for_language(lang))
        except Exception:
            continue
        for cc, cname in lang_map.items():
            rows.append({"country_code": cc, "language": lang, "country_name": cname})
    return pl.DataFrame(rows, schema={"country_code": pl.String, "language": pl.String, "country_name": pl.String})


class BM25Index:
    """
    In-memory BM25 index over multilingual city names enriched with country and region.

    Loading
    -------
    The Parquet file produced by the preprocessing pipeline is exploded so that
    each (name, city) pair becomes its own row.  Each name is enriched with the
    country name and region name in the same language, forming a document like
    "Москва Россия Москва" for the Russian entry of Moscow.  Documents are then
    grouped by this enriched string so the BM25 score reflects how well the query
    matches the full location context.

    Retrieval
    ---------
    1. Tokenise the query into character 3-grams (same as indexing).
    2. Score and retrieve the top-k *document* entries via BM25Okapi.
    3. Collect all geoname_ids behind those documents, then re-rank by BM25 score
       and deduplicate, returning up to top_k unique cities.

    Persistence
    -----------
    Use ``BM25Index.from_parquet(path)`` to build, then ``index.save(path)``
    to serialise.  On subsequent starts use ``BM25Index.load(path)`` — orders
    of magnitude faster than rebuilding from scratch.
    """

    # ── constructors ─────────────────────────────────────────────────────────

    @classmethod
    def from_parquet(
        cls,
        parquet_path: Path,
        regions_path: Path | None = None,
    ) -> "BM25Index":
        """Build the index from a processed cities Parquet file.

        Parameters
        ----------
        parquet_path:  Path to the processed cities Parquet file.
        regions_path:  Path to the regions Parquet file used to enrich documents
                       with region names.  Defaults to ``regions.parquet`` in the
                       same directory as *parquet_path*.
        """
        if regions_path is None:
            regions_path = parquet_path.parent / "regions.parquet"
        instance = cls.__new__(cls)
        instance._build(parquet_path, regions_path)
        return instance

    @classmethod
    def load(cls, index_path: Path) -> "BM25Index":
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
        """Return up to top_k cities matching *query*, sorted by BM25 score then population."""
        tokens = _char_ngrams(query)
        logger.info(
            "BM25 tokenized query",
            extra={"context_id": context_id, "stage": "bm25", "tokens": tokens},
        )
        raw_scores = self._bm25.get_scores(tokens)
        doc_scores: dict[str, float] = dict(zip(self._corpus, raw_scores))
        top_docs: list[str] = self._bm25.get_top_n(tokens, self._corpus, n=top_k)

        # Expand docs → (population, geoname_id) candidates; track best BM25 score per city
        candidates: list[tuple[int, int]] = []
        gid_score: dict[int, float] = {}
        for doc in top_docs:
            score = doc_scores[doc]
            for gid, pop in zip(
                self._doc_to_gids[doc], self._doc_to_populations[doc]
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

    def _build(self, path: Path, regions_path: Path) -> None:
        flat = (
            pl.read_parquet(path)
            .explode("info")
            .unnest("info")
            .explode("names", "is_preferred")
            .rename({"names": "name"})
            .select(
                "name",
                "language",
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

        # Build country name lookup and join
        languages = flat["language"].unique().to_list()
        country_df = _build_country_df(languages)
        regions_df = pl.read_parquet(regions_path).rename({"name": "region_name"})

        flat = (
            flat
            .join(country_df, on=["country_code", "language"], how="left")
            .join(regions_df, on=["admin1_code", "country_code", "language"], how="left")
            .with_columns(
                pl.concat_str(
                    ["name", "country_name", "region_name"],
                    separator=" ",
                    ignore_nulls=True,
                ).alias("doc")
            )
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

        # Doc-level corpus for BM25 (each document = "city_name country_name region_name")
        grouped = flat.group_by("doc").agg("*")
        self._corpus: list[str] = grouped["doc"].to_list()
        self._bm25 = _BM25Uniform([_char_ngrams(d) for d in self._corpus])

        # Lookup tables used during retrieval re-ranking
        self._doc_to_gids: dict[str, list[int]] = {
            d["doc"]: d["geoname_id"]
            for d in grouped.select("doc", "geoname_id").to_dicts()
        }
        self._doc_to_populations: dict[str, list[int]] = {
            d["doc"]: d["population"]
            for d in grouped.select("doc", "population").to_dicts()
        }
