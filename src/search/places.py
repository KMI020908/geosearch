"""Hydration: turning retrieved geonameids back into displayable places.

Retrieval only ever produces ids. The BM25 corpus stores ``(geonameid,
population)`` pairs — enough to rank, nothing to show — so every result has to
be filled in from somewhere before it can be returned.

A protocol with one implementation, deliberately. Hydration is the seam where a
different backing store would plug in — a remote lookup, a cache, a shard — and
keeping it named means :meth:`~src.search.engine.SearchEngine.search` never has
to know. It also documents the contract the artifact builder has to satisfy:
resolve what you know, omit what you do not.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import polars as pl


@dataclass(frozen=True)
class Place:
    """The stored fields of one populated place."""

    geonameid: int
    asciiname: str
    country_code: str
    population: int
    feature_code: str | None
    latitude: float | None
    longitude: float | None


class PlaceStore(Protocol):
    """Look up places by id.

    Implementations must return only the ids they know: a missing id is omitted
    from the mapping, never defaulted. Callers rely on that — a geonameid the
    store cannot resolve is dropped from the result set rather than surfacing as
    a row of zeroes.
    """

    async def fetch(self, geonameids: Sequence[int]) -> dict[int, Place]: ...


class ParquetPlaceStore:
    """Hydrate from ``places.parquet``, with no database at all.

    Holds the table as an Arrow-backed Polars frame rather than a dict of a
    million dataclasses: ~60 MB resident against several hundred, and lookups
    are a semi-join on a tiny id frame — sub-millisecond, against the ~1 ms the
    BM25 scan itself costs.

    The frame is read-only after construction, so one instance is shared across
    concurrent requests exactly as the index and the model are.
    """

    def __init__(self, frame: pl.DataFrame) -> None:
        self._frame = frame

    @classmethod
    def from_parquet(cls, path: str | Path) -> ParquetPlaceStore:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"{path} is missing — build it with `make artifacts` or fetch it "
                "with `make hub-pull`."
            )
        return cls(pl.read_parquet(path))

    def __len__(self) -> int:
        return self._frame.height

    async def fetch(self, geonameids: Sequence[int]) -> dict[int, Place]:
        if not geonameids:
            return {}
        wanted = pl.DataFrame(
            {"geonameid": list(geonameids)}, schema={"geonameid": pl.Int64}
        )
        rows = self._frame.join(wanted, on="geonameid", how="semi")
        return {
            row["geonameid"]: Place(
                geonameid=row["geonameid"],
                asciiname=row["asciiname"],
                country_code=row["country_code"],
                population=row["population"],
                feature_code=row["feature_code"],
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
            for row in rows.iter_rows(named=True)
        }
