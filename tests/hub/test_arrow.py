"""Arrow normalisation must reach *nested* large types, losslessly.

The column that motivates this is `entities` in the query dataset:
``large_list<struct<text: large_string, label: large_string, start: int64, end:
int64>>``. A top-level cast would leave the struct's own `large_string` fields
untouched, so the check that matters is recursive.
"""

import pyarrow as pa
import pyarrow.parquet as pq

from src.hub.arrow import normalize_parquet, normalize_table

ENTITY = pa.struct(
    [
        pa.field("text", pa.large_string()),
        pa.field("label", pa.large_string()),
        pa.field("start", pa.int64()),
        pa.field("end", pa.int64()),
    ]
)
SCHEMA = pa.schema(
    [
        pa.field("query", pa.large_string()),
        pa.field("geonameid", pa.large_list(pa.int64())),
        pa.field("entities", pa.large_list(ENTITY)),
        pa.field("confidence", pa.float64()),
    ]
)
ROWS = {
    "query": ["Хочу в Москву", "莫斯科新闻"],
    "geonameid": [[524901, 5601538], [524901]],
    "entities": [
        [{"text": "Москву", "label": "CITY", "start": 7, "end": 13}],
        [{"text": "莫斯科", "label": "CITY", "start": 0, "end": 3}],
    ],
    "confidence": [0.9, 0.8],
}


def _table() -> pa.Table:
    return pa.table(ROWS, schema=SCHEMA)


def test_top_level_large_types_are_rewritten() -> None:
    schema = normalize_table(_table()).schema
    assert schema.field("query").type == pa.string()
    assert schema.field("geonameid").type == pa.list_(pa.int64())


def test_large_types_nested_in_a_struct_are_rewritten() -> None:
    """The case a top-level cast would silently miss."""
    entities = normalize_table(_table()).schema.field("entities").type
    assert pa.types.is_list(entities)
    struct = entities.value_type
    assert struct.field("text").type == pa.string()
    assert struct.field("label").type == pa.string()
    # Non-large types are left exactly as they were.
    assert struct.field("start").type == pa.int64()


def test_values_are_unchanged() -> None:
    """A type rewrite, not a data rewrite."""
    original, normalized = _table(), normalize_table(_table())
    assert normalized.num_rows == original.num_rows
    for column in SCHEMA.names:
        assert (
            normalized.column(column).to_pylist() == original.column(column).to_pylist()
        )


def test_no_large_types_is_a_no_op() -> None:
    plain = pa.table({"a": pa.array([1, 2])})
    assert normalize_table(plain).schema.equals(plain.schema)


def test_round_trips_through_parquet(tmp_path) -> None:
    source = tmp_path / "in.parquet"
    pq.write_table(_table(), source)
    destination = normalize_parquet(source, tmp_path / "nested" / "out.parquet")

    written = pq.read_table(destination)
    assert written.schema.field("query").type == pa.string()
    assert written.column("entities").to_pylist() == ROWS["entities"]
