"""Normalise Arrow types so the HuggingFace Dataset Viewer can render a table.

Polars writes strings as ``large_string`` and lists as ``large_list`` — correct
Arrow, and what every Parquet this project produces carries. The viewer's type
inference does cope with those, but no other dataset on the Hub looks like this
and the viewer is a moving target, so the published copies are rewritten to the
ordinary types instead of betting on it.

The rewrite is lossless and recursive: ``entities`` in the query dataset is
``large_list<struct<text: large_string, label: large_string, start: int64, end:
int64>>``, i.e. the large types are *nested*, so a top-level cast would not
reach them. Values are unchanged; only the type tags differ.

Local files stay as polars wrote them — this runs on the way out, in
:mod:`src.hub.push`.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def normalize_type(dtype: pa.DataType) -> pa.DataType:
    """Rewrite the large-variant Arrow types to their ordinary equivalents."""
    if pa.types.is_large_string(dtype):
        return pa.string()
    if pa.types.is_large_binary(dtype):
        return pa.binary()
    if pa.types.is_large_list(dtype):
        return pa.list_(normalize_field(dtype.value_field))
    if pa.types.is_list(dtype):
        return pa.list_(normalize_field(dtype.value_field))
    if pa.types.is_struct(dtype):
        return pa.struct(
            [normalize_field(dtype.field(i)) for i in range(dtype.num_fields)]
        )
    if pa.types.is_map(dtype):
        return pa.map_(normalize_type(dtype.key_type), normalize_type(dtype.item_type))
    return dtype


def normalize_field(field: pa.Field) -> pa.Field:
    """Rewrite one field's type, keeping its name and nullability."""
    return field.with_type(normalize_type(field.type))


def normalize_schema(schema: pa.Schema) -> pa.Schema:
    """Rewrite every field, preserving the schema's own metadata."""
    return pa.schema(
        [normalize_field(schema.field(i)) for i in range(len(schema))],
        metadata=schema.metadata,
    )


def normalize_table(table: pa.Table) -> pa.Table:
    """Return *table* with large types rewritten; a no-op when there are none."""
    schema = normalize_schema(table.schema)
    if schema.equals(table.schema):
        return table
    return table.cast(schema)


def normalize_parquet(source: Path, destination: Path) -> Path:
    """Copy a Parquet file, normalising its types on the way."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    table = normalize_table(pq.read_table(source))
    pq.write_table(table, destination, compression="zstd")
    return destination
