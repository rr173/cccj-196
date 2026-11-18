"""Edge cases: footer envelope, empty row groups, empty columns, no-Range object."""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import (
    BadSelectionError,
    FooterError,
    RangeProtocolError,
)


def test_object_shorter_than_min_footer_envelope(client, server):
    # 8 bytes: has room for neither a real footer nor a sane envelope.
    payload = b"PAR1" + b"\x00\x00\x00\x00"
    server.put_object("/short.parquet", payload, '"short-1"')
    obj = client.open(server.url_for("/short.parquet"))
    with pytest.raises(FooterError):
        obj.inspect()


def test_bad_trailing_magic_fails_footer(client, server, tmp_path):
    # Valid length, but the trailing 4 bytes are not PAR1.
    payload = bytes(range(256)) * 16  # 4096 bytes, arbitrary content
    server.put_object("/nomagic.parquet", payload, '"nomagic"')
    obj = client.open(server.url_for("/nomagic.parquet"))
    with pytest.raises(FooterError):
        obj.inspect()


def test_footer_length_overruns_object(client, server):
    # Declare a footer length larger than the object can contain.
    body = b"PAR1" + b"\x00" * 32
    declared_len = (10_000).to_bytes(4, "little")
    payload = body + declared_len + b"PAR1"
    server.put_object("/overrun.parquet", payload, '"overrun"')
    obj = client.open(server.url_for("/overrun.parquet"))
    with pytest.raises(FooterError):
        obj.inspect()


def test_object_without_range_capability(client, server, tmp_path):
    path = str(tmp_path / "n.parquet")
    from tests.conftest import write_parquet

    data = write_parquet(path)
    server.put_object("/norange.parquet", data, '"nr"', accept_ranges=False)
    with pytest.raises(RangeProtocolError):
        client.open(server.url_for("/norange.parquet"))


def test_zero_row_group_file_is_inspectable_and_scan_selection_errors(
    client, server, tmp_path
):
    schema = pa.schema([pa.field("a", pa.int32()), pa.field("b", pa.string())])
    path = str(tmp_path / "emptyrg.parquet")
    with pq.ParquetWriter(path, schema) as w:
        pass  # write zero row groups
    data = open(path, "rb").read()
    import hashlib

    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    server.put_object("/emptyrg.parquet", data, etag)
    obj = client.open(server.url_for("/emptyrg.parquet"))
    info = obj.inspect()
    assert info.num_row_groups == 0
    with pytest.raises(BadSelectionError):
        obj.scan(columns=["a"], row_groups=[0])


def test_zero_row_row_group_roundtrip(client, server, tmp_path):
    # A file with one row group containing zero rows but real columns.
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("nested", pa.list_(pa.struct([pa.field("k", pa.string())]))),
    ])
    empty = schema.empty_table()
    path = str(tmp_path / "zerorows.parquet")
    pq.write_table(empty, path)
    data = open(path, "rb").read()
    import hashlib

    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    server.put_object("/zerorows.parquet", data, etag)
    obj = client.open(server.url_for("/zerorows.parquet"))
    got = obj.scan(columns=["id", "nested"], row_groups=[0])
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_group(0, columns=["id", "nested"])
    assert got.schema.equals(ref.schema, check_metadata=False)
    assert got.num_rows == 0


def test_empty_columns_zero_column_table_row_count(client, parquet_url):
    url, _, _ = parquet_url
    obj = client.open(url)
    t = obj.scan(columns=[], row_groups=[0, 2, 3])
    assert t.num_columns == 0
    assert t.num_rows == 150  # 3 groups x 50 rows


def test_full_nested_selection_matches_local_reference(client, parquet_url):
    url, data, _ = parquet_url
    obj = client.open(url)
    cols = ["id", "tags", "meta", "cat", "blob", "wide1", "wide2"]
    got = obj.scan(columns=cols, row_groups=[0, 1, 2, 3])
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_groups([0, 1, 2, 3], columns=cols)
    assert got.equals(ref)
    # schema (incl. nullability / nesting) and row count agree exactly
    assert got.schema.equals(ref.schema, check_metadata=False)
    assert got.num_rows == ref.num_rows


def test_nested_leaf_subset_struct_matches_reference(client, parquet_url):
    # Selecting only part of a struct still maps leaves correctly.
    url, data, _ = parquet_url
    obj = client.open(url)
    got = obj.scan(columns=["meta"], row_groups=[1])
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_group(1, columns=["meta"])
    assert got.equals(ref)
    # nulls at every nesting level are preserved
    got_py = got.to_pylist()
    assert any(row["meta"] is None for row in got_py)
    assert any(row["meta"] is not None and row["meta"]["x"] is None for row in got_py)
