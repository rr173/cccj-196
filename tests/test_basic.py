"""Correctness: identity pinning, inspect, scan vs local full-download decode."""

from __future__ import annotations

import io

import pyarrow.parquet as pq
import pytest

from parquet_range import (
    BadSelectionError,
    FooterError,
    ObjectIdentity,
    canonical_url,
)
from parquet_range.identity import require_strong_etag
from parquet_range.exceptions import WeakOrMissingETagError


# ------------------------------------------------------------- identity
def test_canonical_url_normalization():
    assert canonical_url("HTTP://Example.COM:80/a/../b") == "http://example.com/b"
    assert (
        canonical_url("https://host:443/x?z=1") == "https://host/x?z=1"
    )
    assert canonical_url("http://h:8080/a/") == "http://h:8080/a/"
    with pytest.raises(ValueError):
        canonical_url("ftp://host/a")


def test_strong_etag_required():
    assert require_strong_etag('"abc123"') == '"abc123"'
    with pytest.raises(WeakOrMissingETagError):
        require_strong_etag('W/"abc"')
    with pytest.raises(WeakOrMissingETagError):
        require_strong_etag(None)
    with pytest.raises(WeakOrMissingETagError):
        require_strong_etag("not-quoted")


def test_identity_version_key_includes_version():
    a = ObjectIdentity("http://h/a", '"v1"', 1000)
    b = ObjectIdentity("http://h/a", '"v2"', 1000)
    assert a.version_id() != b.version_id()
    assert a.block_key(3) != b.block_key(3)
    with pytest.raises(Exception):
        ObjectIdentity("http://h/a", '"v"', 0)


# ------------------------------------------------------------- inspect
def test_inspect_reports_metadata(client, parquet_url):
    url, data, etag = parquet_url
    obj = client.open(url)
    info = obj.inspect()
    md = pq.read_metadata(io.BytesIO(data))
    assert info.num_row_groups == md.num_row_groups
    assert info.num_rows == md.num_rows
    assert info.num_columns == md.num_columns
    assert info.etag == etag
    assert info.length == len(data)
    assert info.schema.names == md.schema.to_arrow_schema().names
    assert info.schema.names == ["id", "tags", "meta", "cat", "blob", "wide1", "wide2"]
    # nested leaves are exposed with full paths
    paths = {c["path"] for c in info.column_chunk_info}
    assert {"id", "tags.list.element", "meta.x", "meta.y",
            "meta.flags.list.element", "cat", "blob", "wide1", "wide2"} <= paths
    # dictionary pages were planned from real offsets
    cat = next(c for c in info.column_chunk_info if c["path"] == "cat")
    assert cat["dictionary_page_offset"] is not None
    assert "RLE_DICTIONARY" in cat["encodings"]


# ------------------------------------------------------------- scan
def test_scan_matches_local_reference_nested_dict_nulls(client, parquet_url):
    url, data, _ = parquet_url
    obj = client.open(url)
    columns = ["id", "tags", "meta", "cat", "blob"]
    got = obj.scan(columns=columns, row_groups=[0, 2])
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_groups([0, 2], columns=columns)
    assert got.equals(ref)
    assert got.schema.equals(ref.schema, check_metadata=False)
    assert got.num_rows == ref.num_rows


def test_scan_single_row_group_subset(client, parquet_url):
    url, data, _ = parquet_url
    obj = client.open(url)
    got = obj.scan(columns=["meta", "id"], row_groups=[3])
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_group(3, columns=["meta", "id"])
    assert got.equals(ref)


def test_scan_sparse_transfer_far_smaller_than_object(client, wide_parquet_url, server):
    url, data, _ = wide_parquet_url
    obj = client.open(url)
    # Select only a narrow column from one row group of a deliberately wide file.
    obj.scan(columns=["id"], row_groups=[0])
    # Evidence from both sides: client wire accounting and the real server log.
    wire = client.stats.snapshot()["wire_bytes"]
    server_bytes = server.physical_bytes("/wide.parquet")
    assert server_bytes == wire
    assert wire < len(data) // 4, (wire, len(data))
    # And the full file must never have been requested: no byte-0 open-ended
    # or whole-object 200 response appears in the real server log.
    gets = [e for e in server.log if e.path == "/wide.parquet" and e.method == "GET"]
    assert all(entry.status == 206 for entry in gets)
    assert not any(
        entry.range_header in (None, "bytes=0-") for entry in gets
    )


def test_scan_empty_columns_returns_zero_column_table_with_row_count(client, parquet_url):
    url, _, _ = parquet_url
    obj = client.open(url)
    t = obj.scan(columns=[], row_groups=[0, 1])
    assert t.num_columns == 0
    assert t.num_rows == 100  # two groups x 50 rows
    # zero-column scans must not pull column bytes
    before = client.stats.snapshot()["physical_requests"]
    obj.scan(columns=[], row_groups=[2])
    after = client.stats.snapshot()["physical_requests"]
    assert after == before


def test_scan_bad_selection(client, parquet_url):
    url, _, _ = parquet_url
    obj = client.open(url)
    with pytest.raises(BadSelectionError):
        obj.scan(columns=["nope"], row_groups=[0])
    with pytest.raises(BadSelectionError):
        obj.scan(columns=["id"], row_groups=[99])
    with pytest.raises(BadSelectionError):
        obj.scan(columns=["id"], row_groups=[])
    with pytest.raises(BadSelectionError):
        obj.scan(columns=["id"], row_groups=[0, 0])


def test_file_smaller_than_footer_envelope(client, server, tmp_path):
    tiny = b"PAR1"  # 4 bytes, no footer envelope possible
    server.put_object("/tiny.parquet", tiny, '"tiny"')
    url = server.url_for("/tiny.parquet")
    server.clear_log()
    obj = client.open(url)
    with pytest.raises(FooterError):
        obj.inspect()
