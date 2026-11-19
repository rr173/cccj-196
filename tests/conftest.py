"""Shared pytest fixtures: real Range server, rich Parquet files, clients."""

from __future__ import annotations

import hashlib
import io
import os
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Load the diff-feature and quality-audit fixtures/helpers as pytest plugins
# so they share the same collection as the rest of the suite without extra
# conftest.py files.
pytest_plugins = ["diff_conftest", "quality_conftest"]

from parquet_range import ClientConfig, RemoteParquetClient
from parquet_range.test_server import TestHTTPServer

SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("tags", pa.list_(pa.string())),
    pa.field(
        "meta",
        pa.struct([
            pa.field("x", pa.int32()),
            pa.field("y", pa.string()),
            pa.field("flags", pa.list_(pa.int32())),
        ]),
    ),
    pa.field("cat", pa.dictionary(pa.int8(), pa.string())),
    pa.field("blob", pa.binary()),
    pa.field("wide1", pa.float64()),
    pa.field("wide2", pa.float64()),
])


def make_row_group(seed: int, n: int = 50) -> pa.Table:
    ids = list(range(seed, seed + n))
    tags: list = []
    metas: list = []
    cats: list = []
    blobs: list = []
    for i in range(n):
        if i % 11 == 0:
            tags.append(None)
        elif i % 7 == 0:
            tags.append([])
        else:
            tags.append([f"t{(seed + i) % 5}", "z"])
        if i % 13 == 0:
            metas.append(None)
        else:
            metas.append({
                "x": None if i % 3 == 0 else seed + i,
                "y": None if i % 4 == 0 else f"y{i % 6}",
                "flags": None if i % 5 == 0 else [i % 4, (i + 1) % 4],
            })
        cats.append(None if i % 9 == 0 else ["alpha", "beta", "gamma"][i % 3])
        blobs.append(None if i % 6 == 0 else bytes([i % 256]) * (4 + i % 8))
    return pa.table({
        "id": ids,
        "tags": tags,
        "meta": metas,
        "cat": pa.array(cats, type=pa.string()).dictionary_encode(),
        "blob": blobs,
        "wide1": [float(seed + i) for i in range(n)],
        "wide2": [float(seed - i) for i in range(n)],
    }, schema=SCHEMA)


def write_parquet(path: str, groups: int = 4, rows: int = 50) -> bytes:
    with pq.ParquetWriter(path, SCHEMA, compression="snappy", use_dictionary=True) as w:
        for g in range(groups):
            w.write_table(make_row_group(g * 1000, rows), row_group_size=rows)
    return open(path, "rb").read()


def write_wide_parquet(path: str, groups: int = 6, rows: int = 4000) -> bytes:
    """Larger file: many wide columns so a one-column scan stays clearly sparse."""
    wide_fields = [pa.field(f"wide_{i}", pa.float64()) for i in range(12)]
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field(
            "meta",
            pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())]),
        ),
        *wide_fields,
    ])
    import random

    rng = random.Random(42)
    with pq.ParquetWriter(path, schema, compression="snappy", use_dictionary=True) as w:
        for g in range(groups):
            cols = {
                "id": list(range(g * rows, (g + 1) * rows)),
                "name": [f"name-{rng.randint(0, 999)}" for _ in range(rows)],
                "meta": [
                    {"a": None if i % 17 == 0 else i, "b": f"s{i % 11}"}
                    for i in range(rows)
                ],
            }
            for i in range(12):
                cols[f"wide_{i}"] = [rng.random() * 1e6 for _ in range(rows)]
            w.write_table(pa.table(cols, schema=schema), row_group_size=rows)
    return open(path, "rb").read()


def write_wide_parquet_v2(path: str, groups: int = 6, rows: int = 4000) -> bytes:
    """A structurally valid *new version* of the wide file (same schema, other data)."""
    wide_fields = [pa.field(f"wide_{i}", pa.float64()) for i in range(12)]
    schema = pa.schema([
        pa.field("id", pa.int64()),
        pa.field("name", pa.string()),
        pa.field(
            "meta",
            pa.struct([pa.field("a", pa.int32()), pa.field("b", pa.string())]),
        ),
        *wide_fields,
    ])
    import random

    rng = random.Random(987654321)
    with pq.ParquetWriter(path, schema, compression="snappy", use_dictionary=True) as w:
        for g in range(groups):
            cols = {
                "id": list(range(10_000_000 + g * rows, 10_000_000 + (g + 1) * rows)),
                "name": [f"v2-{rng.randint(0, 9999)}" for _ in range(rows)],
                "meta": [
                    {"a": rng.randint(0, 999), "b": f"v2-{i % 13}"} for i in range(rows)
                ],
            }
            for i in range(12):
                cols[f"wide_{i}"] = [rng.random() * -1e6 for _ in range(rows)]
            w.write_table(pa.table(cols, schema=schema), row_group_size=rows)
    return open(path, "rb").read()


@pytest.fixture(scope="session")
def server():
    srv = TestHTTPServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def parquet_url(tmp_path, server):
    path = str(tmp_path / "sample.parquet")
    data = write_parquet(path)
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    server.put_object("/sample.parquet", data, etag)
    server.clear_log()
    server.fault.reset()
    server.barrier.reset()
    yield server.url_for("/sample.parquet"), data, etag
    server.fault.reset()
    server.barrier.reset()


@pytest.fixture()
def wide_parquet_url(tmp_path, server):
    path = str(tmp_path / "wide.parquet")
    data = write_wide_parquet(path)
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    server.put_object("/wide.parquet", data, etag)
    server.clear_log()
    server.fault.reset()
    server.barrier.reset()
    yield server.url_for("/wide.parquet"), data, etag
    server.fault.reset()
    server.barrier.reset()


@pytest.fixture()
def client_factory(tmp_path):
    created: list[RemoteParquetClient] = []

    def factory(**overrides) -> RemoteParquetClient:
        cfg = ClientConfig(
            block_size=overrides.pop("block_size", 64 * 1024),
            mem_budget=overrides.pop("mem_budget", 4 << 20),
            disk_budget=overrides.pop("disk_budget", 8 << 20),
            max_inflight_bytes=overrides.pop("max_inflight_bytes", 4 << 20),
            max_inflight_requests=overrides.pop("max_inflight_requests", 4),
            max_merge_gap_blocks=overrides.pop("max_merge_gap_blocks", 1),
            max_retries=overrides.pop("max_retries", 3),
        )
        disk_dir = overrides.pop("disk_dir", str(tmp_path / "diskcache"))
        c = RemoteParquetClient(disk_dir=disk_dir, config=cfg)
        created.append(c)
        return c

    yield factory
    for c in created:
        c.close()


@pytest.fixture()
def client(client_factory):
    return client_factory()


def reference_scan(data: bytes, columns: list[str], row_groups: list[int]) -> pa.Table:
    """Independent full-download reference decode with local PyArrow."""
    pf = pq.ParquetFile(io.BytesIO(data))
    return pf.read_row_groups(row_groups, columns=columns)
