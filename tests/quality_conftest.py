"""Shared fixtures for the quality-audit feature tests.

Everything runs against the real local Range HTTP server; the expected
violations come from the independent oracle in ``quality_oracle.py`` (which
never touches the product's rule executor).
"""

from __future__ import annotations

import datetime
import hashlib
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import (
    ClientConfig,
    QualityAuditRequest,
    QualityContract,
    RemoteParquetClient,
    SnapshotManifest,
)
from parquet_range.test_server import TestHTTPServer

QUALITY_SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("small_i", pa.int8()),
    pa.field("big_u", pa.uint64()),
    pa.field("byte_u", pa.uint8()),
    pa.field("score", pa.float64()),
    pa.field("ratio", pa.float32()),
    pa.field("flag", pa.bool_()),
    pa.field("name", pa.string()),
    pa.field("ccy", pa.string()),
    pa.field("created", pa.timestamp("us", tz="UTC")),
    pa.field("updated", pa.timestamp("us", tz="UTC")),
    pa.field("ts_ns", pa.timestamp("ns")),
    pa.field("day", pa.date32()),
    # Nested + wide columns exist in the file but are never contract-referenced.
    pa.field("meta", pa.struct([
        pa.field("x", pa.int32()),
        pa.field("label", pa.string()),
    ])),
    pa.field("tags", pa.list_(pa.string())),
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(6)],
])

CONTRACT_FIELDS = {
    "id": "int64",
    "small_i": "int8",
    "big_u": "uint64",
    "byte_u": "uint8",
    "score": "float64",
    "ratio": "float32",
    "flag": "bool",
    "name": "string",
    "ccy": "string",
    "created": "timestamp[us, tz=UTC]",
    "updated": "timestamp[us, tz=UTC]",
    "ts_ns": "timestamp[ns]",
    "day": "date32",
}

_BASE_TS = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
_BASE_NS = datetime.datetime(2022, 1, 1)
_BASE_DAY = datetime.date(2022, 1, 1)


def quality_contract() -> dict:
    """The canonical test contract: every rule kind, every semantic corner."""
    return {
        "spec_version": 1,
        "fields": dict(CONTRACT_FIELDS),
        "rules": [
            {"id": "id-non-null", "kind": "not_null", "field": "id"},
            {"id": "small-range", "kind": "range", "field": "small_i",
             "min": -50, "max": 50},
            {"id": "big-u-range", "kind": "range", "field": "big_u",
             "min": 0, "max": 2 ** 64 - 100},
            {"id": "byte-u-range", "kind": "range", "field": "byte_u",
             "min": 10, "max": 250,
             "min_inclusive": False, "max_inclusive": False},
            {"id": "score-range", "kind": "range", "field": "score",
             "min": 0.0, "max": 1000.0, "nan_is_violation": True},
            {"id": "ratio-range", "kind": "range", "field": "ratio",
             "min": -1.5, "max": 1.5},
            {"id": "flag-enum", "kind": "enum", "field": "flag",
             "values": [True]},
            {"id": "ccy-enum", "kind": "enum", "field": "ccy",
             "values": ["USD", "EUR", "JPY"]},
            {"id": "name-pattern", "kind": "pattern", "field": "name",
             "pattern": "^[A-Z][a-z0-9_]{2,15}$"},
            {"id": "ts-order", "kind": "order",
             "left": "created", "right": "updated"},
            {"id": "ts-ns-range", "kind": "range", "field": "ts_ns",
             "min": "2021-01-01T00:00:00", "max": "2030-01-01T00:00:00"},
            {"id": "day-range", "kind": "range", "field": "day",
             "min": "2020-01-01", "max": "2030-12-31"},
        ],
    }


def make_quality_rows(n: int) -> list[dict]:
    """Deterministic rows exercising nulls, NaN, uint64 edges, tz timestamps."""
    rows: list[dict] = []
    for i in range(n):
        rows.append({
            "id": None if i % 13 == 0 else i,
            "small_i": (i % 200) - 100,
            "big_u": i if i % 5 == 0 else (2 ** 64 - 1 - i),
            "byte_u": i % 256,
            "score": float("nan") if i % 11 == 0 else float((i * 37) % 2000),
            "ratio": ((i % 40) - 20) / 10.0,
            "flag": i % 2 == 0,
            "name": f"bad name {i}" if i % 3 == 0 else f"Name_{i}",
            "ccy": ["USD", "EUR", "JPY", "GBP"][i % 4],
            "created": _BASE_TS + datetime.timedelta(seconds=i),
            "updated": _BASE_TS + datetime.timedelta(seconds=i + (i % 7) - 3),
            "ts_ns": (
                _BASE_NS - datetime.timedelta(days=400)
                if i % 17 == 0
                else _BASE_NS + datetime.timedelta(seconds=i * 30)
            ),
            "day": (
                datetime.date(2018, 6, 1)
                if i % 19 == 0
                else _BASE_DAY + datetime.timedelta(days=i % 800)
            ),
            "meta": None if i % 4 == 0 else {"x": i, "label": f"m{i}"},
            "tags": None if i % 6 == 0 else [f"t{i}", "z"],
            **{f"wide_{k}": float((i * 31 + k * 17) % 1000) / 7.0
               for k in range(6)},
        })
    return rows


def make_clean_rows(n: int) -> list[dict]:
    """Rows that violate *nothing* in quality_contract()."""
    rows: list[dict] = []
    for i in range(n):
        rows.append({
            "id": i + 1,
            "small_i": 0,
            "big_u": i,
            "byte_u": 100,
            "score": 500.0,
            "ratio": 0.5,
            "flag": True,
            "name": f"Name_{i}",
            "ccy": "USD",
            "created": _BASE_TS + datetime.timedelta(seconds=i),
            "updated": _BASE_TS + datetime.timedelta(seconds=i + 60),
            "ts_ns": _BASE_NS + datetime.timedelta(seconds=i * 30),
            "day": _BASE_DAY + datetime.timedelta(days=i),
            "meta": {"x": i, "label": f"m{i}"},
            "tags": [f"t{i}"],
            **{f"wide_{k}": 1.0 for k in range(6)},
        })
    return rows


def write_quality_parquet(path: str, rows: list[dict], *, groups: int) -> bytes:
    with pq.ParquetWriter(path, QUALITY_SCHEMA, compression="snappy",
                          use_dictionary=True) as writer:
        if rows:
            per = max(1, (len(rows) + groups - 1) // groups)
            for g in range(groups):
                part = rows[g * per:] if g == groups - 1 else rows[g * per:(g + 1) * per]
                if part:
                    table = pa.table(
                        {name: [r[name] for r in part] for name in QUALITY_SCHEMA.names},
                        schema=QUALITY_SCHEMA,
                    )
                    writer.write_table(table, row_group_size=len(part))
    with open(path, "rb") as fh:
        return fh.read()


def etag_of(data: bytes) -> str:
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


@pytest.fixture(scope="session")
def quality_server():
    srv = TestHTTPServer()
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def quality_snapshot(tmp_path, quality_server):
    """A published multi-row-group snapshot; returns identity + raw bytes."""
    path = str(tmp_path / "quality.parquet")
    rows = make_quality_rows(111)  # 3 row groups of 37
    data = write_quality_parquet(path, rows, groups=3)
    etag = etag_of(data)
    quality_server.put_object("/quality.parquet", data, etag)
    quality_server.clear_log()
    quality_server.fault.reset()
    quality_server.barrier.reset()
    yield {
        "url": quality_server.url_for("/quality.parquet"),
        "etag": etag,
        "length": len(data),
        "data": data,
        "rows": rows,
    }
    quality_server.fault.reset()
    quality_server.barrier.reset()


@pytest.fixture()
def quality_client_factory(tmp_path):
    created: list[RemoteParquetClient] = []

    def factory(**overrides) -> RemoteParquetClient:
        cfg = ClientConfig(
            block_size=overrides.pop("block_size", 32 * 1024),
            mem_budget=overrides.pop("mem_budget", 2 << 20),
            disk_budget=overrides.pop("disk_budget", 4 << 20),
            max_inflight_bytes=overrides.pop("max_inflight_bytes", 2 << 20),
            max_inflight_requests=overrides.pop("max_inflight_requests", 4),
            max_merge_gap_blocks=overrides.pop("max_merge_gap_blocks", 1),
            max_retries=overrides.pop("max_retries", 3),
        )
        c = RemoteParquetClient(
            disk_dir=str(tmp_path / "qualitycache"), config=cfg
        )
        created.append(c)
        return c

    yield factory
    for c in created:
        c.close()


@pytest.fixture()
def make_quality_request(tmp_path):
    def _make(snapshot, *, contract=None, batch_rows=16,
              violations_name="violations.jsonl", report_name="report.json"):
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        return QualityAuditRequest(
            object=SnapshotManifest(
                snapshot["url"], snapshot["etag"], snapshot["length"]
            ),
            contract=QualityContract.from_dict(contract or quality_contract()),
            violations_path=str(out_dir / violations_name),
            report_path=str(out_dir / report_name),
            batch_rows=batch_rows,
        )
    return _make
