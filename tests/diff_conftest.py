"""Shared fixtures and the *independent* diff oracle.

The oracle computes expected diffs by reading the two generated Parquet files
directly with local PyArrow (full, separate code path -- it never imports or
calls the product's diff engine).  Tests compare the product's JSONL facts
line-for-line against this expectation.
"""

from __future__ import annotations

import hashlib
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import ClientConfig, DiffRequest, RemoteParquetClient, SnapshotManifest
from parquet_range.test_server import TestHTTPServer

# A deliberately rich schema exercising every required value shape.
DIFF_SCHEMA = pa.schema([
    pa.field("kid", pa.int64()),
    pa.field("code", pa.string()),
    pa.field("active", pa.bool_()),
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
    pa.field("filler", pa.float64()),
])

KEY_COLUMN = "kid"
COMPARE_COLUMNS = ("code", "active", "tags", "meta", "cat", "blob")


def _row(kid, code, active, tags, meta, cat, blob, filler):
    return {
        "kid": kid, "code": code, "active": active, "tags": tags,
        "meta": meta, "cat": cat, "blob": blob, "filler": filler,
    }


def _make_rows(spec):
    """Build row dicts from compact (kid, code, tag-pick) specifications."""
    rows = []
    for kid, code, pick in spec:
        if pick == 0:
            tags, meta, cat, blob = None, None, None, None
        elif pick == 1:
            tags, meta, cat, blob = [], {"x": None, "y": "", "flags": []}, "alpha", b""
        elif pick == 2:
            tags = ["a", "b"]
            meta = {"x": kid, "y": f"y{kid}", "flags": [1, 2, 3]}
            cat = "beta"
            blob = bytes([kid & 0xFF]) * 3
        else:
            tags = [f"t{kid}"]
            meta = {"x": -kid, "y": None, "flags": None}
            cat = "gamma"
            blob = b"\x00\xff" * 2
        rows.append(_row(kid, code, pick % 2 == 0, tags, meta, cat, blob,
                         float(kid)))
    return rows


def write_diff_parquet(path, rows, *, groups, seed=0, duplicate_key=None,
                       null_key=False):
    """Write rows spread over ``groups`` row groups, optionally shuffled."""
    rows = [dict(r) for r in rows]
    if null_key and rows:
        rows[len(rows) // 2] = dict(rows[len(rows) // 2], kid=None)
    if duplicate_key is not None and rows:
        rows.append(dict(rows[0], kid=duplicate_key))
        rows[len(rows) // 3] = dict(rows[len(rows) // 3], kid=duplicate_key)
    with pq.ParquetWriter(path, DIFF_SCHEMA, compression="snappy",
                          use_dictionary=True) as w:
        per = max(1, len(rows) // groups)
        for g in range(groups):
            part = rows[g * per:] if g == groups - 1 else rows[g * per:(g + 1) * per]
            if part:
                t = pa.table({k: [r[k] for r in part] for k in DIFF_SCHEMA.names},
                             schema=DIFF_SCHEMA)
                w.write_table(t, row_group_size=len(part))
    return open(path, "rb").read()


def baseline_spec():
    # ids 1..12 with varied shapes
    return [
        (1, "a", 2), (2, "b", 0), (3, "c", 1), (4, "d", 3),
        (5, "e", 2), (6, "f", 0), (7, "g", 1), (8, "h", 3),
        (9, "i", 2), (10, "j", 0), (11, "k", 1), (12, "l", 3),
    ]


def target_spec():
    # insert 13,14 ; delete 11,12 ; update 2,5,9 ; keep the rest; REORDERED.
    kept = [(1, "a", 2), (3, "c", 1), (4, "d", 3), (6, "f", 0),
            (7, "g", 1), (8, "h", 3), (10, "j", 0)]
    updated = [(2, "B*", 0), (5, "E*", 1), (9, "I*", 3)]
    inserted = [(13, "m", 2), (14, "n", 0)]
    shuffled = updated[1:] + kept[3:] + inserted[:1] + updated[:1] + kept[:3] + inserted[1:]
    return shuffled


@pytest.fixture(scope="session")
def diff_server():
    srv = TestHTTPServer()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def _etag(data: bytes) -> str:
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


@pytest.fixture()
def diff_snapshots(tmp_path, diff_server):
    """Publish a baseline and a target snapshot; return manifests + raw bytes."""
    left_path = str(tmp_path / "base.parquet")
    right_path = str(tmp_path / "target.parquet")
    left_rows = _make_rows(baseline_spec())
    right_rows = _make_rows(target_spec())
    left_data = write_diff_parquet(left_path, left_rows, groups=4)
    right_data = write_diff_parquet(right_path, right_rows, groups=3)
    le, re_ = _etag(left_data), _etag(right_data)
    diff_server.put_object("/base.parquet", left_data, le)
    diff_server.put_object("/target.parquet", right_data, re_)
    diff_server.clear_log()
    diff_server.fault.reset()
    diff_server.barrier.reset()
    yield {
        "left": (diff_server.url_for("/base.parquet"), le, len(left_data), left_data),
        "right": (diff_server.url_for("/target.parquet"), re_, len(right_data), right_data),
        "left_rows": left_rows,
        "right_rows": right_rows,
    }
    diff_server.fault.reset()
    diff_server.barrier.reset()


@pytest.fixture()
def diff_client_factory(tmp_path):
    created: list[RemoteParquetClient] = []

    def factory(**overrides):
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
            disk_dir=str(tmp_path / "diffcache"), config=cfg
        )
        created.append(c)
        return c

    yield factory
    for c in created:
        c.close()


@pytest.fixture()
def make_request(tmp_path):
    def _make(snapshots, *, memory_limit=4 << 20,
              key_column=KEY_COLUMN, compare_columns=COMPARE_COLUMNS,
              diff_name="facts.jsonl", report_name="report.json"):
        lurl, le, llen, _ = snapshots["left"]
        rurl, re_, rlen, _ = snapshots["right"]
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        return DiffRequest(
            left=SnapshotManifest(lurl, le, llen),
            right=SnapshotManifest(rurl, re_, rlen),
            key_column=key_column,
            compare_columns=tuple(compare_columns),
            diff_path=str(out_dir / diff_name),
            report_path=str(out_dir / report_name),
            memory_limit=memory_limit,
        )
    return _make
