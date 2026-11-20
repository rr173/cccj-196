"""Shared fixtures for the temporal-enrich feature tests.

Everything runs against the real local Range HTTP server; expected results
come from the independent oracle in ``temporal_oracle.py`` (which never
touches the product's matching engine).
"""

from __future__ import annotations

import datetime
import hashlib
import threading
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import ClientConfig, RemoteParquetClient, TemporalEnrichRequest
from parquet_range.test_server import TestHTTPServer

FACT_SCHEMA = pa.schema([
    pa.field("entity_key", pa.string()),
    pa.field("event_at", pa.int64()),
    pa.field("known_at", pa.int64()),
    pa.field("fact_id", pa.int64()),
    pa.field("payload", pa.string()),
    pa.field("tag", pa.binary()),
    pa.field("amount", pa.decimal128(12, 3)),
    pa.field("moment", pa.timestamp("ms")),
    pa.field("unused_f", pa.float64()),   # never selected: proves projection
])

REVISION_SCHEMA = pa.schema([
    pa.field("entity_key", pa.string()),
    pa.field("window_id", pa.string()),
    pa.field("recorded_at", pa.int64()),
    pa.field("op", pa.string()),
    pa.field("valid_from", pa.int64()),
    pa.field("valid_to", pa.int64()),
    pa.field("priority", pa.int64()),
    pa.field("service_area", pa.string()),
    pa.field("owner", pa.string()),
    pa.field("quota", pa.int64()),
    pa.field("active", pa.bool_()),
    pa.field("blob", pa.binary()),
    pa.field("rate", pa.decimal128(10, 2)),
    pa.field("since", pa.timestamp("us", tz="UTC")),
    pa.field("unused_r", pa.float64()),   # never selected: proves projection
])

FACT_COLUMNS = ["fact_id", "payload", "tag", "amount", "moment"]
DIM_COLUMNS = ["service_area", "owner", "quota", "active", "blob", "rate", "since"]

_BASE_MS = datetime.datetime(2022, 3, 1)
_BASE_US = datetime.datetime(2022, 3, 1, tzinfo=datetime.timezone.utc)


def fact_row(fid, key, event_at, known_at, payload=None, tag=None,
             amount=None, moment=None):
    return {
        "entity_key": key,
        "event_at": event_at,
        "known_at": known_at,
        "fact_id": fid,
        "payload": payload if payload is not None else f"p-{fid}",
        "tag": tag if tag is not None else bytes([fid % 256]) * 3,
        "amount": amount if amount is not None else Decimal(f"{fid}.001"),
        "moment": moment if moment is not None
        else _BASE_MS + datetime.timedelta(milliseconds=fid),
        "unused_f": float(fid) * 1.5,
    }


def rev_upsert(key, wid, recorded_at, vf, vt, prio, *, area=None, owner=None,
               quota=None, active=None, blob=None, rate=None, since=None):
    return {
        "entity_key": key,
        "window_id": wid,
        "recorded_at": recorded_at,
        "op": "upsert",
        "valid_from": vf,
        "valid_to": vt,
        "priority": prio,
        "service_area": area if area is not None else f"area-{wid}",
        "owner": owner if owner is not None else f"owner-{wid}",
        "quota": quota if quota is not None else len(wid) * 100 + recorded_at % 97,
        "active": active if active is not None else (recorded_at % 2 == 0),
        "blob": blob if blob is not None else bytes([recorded_at % 256, 0xFF]),
        "rate": rate if rate is not None
        else Decimal(f"{prio if prio is not None else 0}.25"),
        "since": since if since is not None
        else _BASE_US + datetime.timedelta(microseconds=recorded_at),
        "unused_r": float(recorded_at) / 7.0,
    }


def rev_withdraw(key, wid, recorded_at):
    return {
        "entity_key": key,
        "window_id": wid,
        "recorded_at": recorded_at,
        "op": "withdraw",
        "valid_from": None,
        "valid_to": None,
        "priority": None,
        "service_area": None,
        "owner": None,
        "quota": None,
        "active": None,
        "blob": None,
        "rate": None,
        "since": None,
        "unused_r": float(recorded_at) / 7.0,
    }


def _write_shard(path, rows, schema, rows_per_group):
    with pq.ParquetWriter(path, schema, compression="snappy",
                          use_dictionary=False) as writer:
        for start in range(0, len(rows), rows_per_group):
            part = rows[start:start + rows_per_group]
            table = pa.table(
                {name: [r[name] for r in part] for name in schema.names},
                schema=schema,
            )
            writer.write_table(table, row_group_size=len(part))
    return open(path, "rb").read()


def publish_shards(server, tmp_path, stem, rows, schema, n_shards,
                   rows_per_group):
    """Write rows (in order) to n_shards parquet files and publish them.

    Returns ``(manifests, positioned_rows)`` where positioned_rows is the
    flat list of ``(row_dict, shard_id, row_group, row)`` in physical order.
    """
    manifests = []
    positioned = []
    per = (len(rows) + n_shards - 1) // n_shards if rows else 0
    for s in range(n_shards):
        part = rows[s * per:(s + 1) * per] if rows else []
        if not part:
            continue
        shard_id = f"{stem}-{s}"
        path = str(tmp_path / f"{shard_id}.parquet")
        data = _write_shard(path, part, schema, rows_per_group)
        etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
        server.put_object(f"/{shard_id}.parquet", data, etag)
        manifests.append({
            "shard_id": shard_id,
            "url": server.url_for(f"/{shard_id}.parquet"),
            "etag": etag,
            "length": len(data),
        })
        for i, row in enumerate(part):
            positioned.append((row, shard_id, i // rows_per_group, i))
    return manifests, positioned


@pytest.fixture(scope="session")
def temporal_server():
    srv = TestHTTPServer()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


@pytest.fixture()
def temporal_client_factory(tmp_path):
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
        c = RemoteParquetClient(disk_dir=str(tmp_path / "temporalcache"),
                                config=cfg)
        created.append(c)
        return c

    yield factory
    for c in created:
        c.close()


@pytest.fixture()
def make_temporal_request(tmp_path):
    def _make(fact_manifests, revision_manifests, *, output_name="wide.parquet",
              memory_limit=4 << 20, fact_columns=FACT_COLUMNS,
              dim_columns=DIM_COLUMNS, spill_dir=None, batch_rows=64):
        out_dir = tmp_path / "out"
        out_dir.mkdir(exist_ok=True)
        return TemporalEnrichRequest.from_dict({
            "spec_version": 1,
            "fact_shards": fact_manifests,
            "revision_shards": revision_manifests,
            "fact_columns": list(fact_columns),
            "dim_columns": list(dim_columns),
            "output_path": str(out_dir / output_name),
            "memory_limit": memory_limit,
            "spill_dir": spill_dir,
            "batch_rows": batch_rows,
        })
    return _make


# ------------------------------------------------------------------ scenario
def timeline_revisions(include_tie_window: bool):
    """The required bitemporal scenario for key 'A' (and 'B' with the tie).

    Key A window W1: [0,100) at registration 10, moves to [50,100) at 20,
    withdrawn at 30, reopened as [0,100) at 40.  Low-priority W2 always
    covers event 25.  Key B adds high-priority W3 covering [25,26) from
    registration 15 until a withdrawal at 35, so W3 ties W1 only at one of
    the probed knowledge times.
    """
    rows = []
    for key in ("A", "B"):
        rows += [
            rev_upsert(key, "W2", 5, 0, 1000, 1, area="L"),
            rev_upsert(key, "W1", 10, 0, 100, 5, area="A1"),
            rev_upsert(key, "W1", 20, 50, 100, 5, area="A2"),
            rev_withdraw(key, "W1", 30),
            rev_upsert(key, "W1", 40, 0, 100, 5, area="A3"),
        ]
    if include_tie_window:
        rows += [
            rev_upsert("B", "W3", 15, 25, 26, 5, area="H"),
            rev_withdraw("B", "W3", 35),
        ]
    return rows


def timeline_facts():
    """Facts probing event 25 at known_at 19/20/30/40 plus boundary probes."""
    rows = []
    fid = 0
    for key in ("A", "B"):
        for known_at in (19, 20, 30, 40):
            rows.append(fact_row(fid, key, 25, known_at))
            fid += 1
        # Left-closed / right-open boundary probes.
        rows.append(fact_row(fid, key, 26, 19))   # outside W3 [25,26)
        fid += 1
        rows.append(fact_row(fid, key, 50, 20))   # left edge of W1 [50,100)
        fid += 1
        rows.append(fact_row(fid, key, 100, 40))  # right edge of W1 [0,100)
        fid += 1
    return rows


@pytest.fixture()
def timeline_dataset(tmp_path, temporal_server):
    """Publish the timeline scenario; return manifests + positioned rows."""
    facts = timeline_facts()
    revisions = timeline_revisions(include_tie_window=True)
    fact_manifests, positioned_facts = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA,
        n_shards=2, rows_per_group=5,
    )
    rev_manifests, positioned_revs = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        n_shards=2, rows_per_group=4,
    )
    temporal_server.clear_log()
    temporal_server.fault.reset()
    temporal_server.barrier.reset()
    yield {
        "fact_manifests": fact_manifests,
        "revision_manifests": rev_manifests,
        "facts": positioned_facts,
        "revisions": positioned_revs,
    }
    temporal_server.fault.reset()
    temporal_server.barrier.reset()
