"""Temporal-enrich scale and resource tests (real Range server).

The skewed dataset has one hot key with >= 100k window revisions and >=
100k facts (overlapping windows, withdrawals, shuffled registration times),
plus normal keys.  The run must complete under a memory budget smaller than
the intermediate data size, with spill evidence, bounded RSS, column
projection, conditional requests, and results matching a brute-force oracle
on a deterministic sample.
"""

from __future__ import annotations

import hashlib
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import TemporalEnrichEngine, TemporalEnrichRequest

from temporal_conftest import (
    DIM_COLUMNS,
    FACT_COLUMNS,
)
from temporal_oracle import expected_for_fact

HOT_KEY = "entity-hot"
TIME_BASE = 1_000_000

# Schemas with many unused wide columns so column projection is measurable.
SCALE_FACT_SCHEMA = pa.schema([
    pa.field("entity_key", pa.string()),
    pa.field("event_at", pa.int64()),
    pa.field("known_at", pa.int64()),
    pa.field("fact_id", pa.int64()),
    pa.field("payload", pa.string()),
    pa.field("tag", pa.binary()),
    pa.field("amount", pa.decimal128(12, 3)),
    pa.field("moment", pa.timestamp("ms")),
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(8)],
])

SCALE_REVISION_SCHEMA = pa.schema([
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
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(4)],
])


def _hot_revisions(n_windows: int, per_window: int) -> list[dict]:
    rows = []
    # Random wide columns are incompressible and dominate the object size.
    rng = random.Random(99)
    for i in range(n_windows):
        wid = f"w-{i:04d}"
        for j in range(per_window):
            recorded_at = TIME_BASE + j * 1000 + (i % 500)
            vf = i * 100 + (j % 5) * 10
            vt = vf + 5000
            base = {f"wide_{c}": rng.random() * 1e6 for c in range(4)}
            if j > 0 and j % 17 == 0:
                rows.append({
                    "entity_key": HOT_KEY, "window_id": wid,
                    "recorded_at": recorded_at, "op": "withdraw",
                    "valid_from": None, "valid_to": None, "priority": None,
                    "service_area": None, "owner": None, "quota": None,
                    "active": None, "blob": None, "rate": None, "since": None,
                    **base,
                })
            else:
                rows.append({
                    "entity_key": HOT_KEY, "window_id": wid,
                    "recorded_at": recorded_at, "op": "upsert",
                    "valid_from": vf, "valid_to": vt,
                    "priority": (i + j) % 7 - 3,
                    "service_area": f"area-{(i * 31 + j) % 50}",
                    "owner": f"owner-{i % 20}",
                    "quota": i * 1000 + j,
                    "active": (i + j) % 2 == 0,
                    "blob": bytes([i % 256, j % 256]),
                    "rate": None,
                    "since": None,
                    **base,
                })
    return rows


def _hot_facts(n_facts: int) -> list[dict]:
    rows = []
    rng = random.Random(98)
    for k in range(n_facts):
        rows.append({
            "entity_key": HOT_KEY,
            "event_at": (k * 37) % 105_000,
            "known_at": TIME_BASE + (k * 997) % 100_000,
            "fact_id": k,
            "payload": f"payload-{k % 1000}",
            "tag": bytes([k % 256]) * 2,
            "amount": None,
            "moment": None,
            **{f"wide_{c}": rng.random() * 1e6 for c in range(8)},
        })
    return rows


def _normal_rows(n_keys: int, rng: random.Random):
    revisions, facts = [], []
    fid = 10_000_000
    for k in range(n_keys):
        key = f"entity-{k:06d}"
        for w in range(1 + k % 3):
            wid = f"nw-{k}-{w}"
            vf = rng.randint(0, 500)
            for r in range(1 + (k + w) % 3):
                revisions.append({
                    "entity_key": key, "window_id": wid,
                    "recorded_at": TIME_BASE + k * 10 + r, "op": "upsert",
                    "valid_from": vf + r, "valid_to": vf + 400 + r,
                    "priority": (k + w + r) % 5,
                    "service_area": f"area-{k % 7}", "owner": f"owner-{k % 5}",
                    "quota": k * 10 + w, "active": True, "blob": None,
                    "rate": None, "since": None,
                    **{f"wide_{c}": 0.0 for c in range(4)},
                })
        for _ in range(1 + k % 4):
            facts.append({
                "entity_key": key, "event_at": rng.randint(0, 1000),
                "known_at": TIME_BASE + k * 10 + rng.randint(0, 5),
                "fact_id": fid, "payload": f"np-{k}", "tag": None,
                "amount": None, "moment": None,
                **{f"wide_{c}": 0.0 for c in range(8)},
            })
            fid += 1
    return revisions, facts


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


def _publish(server, tmp_path, stem, rows, schema, n_shards, rows_per_group,
             rng):
    rows = list(rows)
    rng.shuffle(rows)
    manifests = []
    positioned = []
    per = (len(rows) + n_shards - 1) // n_shards
    for s in range(n_shards):
        part = rows[s * per:(s + 1) * per]
        if not part:
            continue
        shard_id = f"{stem}-{s}"
        data = _write_shard(str(tmp_path / f"{shard_id}.parquet"), part,
                            schema, rows_per_group)
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


@pytest.fixture()
def scale_dataset(tmp_path, temporal_server):
    """Publish the skewed dataset; return manifests + positioned rows."""
    rng = random.Random(2026)
    hot_revs = _hot_revisions(1000, 100)          # 100k revisions
    hot_facs = _hot_facts(100_000)                # 100k facts
    normal_revs, normal_facs = _normal_rows(2000, rng)
    revisions = hot_revs + normal_revs
    facts = hot_facs + normal_facs
    rev_manifests, positioned_revs = _publish(
        temporal_server, tmp_path, "revisions", revisions, SCALE_REVISION_SCHEMA,
        n_shards=3, rows_per_group=20_000, rng=rng,
    )
    fact_manifests, positioned_facts = _publish(
        temporal_server, tmp_path, "facts", facts, SCALE_FACT_SCHEMA,
        n_shards=3, rows_per_group=20_000, rng=rng,
    )
    temporal_server.clear_log()
    temporal_server.fault.reset()
    temporal_server.barrier.reset()
    yield {
        "fact_manifests": fact_manifests,
        "revision_manifests": rev_manifests,
        "facts": positioned_facts,
        "revisions": positioned_revs,
        "n_facts": len(facts),
        "n_revisions": len(revisions),
    }
    temporal_server.fault.reset()
    temporal_server.barrier.reset()


def test_scale_run_completes_with_spill_and_bounded_memory(
    temporal_client_factory, temporal_server, tmp_path, scale_dataset
):
    """A 100k+100k hot-key run completes under a budget < intermediate size."""
    client = temporal_client_factory()
    memory_limit = 4 << 20   # 4 MiB: far below the intermediate data size
    request = TemporalEnrichRequest.from_dict({
        "spec_version": 1,
        "fact_shards": scale_dataset["fact_manifests"],
        "revision_shards": scale_dataset["revision_manifests"],
        "fact_columns": list(FACT_COLUMNS),
        "dim_columns": list(DIM_COLUMNS),
        "output_path": str(tmp_path / "out" / "wide.parquet"),
        "memory_limit": memory_limit,
        "batch_rows": 4096,
    })
    report = TemporalEnrichEngine(client, request).run()

    # --- basic counts
    counts = report["counts"]
    assert counts["facts"] == scale_dataset["n_facts"]
    assert counts["revisions"] == scale_dataset["n_revisions"]
    assert counts["matched"] + counts["ambiguous"] + counts["unmatched"] \
        == counts["facts"]

    # --- spill evidence: the intermediate data exceeds the budget, so
    # segments and key buffers must have spilled.
    spill = report["resources"]["spill"]
    assert spill["spilled_bytes"] > 0
    assert spill["key_buffers_spilled"] >= 1, "hot key must spill"
    assert spill["temp_bytes_peak"] > memory_limit, (
        f"temp space {spill['temp_bytes_peak']} should exceed budget "
        f"{memory_limit}"
    )
    # --- memory evidence: peak RSS is bounded (not the full dataset).
    peak = report["resources"]["peak"]
    assert peak["max_rss_bytes"] > 0
    # The resident record bytes never exceeded the budget.
    assert peak["resident_record_bytes_peak"] <= memory_limit
    # --- matching workload: stabbing queries == facts, episodes == upserts.
    matching = report["resources"]["matching"]
    assert matching["stabbing_queries"] == counts["facts"]
    assert matching["episodes"] > 0
    assert matching["keys_total"] > 0

    # --- network: wire bytes are a fraction of the full objects (projection).
    net = report["resources"]["network_cache"]
    total_object_bytes = (
        sum(m["length"] for m in scale_dataset["fact_manifests"])
        + sum(m["length"] for m in scale_dataset["revision_manifests"])
    )
    assert net["wire_bytes"] < total_object_bytes // 2, (
        f"wire bytes {net['wire_bytes']} should be much less than "
        f"object bytes {total_object_bytes}"
    )
    # Every data GET carries the pinned If-Match conditional header.
    gets = temporal_server.range_gets()
    assert gets
    etags = {m["etag"] for m in scale_dataset["fact_manifests"]} | {
        m["etag"] for m in scale_dataset["revision_manifests"]
    }
    for entry in gets:
        assert entry.status == 206
        assert entry.if_match in etags
        assert entry.range_header.startswith("bytes=")

    # --- output is a single complete Parquet with the right row count.
    out_path = request.output_path
    assert os.path.getsize(out_path) > 0
    pf = pq.ParquetFile(out_path)
    assert pf.metadata.num_rows == counts["facts"]
    assert pf.schema_arrow.field("_match_status").type == pa.string()
    assert pf.schema_arrow.field("_dim").type.num_fields == len(DIM_COLUMNS)

    # --- sample brute-force verification against the oracle.
    table = pq.read_table(out_path)
    rows = table.to_pylist()
    positioned_facts = scale_dataset["facts"]
    positioned_revs = scale_dataset["revisions"]
    revs_by_key = {}
    for entry in positioned_revs:
        revs_by_key.setdefault(entry[0]["entity_key"], []).append(entry)
    sample_indices = sorted({
        (i * 9973) % len(rows) for i in range(48)
    } | {0, len(rows) // 2, len(rows) - 1})
    for out_row in sample_indices:
        fact, shard_id, rg, row = positioned_facts[out_row]
        exp = expected_for_fact(
            revs_by_key.get(fact["entity_key"], []),
            fact["event_at"], fact["known_at"], DIM_COLUMNS,
        )
        got = rows[out_row]
        assert got["_match_status"] == exp["status"], (
            f"row {out_row}: {got['_match_status']} != {exp['status']}"
        )
        assert got["_top_window_count"] == exp["count"]
        assert got["_fact_source"] == {
            "shard_id": shard_id, "row_group": rg, "row": row,
        }
        if exp["dim"] is None:
            assert got["_dim"] is None
        else:
            for col in DIM_COLUMNS:
                assert got["_dim"][col] == exp["dim"][col]
        assert got["_dim_sources"] == exp["dim_sources"]

    # --- temp files are cleaned up.
    spill_root = tmp_path / "out" / ".pqr-temporal-tmp"
    assert not spill_root.exists()
    assert not list((tmp_path / "out").glob("*.tmp"))


def test_scale_run_deterministic(
    temporal_client_factory, temporal_server, tmp_path, scale_dataset
):
    """Two runs of the same dataset produce byte-identical output."""
    client = temporal_client_factory()
    req1 = TemporalEnrichRequest.from_dict({
        "spec_version": 1,
        "fact_shards": scale_dataset["fact_manifests"],
        "revision_shards": scale_dataset["revision_manifests"],
        "fact_columns": list(FACT_COLUMNS),
        "dim_columns": list(DIM_COLUMNS),
        "output_path": str(tmp_path / "out1" / "wide.parquet"),
        "memory_limit": 4 << 20,
        "batch_rows": 4096,
    })
    TemporalEnrichEngine(client, req1).run()
    req2 = TemporalEnrichRequest.from_dict({
        "spec_version": 1,
        "fact_shards": scale_dataset["fact_manifests"],
        "revision_shards": scale_dataset["revision_manifests"],
        "fact_columns": list(FACT_COLUMNS),
        "dim_columns": list(DIM_COLUMNS),
        "output_path": str(tmp_path / "out2" / "wide.parquet"),
        "memory_limit": 4 << 20,
        "batch_rows": 4096,
    })
    TemporalEnrichEngine(client, req2).run()
    assert open(req1.output_path, "rb").read() == open(
        req2.output_path, "rb").read()


def test_scale_run_small_budget_still_completes(
    temporal_client_factory, temporal_server, tmp_path, scale_dataset
):
    """A very small budget still completes (spill dominates)."""
    client = temporal_client_factory()
    request = TemporalEnrichRequest.from_dict({
        "spec_version": 1,
        "fact_shards": scale_dataset["fact_manifests"],
        "revision_shards": scale_dataset["revision_manifests"],
        "fact_columns": list(FACT_COLUMNS),
        "dim_columns": list(DIM_COLUMNS),
        "output_path": str(tmp_path / "out" / "wide.parquet"),
        "memory_limit": 1 << 20,   # 1 MiB
        "batch_rows": 2048,
    })
    report = TemporalEnrichEngine(client, request).run()
    assert report["counts"]["facts"] == scale_dataset["n_facts"]
    spill = report["resources"]["spill"]
    assert spill["spilled_bytes"] > 0
    assert spill["key_buffers_spilled"] >= 1
