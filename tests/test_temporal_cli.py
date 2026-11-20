"""Temporal-enrich CLI tests (real Range server, real JSON manifest).

The CLI must produce the same wide table as the Python API, print the run
statistics as JSON, and return distinct exit codes for the failure classes.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pyarrow.parquet as pq
import pytest

from temporal_conftest import (
    DIM_COLUMNS,
    FACT_COLUMNS,
    FACT_SCHEMA,
    REVISION_SCHEMA,
    fact_row,
    publish_shards,
    rev_upsert,
)


def _run_cli(args, cwd=None):
    cmd = [sys.executable, "bin/pqr.py"] + args
    return subprocess.run(cmd, capture_output=True, text=True, cwd=cwd or "/workspace")


def _write_request(tmp_path, fact_manifests, rev_manifests, output_path,
                   memory_limit=4 << 20, **overrides):
    req = {
        "spec_version": 1,
        "fact_shards": fact_manifests,
        "revision_shards": rev_manifests,
        "fact_columns": list(FACT_COLUMNS),
        "dim_columns": list(DIM_COLUMNS),
        "output_path": str(output_path),
        "memory_limit": memory_limit,
        "batch_rows": 32,
    }
    req.update(overrides)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(req), encoding="utf-8")
    return str(path)


def test_cli_run_produces_wide_table_and_stats(
    temporal_server, tmp_path
):
    facts = [fact_row(i, f"k{i % 3}", 10 + i, 100 + i) for i in range(12)]
    revisions = [
        rev_upsert("k0", "w0", 5, 0, 100, 1, area="A"),
        rev_upsert("k1", "w1", 5, 0, 100, 2, area="B"),
        rev_upsert("k2", "w2", 5, 0, 100, 3, area="C"),
    ]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 2, 6
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    out_path = tmp_path / "out" / "wide.parquet"
    request_path = _write_request(tmp_path, fact_manifests, rev_manifests,
                                  out_path)
    result = _run_cli(["temporal-enrich", request_path])
    assert result.returncode == 0, result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "completed"
    assert report["counts"]["facts"] == 12
    assert report["output"]["path"] == str(out_path)
    assert report["resources"]["matching"]["stabbing_queries"] == 12
    assert report["resources"]["network_cache"]["wire_bytes"] > 0
    # The wide table is a valid Parquet with the right row count.
    table = pq.read_table(out_path)
    assert table.num_rows == 12
    assert "_match_status" in table.column_names
    assert "_dim" in table.column_names
    # The stderr summary mentions the published path.
    assert "temporal enrich published" in result.stderr


def test_cli_manifest_error_exit_code(temporal_server, tmp_path):
    out_path = tmp_path / "out" / "wide.parquet"
    req_path = _write_request(tmp_path, [], [], out_path,
                              fact_columns=["a", "a"])
    result = _run_cli(["temporal-enrich", req_path])
    assert result.returncode == 3
    assert "duplicates" in result.stderr


def test_cli_schema_error_exit_code(temporal_server, tmp_path):
    import pyarrow as pa
    bad_schema = pa.schema([
        pa.field("entity_key", pa.string()),
        pa.field("event_at", pa.int64()),
        # known_at missing
        pa.field("fact_id", pa.int64()),
    ])
    facts = [{"entity_key": "k", "event_at": 1, "fact_id": 1}]
    fact_path = tmp_path / "facts-0.parquet"
    pq.write_table(pa.table({c: [r[c]] for r in facts for c in [c]}
                            if False else {
                                "entity_key": ["k"], "event_at": [1],
                                "fact_id": [1],
                            }, schema=bad_schema), str(fact_path))
    import hashlib
    data = open(str(fact_path), "rb").read()
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    temporal_server.put_object("/facts-0.parquet", data, etag)
    fact_manifests = [{
        "shard_id": "facts-0", "url": temporal_server.url_for("/facts-0.parquet"),
        "etag": etag, "length": len(data),
    }]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    out_path = tmp_path / "out" / "wide.parquet"
    req_path = _write_request(tmp_path, fact_manifests, rev_manifests,
                              out_path)
    result = _run_cli(["temporal-enrich", req_path])
    assert result.returncode == 4
    assert "known_at" in result.stderr


def test_cli_data_error_exit_code(temporal_server, tmp_path):
    bad = fact_row(0, "k", None, 10)
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", [bad], FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions",
        [rev_upsert("k", "w", 1, 0, 100, 1)], REVISION_SCHEMA, 1, 10,
    )
    out_path = tmp_path / "out" / "wide.parquet"
    req_path = _write_request(tmp_path, fact_manifests, rev_manifests,
                              out_path)
    result = _run_cli(["temporal-enrich", req_path])
    assert result.returncode == 6
    assert "event_at" in result.stderr


def test_cli_publish_error_exit_code(temporal_server, tmp_path):
    facts = [fact_row(0, "k", 1, 10)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    # Output path is a directory: the atomic replace fails.
    out_path = tmp_path / "out" / "wide.parquet"
    out_path.mkdir(parents=True)
    req_path = _write_request(tmp_path, fact_manifests, rev_manifests,
                              out_path)
    result = _run_cli(["temporal-enrich", req_path])
    assert result.returncode == 5
    assert "temporal-enrich" in result.stderr.lower() or "publish" in result.stderr.lower()


def test_cli_version_mismatch_exit_code(temporal_server, tmp_path):
    facts = [fact_row(0, "k", 1, 10)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    # Swap the revision shard to a different version.
    new_rev = [rev_upsert("k", "w2", 2, 0, 200, 2)]
    path = tmp_path / "revisions-0.parquet"
    import pyarrow as pa
    import pyarrow.parquet as pq
    table = pa.table({c: [r[c] for r in new_rev] for c in REVISION_SCHEMA.names},
                     schema=REVISION_SCHEMA)
    pq.write_table(table, str(path))
    new_data = open(str(path), "rb").read()
    temporal_server.swap_version("/revisions-0.parquet", new_data,
                                 '"new-etag"', retain_history=False)
    out_path = tmp_path / "out" / "wide.parquet"
    req_path = _write_request(tmp_path, fact_manifests, rev_manifests,
                              out_path)
    result = _run_cli(["temporal-enrich", req_path])
    assert result.returncode == 2
    assert "412" in result.stderr or "version" in result.stderr.lower()


def test_cli_help_lists_temporal_enrich():
    result = _run_cli(["--help"])
    assert result.returncode == 0
    assert "temporal-enrich" in result.stdout
