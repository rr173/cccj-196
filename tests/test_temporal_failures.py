"""Temporal-enrich failure-mode tests (real Range server).

Covers manifest/schema errors (rejected before any row data is read), bad
rows with exact source coordinates, duplicate revisions (cross-shard,
identical content, unseen by any fact), version-swap failures, publish
failures, and cancellation — all with the guarantee that a previously
published wide table is never overwritten by a failed run.
"""

from __future__ import annotations

import json
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import (
    TemporalEnrichEngine,
    TemporalEnrichRequest,
)
from parquet_range.exceptions import (
    TemporalDataError,
    TemporalPublishError,
    TemporalRequestError,
    TemporalSchemaError,
    VersionMismatchError,
)

from temporal_conftest import (
    DIM_COLUMNS,
    FACT_COLUMNS,
    FACT_SCHEMA,
    REVISION_SCHEMA,
    fact_row,
    publish_shards,
    rev_upsert,
    rev_withdraw,
)


def _write_parquet_file(tmp_path, name, rows, schema, rows_per_group=10):
    path = str(tmp_path / name)
    with pq.ParquetWriter(path, schema, compression="snappy",
                          use_dictionary=False) as writer:
        for start in range(0, len(rows), rows_per_group):
            part = rows[start:start + rows_per_group]
            table = pa.table(
                {col: [r[col] for r in part] for col in schema.names},
                schema=schema,
            )
            writer.write_table(table, row_group_size=len(part))
    return path


def _manifest(server, tmp_path, name, rows, schema, rows_per_group=10):
    path = _write_parquet_file(tmp_path, name, rows, schema, rows_per_group)
    data = open(path, "rb").read()
    import hashlib
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    server.put_object(f"/{name}", data, etag)
    return {
        "shard_id": name.replace(".parquet", ""),
        "url": server.url_for(f"/{name}"),
        "etag": etag,
        "length": len(data),
    }


def _request_dict(fact_shards, revision_shards, out_path, **overrides):
    req = {
        "spec_version": 1,
        "fact_shards": fact_shards,
        "revision_shards": revision_shards,
        "fact_columns": list(FACT_COLUMNS),
        "dim_columns": list(DIM_COLUMNS),
        "output_path": str(out_path),
        "memory_limit": 4 << 20,
        "batch_rows": 32,
    }
    req.update(overrides)
    return req


def _run_request(request_dict, client, tmp_path):
    request = TemporalEnrichRequest.from_dict(request_dict)
    return TemporalEnrichEngine(client, request).run()


# ------------------------------------------------------------- manifest
def test_manifest_unknown_key_rejected(tmp_path):
    req = _request_dict([], [], tmp_path / "out" / "wide.parquet")
    req["surprise"] = 1
    with pytest.raises(TemporalRequestError, match="unknown keys"):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_duplicate_shard_id_rejected(tmp_path):
    shard = {"shard_id": "x", "url": "http://h/a.parquet",
             "etag": '"abc"', "length": 100}
    req = _request_dict([shard, dict(shard, url="http://h/b.parquet")], [],
                        tmp_path / "out" / "wide.parquet")
    with pytest.raises(TemporalRequestError, match="duplicate shard_id"):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_duplicate_columns_rejected(tmp_path):
    req = _request_dict([], [], tmp_path / "out" / "wide.parquet",
                        fact_columns=["a", "a"])
    with pytest.raises(TemporalRequestError, match="duplicates"):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_reserved_column_rejected(tmp_path):
    req = _request_dict([], [], tmp_path / "out" / "wide.parquet",
                        dim_columns=["_match_status"])
    with pytest.raises(TemporalRequestError, match="reserved"):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_empty_dim_columns_rejected(tmp_path):
    req = _request_dict([], [], tmp_path / "out" / "wide.parquet",
                        dim_columns=[])
    with pytest.raises(TemporalRequestError, match="dim_columns"):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_bad_spec_version_rejected(tmp_path):
    req = _request_dict([], [], tmp_path / "out" / "wide.parquet",
                        spec_version=99)
    with pytest.raises(TemporalRequestError, match="spec_version"):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_weak_etag_rejected(tmp_path):
    shard = {"shard_id": "x", "url": "http://h/a.parquet",
             "etag": 'W/"weak"', "length": 100}
    req = _request_dict([shard], [], tmp_path / "out" / "wide.parquet")
    with pytest.raises(TemporalRequestError):
        TemporalEnrichRequest.from_dict(req)


def test_manifest_missing_required_key_rejected(tmp_path):
    req = _request_dict([], [], tmp_path / "out" / "wide.parquet")
    del req["output_path"]
    with pytest.raises(TemporalRequestError, match="output_path"):
        TemporalEnrichRequest.from_dict(req)


def test_request_file_rejects_duplicate_json_keys(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(
        '{"spec_version": 1, "fact_shards": [], "fact_shards": []}',
        encoding="utf-8",
    )
    with pytest.raises(TemporalRequestError, match="duplicate key"):
        TemporalEnrichRequest.from_json_file(str(path))


# --------------------------------------------------------------- schema
def _schema_error_request(temporal_server, tmp_path, fact_schema, rev_schema):
    facts = [fact_row(0, "k", 1, 10)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                facts, fact_schema)]
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               revisions, rev_schema)]
    return fact_manifests, rev_manifests


def test_schema_missing_required_column_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad_fact_schema = pa.schema([
        pa.field("entity_key", pa.string()),
        pa.field("event_at", pa.int64()),
        # known_at missing
        pa.field("fact_id", pa.int64()),
    ])
    fact_manifests, rev_manifests = _schema_error_request(
        temporal_server, tmp_path, bad_fact_schema, REVISION_SCHEMA
    )
    client = temporal_client_factory()
    with pytest.raises(TemporalSchemaError, match="known_at"):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )


def test_schema_wrong_time_type_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad_fact_schema = pa.schema([
        pa.field("entity_key", pa.string()),
        pa.field("event_at", pa.int32()),   # must be int64
        pa.field("known_at", pa.int64()),
        pa.field("fact_id", pa.int64()),
    ])
    fact_manifests, rev_manifests = _schema_error_request(
        temporal_server, tmp_path, bad_fact_schema, REVISION_SCHEMA
    )
    client = temporal_client_factory()
    with pytest.raises(TemporalSchemaError, match="event_at"):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )


def test_schema_type_mismatch_across_shards_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    other_schema = pa.schema([
        pa.field("entity_key", pa.string()),
        pa.field("event_at", pa.int64()),
        pa.field("known_at", pa.int64()),
        pa.field("fact_id", pa.int32()),   # int32 here, int64 elsewhere
        pa.field("payload", pa.string()),
        pa.field("tag", pa.binary()),
        pa.field("amount", pa.decimal128(12, 3)),
        pa.field("moment", pa.timestamp("ms")),
        pa.field("unused_f", pa.float64()),
    ])
    facts0 = [fact_row(0, "k", 1, 10)]
    facts1 = [fact_row(1, "k", 2, 10)]
    fact_manifests = [
        _manifest(temporal_server, tmp_path, "facts-0.parquet", facts0,
                  FACT_SCHEMA),
        _manifest(temporal_server, tmp_path, "facts-1.parquet", facts1,
                  other_schema),
    ]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               revisions, REVISION_SCHEMA)]
    client = temporal_client_factory()
    with pytest.raises(TemporalSchemaError, match="fact_id"):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )


def test_schema_unsupported_business_type_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad_rev_schema = pa.schema([
        pa.field("entity_key", pa.string()),
        pa.field("window_id", pa.string()),
        pa.field("recorded_at", pa.int64()),
        pa.field("op", pa.string()),
        pa.field("valid_from", pa.int64()),
        pa.field("valid_to", pa.int64()),
        pa.field("priority", pa.int64()),
        pa.field("service_area", pa.list_(pa.string())),  # nested: unsupported
    ])
    revisions = [{
        "entity_key": "k", "window_id": "w", "recorded_at": 1,
        "op": "upsert", "valid_from": 0, "valid_to": 100, "priority": 1,
        "service_area": ["a", "b"],
    }]
    facts = [fact_row(0, "k", 1, 10)]
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                facts, FACT_SCHEMA)]
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               revisions, bad_rev_schema)]
    client = temporal_client_factory()
    with pytest.raises(TemporalSchemaError, match="service_area"):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet",
                          dim_columns=["service_area"]),
            client, tmp_path,
        )


def test_schema_error_happens_before_row_data(
    temporal_client_factory, temporal_server, tmp_path
):
    """Schema errors are raised before any row-group data is fetched."""
    bad_fact_schema = pa.schema([
        pa.field("entity_key", pa.string()),
        pa.field("event_at", pa.int64()),
        # known_at missing
        pa.field("fact_id", pa.int64()),
    ])
    fact_manifests, rev_manifests = _schema_error_request(
        temporal_server, tmp_path, bad_fact_schema, REVISION_SCHEMA
    )
    temporal_server.clear_log()
    client = temporal_client_factory()
    with pytest.raises(TemporalSchemaError):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )
    # Only footer/metadata ranges were fetched: every GET must be a small
    # range (footer or 1-byte probe), never a row-group column chunk.
    gets = temporal_server.range_gets()
    assert gets, "expected footer/probe reads"
    for entry in gets:
        assert entry.status == 206
        start, end = entry.range_header.replace("bytes=", "").split("-")
        assert int(end) - int(start) < 32 * 1024, (
            f"row-data range fetched: {entry.range_header}"
        )


# ---------------------------------------------------------------- bad rows
def _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                   fact_rows, rev_rows, fact_schema=FACT_SCHEMA,
                   rev_schema=REVISION_SCHEMA, **req_overrides):
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                fact_rows, fact_schema)]
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               rev_rows, rev_schema)]
    client = temporal_client_factory()
    return _run_request(
        _request_dict(fact_manifests, rev_manifests,
                      tmp_path / "out" / "wide.parquet", **req_overrides),
        client, tmp_path,
    )


def test_null_entity_key_in_fact_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = fact_row(0, "k", 1, 10)
    bad["entity_key"] = None
    with pytest.raises(TemporalDataError, match="entity_key"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [bad], [rev_upsert("k", "w", 1, 0, 100, 1)])


def test_empty_entity_key_in_fact_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = fact_row(0, "", 1, 10)
    with pytest.raises(TemporalDataError, match="non-empty"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [bad], [rev_upsert("k", "w", 1, 0, 100, 1)])


def test_null_event_at_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = fact_row(0, "k", None, 10)
    with pytest.raises(TemporalDataError, match="event_at"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [bad], [rev_upsert("k", "w", 1, 0, 100, 1)])


def test_null_known_at_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = fact_row(0, "k", 1, None)
    with pytest.raises(TemporalDataError, match="known_at"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [bad], [rev_upsert("k", "w", 1, 0, 100, 1)])


def test_null_recorded_at_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = rev_upsert("k", "w", 1, 0, 100, 1)
    bad["recorded_at"] = None
    with pytest.raises(TemporalDataError, match="recorded_at"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])


def test_bad_op_rejected(temporal_client_factory, temporal_server, tmp_path):
    bad = rev_upsert("k", "w", 1, 0, 100, 1)
    bad["op"] = "delete"
    with pytest.raises(TemporalDataError, match="upsert.*withdraw"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])


def test_upsert_missing_valid_from_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = rev_upsert("k", "w", 1, None, 100, 1)
    with pytest.raises(TemporalDataError, match="valid_from"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])


def test_upsert_missing_priority_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = rev_upsert("k", "w", 1, 0, 100, None)
    with pytest.raises(TemporalDataError, match="priority"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])


def test_upsert_valid_to_not_after_valid_from_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = rev_upsert("k", "w", 1, 100, 100, 1)   # vt == vf: not strict
    with pytest.raises(TemporalDataError, match="strictly greater"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])
    bad2 = rev_upsert("k", "w", 1, 100, 50, 1)   # vt < vf
    with pytest.raises(TemporalDataError, match="strictly greater"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad2])


def test_withdraw_with_non_null_interval_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = rev_withdraw("k", "w", 1)
    bad["valid_from"] = 0
    with pytest.raises(TemporalDataError, match="withdraw.*null"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])


def test_withdraw_with_non_null_dim_column_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = rev_withdraw("k", "w", 1)
    bad["service_area"] = "x"
    with pytest.raises(TemporalDataError, match="withdraw.*null"):
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       [fact_row(0, "k", 1, 10)], [bad])


def test_bad_row_diagnostic_has_source_coordinates(
    temporal_client_factory, temporal_server, tmp_path
):
    bad = fact_row(0, "k", None, 10)
    facts = [fact_row(1, "k", 1, 10), fact_row(2, "k", 2, 10), bad]
    try:
        _run_with_rows(temporal_client_factory, temporal_server, tmp_path,
                       facts, [rev_upsert("k", "w", 1, 0, 100, 1)])
        assert False, "expected TemporalDataError"
    except TemporalDataError as exc:
        diag = exc.diagnostics[0]
        assert diag["shard_id"] == "facts-0"
        assert diag["row_group"] == 0
        assert diag["row"] == 2          # third row of the shard
        assert diag["column"] == "event_at"


# ------------------------------------------------------- duplicate revisions
def test_duplicate_revision_cross_shard_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    rev = rev_upsert("k", "w", 100, 0, 50, 1)
    rev_manifests = [
        _manifest(temporal_server, tmp_path, "revisions-0.parquet", [rev],
                  REVISION_SCHEMA),
        _manifest(temporal_server, tmp_path, "revisions-1.parquet", [rev],
                  REVISION_SCHEMA),
    ]
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                [fact_row(0, "k", 10, 200)], FACT_SCHEMA)]
    client = temporal_client_factory()
    try:
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )
        assert False, "expected TemporalDataError"
    except TemporalDataError as exc:
        assert "duplicate revision" in str(exc)
        occ = exc.diagnostics[0]["occurrences"]
        assert {o["shard_id"] for o in occ} == {"revisions-0", "revisions-1"}


def test_duplicate_revision_identical_content_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    """Identical content on both occurrences is still a job error."""
    rev = rev_upsert("k", "w", 100, 0, 50, 1)
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               [rev, rev], REVISION_SCHEMA, rows_per_group=1)]
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                [fact_row(0, "k", 10, 200)], FACT_SCHEMA)]
    client = temporal_client_factory()
    with pytest.raises(TemporalDataError, match="duplicate revision"):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )


def test_duplicate_revision_unseen_by_any_fact_rejected(
    temporal_client_factory, temporal_server, tmp_path
):
    """A duplicate in a key with no facts is still a job error."""
    rev = rev_upsert("no-facts-key", "w", 100, 0, 50, 1)
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               [rev, rev], REVISION_SCHEMA, rows_per_group=1)]
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                [fact_row(0, "other-key", 10, 200)],
                                FACT_SCHEMA)]
    client = temporal_client_factory()
    with pytest.raises(TemporalDataError, match="duplicate revision"):
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )


def test_duplicate_revision_error_includes_both_sources(
    temporal_client_factory, temporal_server, tmp_path
):
    rev = rev_upsert("k", "w", 100, 0, 50, 1)
    rev_manifests = [
        _manifest(temporal_server, tmp_path, "revisions-0.parquet", [rev],
                  REVISION_SCHEMA),
        _manifest(temporal_server, tmp_path, "revisions-1.parquet", [rev],
                  REVISION_SCHEMA),
    ]
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                [fact_row(0, "k", 10, 200)], FACT_SCHEMA)]
    client = temporal_client_factory()
    try:
        _run_request(
            _request_dict(fact_manifests, rev_manifests,
                          tmp_path / "out" / "wide.parquet"),
            client, tmp_path,
        )
        assert False, "expected TemporalDataError"
    except TemporalDataError as exc:
        diag = exc.diagnostics[0]
        assert diag["entity_key"] == "k"
        assert diag["window_id"] == "w"
        assert diag["recorded_at"] == 100
        assert len(diag["occurrences"]) == 2
        for o in diag["occurrences"]:
            assert "shard_id" in o and "row_group" in o and "row" in o


# ------------------------------------------------------------ version swap
def test_version_swap_before_run_fails(
    temporal_client_factory, temporal_server, tmp_path
):
    """A shard that changed since the manifest was pinned fails the run."""
    facts = [fact_row(0, "k", 1, 10)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    # Swap the revision shard to a different version (no history retained).
    new_rev = [rev_upsert("k", "w2", 2, 0, 200, 2)]
    _write_parquet_file(tmp_path, "revisions-0.parquet", new_rev, REVISION_SCHEMA)
    new_data = open(str(tmp_path / "revisions-0.parquet"), "rb").read()
    temporal_server.swap_version("/revisions-0.parquet", new_data,
                                 '"new-etag"', retain_history=False)
    client = temporal_client_factory()
    request = TemporalEnrichRequest.from_dict(
        _request_dict(fact_manifests, rev_manifests,
                      tmp_path / "out" / "wide.parquet")
    )
    with pytest.raises(VersionMismatchError):
        TemporalEnrichEngine(client, request).run()
    # No output file was created.
    assert not os.path.exists(request.output_path)


def test_version_swap_mid_run_fails_and_old_output_intact(
    temporal_client_factory, temporal_server, tmp_path
):
    """A version change mid-run aborts and leaves the old wide table intact."""
    facts = [fact_row(i, "k", 1, 10) for i in range(20)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    client = temporal_client_factory()
    # First successful run publishes a valid wide table.
    request = TemporalEnrichRequest.from_dict(
        _request_dict(fact_manifests, rev_manifests,
                      tmp_path / "out" / "wide.parquet")
    )
    TemporalEnrichEngine(client, request).run()
    old_bytes = open(request.output_path, "rb").read()

    # Swap the fact shard; the next run must fail and keep the old table.
    new_facts = [fact_row(100, "k", 1, 10)]
    _write_parquet_file(tmp_path, "facts-0.parquet", new_facts, FACT_SCHEMA)
    new_data = open(str(tmp_path / "facts-0.parquet"), "rb").read()
    temporal_server.swap_version("/facts-0.parquet", new_data, '"new-etag"',
                                 retain_history=False)
    with pytest.raises(VersionMismatchError):
        TemporalEnrichEngine(client, request).run()
    assert open(request.output_path, "rb").read() == old_bytes


# -------------------------------------------------------------- publish
def test_publish_failure_leaves_old_output_intact(
    temporal_client_factory, temporal_server, tmp_path
):
    """A write failure aborts the run and preserves the previous wide table."""
    facts = [fact_row(0, "k", 1, 10)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    client = temporal_client_factory()
    out_path = tmp_path / "out" / "wide.parquet"
    request = TemporalEnrichRequest.from_dict(
        _request_dict(fact_manifests, rev_manifests, out_path)
    )
    TemporalEnrichEngine(client, request).run()
    old_bytes = open(out_path, "rb").read()

    # Make the output directory read-only so the staging/temp files can't
    # be created (a real write failure at the publish step).
    out_dir = tmp_path / "out"
    out_dir.chmod(0o555)
    try:
        with pytest.raises(TemporalPublishError):
            TemporalEnrichEngine(client, request).run()
    finally:
        out_dir.chmod(0o755)
    # The old wide table is untouched.
    assert open(out_path, "rb").read() == old_bytes
    # No staging temp files remain.
    assert not list(out_dir.glob("*.tmp"))


def test_stale_staging_file_removed_on_new_run(
    temporal_client_factory, temporal_server, tmp_path
):
    """A leftover staging temp file is removed before a new run."""
    facts = [fact_row(0, "k", 1, 10)]
    revisions = [rev_upsert("k", "w", 1, 0, 100, 1)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 1, 10
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        1, 10,
    )
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    staging = out_dir / ".temporal-out-abc123.parquet.tmp"
    staging.write_bytes(b"garbage")
    client = temporal_client_factory()
    request = TemporalEnrichRequest.from_dict(
        _request_dict(fact_manifests, rev_manifests, out_dir / "wide.parquet")
    )
    TemporalEnrichEngine(client, request).run()
    # The stale staging file is gone and the real output exists.
    assert not staging.exists()
    assert os.path.exists(request.output_path)


def test_failed_run_leaves_no_staging_files(
    temporal_client_factory, temporal_server, tmp_path
):
    """A failed run removes its temporary files."""
    bad = fact_row(0, "k", None, 10)
    fact_manifests = [_manifest(temporal_server, tmp_path, "facts-0.parquet",
                                [bad], FACT_SCHEMA)]
    rev_manifests = [_manifest(temporal_server, tmp_path, "revisions-0.parquet",
                               [rev_upsert("k", "w", 1, 0, 100, 1)],
                               REVISION_SCHEMA)]
    client = temporal_client_factory()
    out_dir = tmp_path / "out"
    request = TemporalEnrichRequest.from_dict(
        _request_dict(fact_manifests, rev_manifests, out_dir / "wide.parquet")
    )
    with pytest.raises(TemporalDataError):
        TemporalEnrichEngine(client, request).run()
    # No staging files or spill directories remain.
    assert not list(out_dir.glob("*.tmp")) if out_dir.exists() else True
    spill_root = out_dir / ".pqr-temporal-tmp"
    assert not spill_root.exists()


# --------------------------------------------------------------- cancellation
def test_cancelled_run_removes_temp_files(
    temporal_client_factory, temporal_server, tmp_path
):
    facts = [fact_row(i, f"k{i % 5}", 1, 10) for i in range(200)]
    revisions = [rev_upsert(f"k{i % 5}", f"w{i}", i, 0, 100, 1)
                 for i in range(50)]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA, 2, 20
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        2, 20,
    )
    client = temporal_client_factory()
    token = client.new_token()
    request = TemporalEnrichRequest.from_dict(
        _request_dict(fact_manifests, rev_manifests,
                      tmp_path / "out" / "wide.parquet")
    )
    engine = TemporalEnrichEngine(client, request, token=token)
    # Cancel as soon as the first row groups are fetched.
    token.cancel()
    with pytest.raises(Exception) as exc_info:
        engine.run()
    assert "cancel" in str(exc_info.value).lower()
    # Temp files are cleaned up.
    spill_root = tmp_path / "out" / ".pqr-temporal-tmp"
    assert not spill_root.exists()
    assert not os.path.exists(request.output_path)
