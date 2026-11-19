"""End-to-end quality-audit correctness against the independent oracle.

The oracle (``tests/quality_oracle.py``) computes expected violations with a
fully separate implementation; the product is never used as its own oracle.
"""

from __future__ import annotations

import hashlib
import json
import os

import pytest

from parquet_range import (
    QualityAuditEngine,
    QualityAuditRequest,
    QualityContract,
    SnapshotManifest,
    read_violations,
)

from quality_conftest import (
    etag_of,
    make_clean_rows,
    make_quality_rows,
    quality_contract,
    write_quality_parquet,
)
from quality_oracle import oracle_expected_violations, version_id_of


def _identity(snapshot):
    return {
        "url": snapshot["url"],
        "etag": snapshot["etag"],
        "length": snapshot["length"],
        "version_id": version_id_of(
            snapshot["url"], snapshot["etag"], snapshot["length"]
        ),
    }


def _load_violations(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_audit_matches_independent_oracle(
    quality_client_factory, quality_snapshot, make_quality_request
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    report = QualityAuditEngine(client, request).run()

    expected = oracle_expected_violations(
        quality_snapshot["data"], quality_contract(), _identity(quality_snapshot)
    )
    got = _load_violations(request.violations_path)
    assert len(expected) > 0, "oracle fixture must produce violations"
    assert got == expected
    assert report["total_violations"] == len(expected)
    assert report["rows_scanned"] == 111
    assert report["status"] == "completed"
    # Every rule fired at least once on this fixture.
    for rid, entry in report["violations_per_rule"].items():
        assert entry["violations"] > 0, f"rule {rid} never fired"


def test_multiple_rules_hit_same_row_without_merging(
    quality_client_factory, quality_snapshot, make_quality_request
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    got = _load_violations(request.violations_path)
    by_row: dict[tuple, list[str]] = {}
    for rec in got:
        by_row.setdefault((rec["row_group"], rec["row"]), []).append(rec["rule_id"])
    multi = [rules for rules in by_row.values() if len(rules) > 1]
    assert multi, "fixture should have rows violating several rules"
    for rules in multi:
        # One record per (row, rule): identities never merged or dropped.
        assert len(rules) == len(set(rules))


def test_results_deterministically_ordered(
    quality_client_factory, quality_snapshot, make_quality_request
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    report = QualityAuditEngine(client, request).run()
    rule_index = {rid: i for i, rid in enumerate(report["rules"])}
    got = _load_violations(request.violations_path)
    triples = [
        (rec["row_group"], rec["row"], rule_index[rec["rule_id"]]) for rec in got
    ]
    assert triples == sorted(triples)
    assert len(set(triples)) == len(triples)  # strictly increasing


def test_repeated_runs_byte_stable(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    client = quality_client_factory()
    req1 = make_quality_request(
        quality_snapshot, violations_name="v1.jsonl", report_name="r1.json"
    )
    req2 = make_quality_request(
        quality_snapshot, violations_name="v2.jsonl", report_name="r2.json"
    )
    QualityAuditEngine(client, req1).run()
    QualityAuditEngine(client, req2).run()
    assert open(req1.violations_path, "rb").read() == open(
        req2.violations_path, "rb"
    ).read()
    # Reports are identical apart from volatile timing/resource fields.
    r1 = json.load(open(req1.report_path))
    r2 = json.load(open(req2.report_path))
    for volatile in ("started_at_utc", "finished_at_utc", "resources"):
        r1.pop(volatile)
        r2.pop(volatile)
    r1["violations"]["path"] = r2["violations"]["path"] = "<normalized>"
    assert r1 == r2


def test_empty_table_zero_rows(
    quality_client_factory, quality_server, make_quality_request, tmp_path
):
    path = str(tmp_path / "empty.parquet")
    data = write_quality_parquet(path, [], groups=3)
    etag = etag_of(data)
    quality_server.put_object("/empty.parquet", data, etag)
    snapshot = {
        "url": quality_server.url_for("/empty.parquet"),
        "etag": etag,
        "length": len(data),
        "data": data,
    }
    client = quality_client_factory()
    request = make_quality_request(snapshot)
    report = QualityAuditEngine(client, request).run()
    assert report["status"] == "completed"
    assert report["rows_scanned"] == 0
    assert report["total_violations"] == 0
    assert open(request.violations_path, "rb").read() == b""
    # The empty file still verifies offline.
    out = read_violations(request.report_path)
    assert out["verified"] and out["page"] == []


def test_zero_violations_report(
    quality_client_factory, quality_server, make_quality_request, tmp_path
):
    path = str(tmp_path / "clean.parquet")
    data = write_quality_parquet(path, make_clean_rows(50), groups=2)
    etag = etag_of(data)
    quality_server.put_object("/clean.parquet", data, etag)
    snapshot = {
        "url": quality_server.url_for("/clean.parquet"),
        "etag": etag,
        "length": len(data),
        "data": data,
    }
    client = quality_client_factory()
    request = make_quality_request(snapshot)
    report = QualityAuditEngine(client, request).run()
    assert report["total_violations"] == 0
    assert report["rows_scanned"] == 50
    assert all(
        entry["violations"] == 0
        for entry in report["violations_per_rule"].values()
    )
    assert open(request.violations_path, "rb").read() == b""
    # Empty violations file has a well-defined checksum (SHA-256 of nothing).
    assert report["violations"]["sha256"] == hashlib.sha256(b"").hexdigest()
    out = read_violations(request.report_path)
    assert out["verified"]


def test_null_nan_semantics_explicit(
    quality_client_factory, quality_server, make_quality_request, tmp_path
):
    """Default: null/NaN are not-applicable; flags turn them into violations."""
    contract = quality_contract()
    # Strip to a couple of targeted rules.
    contract["rules"] = [
        {"id": "score-default", "kind": "range", "field": "score",
         "min": 0.0, "max": 1e9},
        {"id": "score-strict", "kind": "range", "field": "score",
         "min": 0.0, "max": 1e9, "null_is_violation": True,
         "nan_is_violation": True},
        {"id": "name-default", "kind": "pattern", "field": "name",
         "pattern": ".*"},
        {"id": "name-strict", "kind": "pattern", "field": "name",
         "pattern": ".*", "null_is_violation": True},
    ]
    import pyarrow as pa
    import pyarrow.parquet as pq
    from quality_conftest import QUALITY_SCHEMA

    rows = []
    for i in range(4):
        rows.append({
            "id": i, "small_i": 0, "big_u": i, "byte_u": 1,
            "score": [1.0, float("nan"), None, 2.0][i],
            "ratio": 0.0, "flag": True,
            "name": ["ok", None, "also_ok", None][i],
            "ccy": "USD",
            "created": None, "updated": None, "ts_ns": None, "day": None,
            "meta": None, "tags": None,
            **{f"wide_{k}": 0.0 for k in range(6)},
        })
    path = str(tmp_path / "nullnan.parquet")
    table = pa.table(
        {name: [r[name] for r in rows] for name in QUALITY_SCHEMA.names},
        schema=QUALITY_SCHEMA,
    )
    pq.write_table(table, path, row_group_size=4)
    data = open(path, "rb").read()
    etag = etag_of(data)
    quality_server.put_object("/nullnan.parquet", data, etag)
    snapshot = {
        "url": quality_server.url_for("/nullnan.parquet"),
        "etag": etag, "length": len(data), "data": data,
    }
    client = quality_client_factory()
    request = make_quality_request(snapshot, contract=contract)
    report = QualityAuditEngine(client, request).run()
    got = _load_violations(request.violations_path)
    by_rule: dict[str, list] = {}
    for rec in got:
        by_rule.setdefault(rec["rule_id"], []).append(rec)
    # Default rules: null/NaN are skipped, never violations.
    assert "score-default" not in by_rule
    assert "name-default" not in by_rule
    # Strict rules: exactly the NaN row and the null rows, with reason codes.
    strict_score = by_rule["score-strict"]
    assert [(r["row"], r["reason"]) for r in strict_score] == [
        (1, "nan"), (2, "null"),
    ]
    strict_name = by_rule["name-strict"]
    assert [(r["row"], r["reason"]) for r in strict_name] == [
        (1, "null"), (3, "null"),
    ]
    # Skipped counters are reported for the default rules.
    per_rule = report["violations_per_rule"]
    assert per_rule["score-default"]["nan_skipped"] == 1
    assert per_rule["score-default"]["null_skipped"] == 1
    assert per_rule["name-default"]["null_skipped"] == 2


def test_unsigned_and_temporal_edges(
    quality_client_factory, quality_server, make_quality_request, tmp_path
):
    """uint64 max, int64 min, tz-aware equality at boundaries, ns precision."""
    import datetime

    import pyarrow as pa
    import pyarrow.parquet as pq

    contract = {
        "spec_version": 1,
        "fields": {
            "big_u": "uint64",
            "id": "int64",
            "created": "timestamp[us, tz=UTC]",
            "ts_ns": "timestamp[ns]",
        },
        "rules": [
            {"id": "u-max", "kind": "range", "field": "big_u",
             "min": 0, "max": 2 ** 64 - 2},
            {"id": "i-min", "kind": "range", "field": "id",
             "min": -(2 ** 63), "max": 0},
            # Bound expressed in a *different offset* than UTC: the instant
            # 2022-01-01T05:30:00+05:30 == 2022-01-01T00:00:00Z.
            {"id": "ts-bound", "kind": "range", "field": "created",
             "min": "2022-01-01T05:30:00+05:30",
             "max": "2022-01-01T05:30:00+05:30"},
            {"id": "ns-bound", "kind": "range", "field": "ts_ns",
             "min": "2022-01-01T00:00:00",
             "max": "2022-01-01T00:00:00"},
        ],
    }
    schema = pa.schema([
        pa.field("big_u", pa.uint64()),
        pa.field("id", pa.int64()),
        pa.field("created", pa.timestamp("us", tz="UTC")),
        pa.field("ts_ns", pa.timestamp("ns")),
    ])
    utc = datetime.timezone.utc
    instant = datetime.datetime(2022, 1, 1, tzinfo=utc)
    rows = {
        "big_u": [2 ** 64 - 1, 2 ** 64 - 2, 0],
        "id": [-(2 ** 63), 0, 1],
        "created": [instant] * 3,
        "ts_ns": [datetime.datetime(2022, 1, 1)] * 3,
    }
    path = str(tmp_path / "edges.parquet")
    pq.write_table(pa.table(rows, schema=schema), path)
    data = open(path, "rb").read()
    etag = etag_of(data)
    quality_server.put_object("/edges.parquet", data, etag)
    snapshot = {
        "url": quality_server.url_for("/edges.parquet"),
        "etag": etag, "length": len(data), "data": data,
    }
    client = quality_client_factory()
    request = make_quality_request(snapshot, contract=contract)
    QualityAuditEngine(client, request).run()
    got = _load_violations(request.violations_path)
    flat = {(r["rule_id"], r["row"], r["reason"]) for r in got}
    # uint64 max exceeds max=2^64-2; 2^64-2 itself and 0 pass.
    assert ("u-max", 0, "above_max") in flat
    assert not any(f[0] == "u-max" and f[1] in (1, 2) for f in flat)
    # int64 min is accepted (inclusive bound); 1 exceeds max=0.
    assert ("i-min", 2, "above_max") in flat
    assert not any(f[0] == "i-min" and f[1] in (0, 1) for f in flat)
    # The UTC instant equals both bounds (inclusive): no violations.
    assert not any(f[0] == "ts-bound" for f in flat)
    assert not any(f[0] == "ns-bound" for f in flat)


def test_only_referenced_columns_fetched(
    quality_client_factory, quality_server, tmp_path
):
    """Server log proves nested/wide unreferenced columns are never fetched."""
    # A larger file where the unreferenced columns dominate the byte volume.
    rows = make_quality_rows(12_000)
    path = str(tmp_path / "wide.parquet")
    data = write_quality_parquet(path, rows, groups=4)
    etag = etag_of(data)
    quality_server.put_object("/wide.parquet", data, etag)
    quality_server.clear_log()
    # A narrow contract: only id and score are ever needed.
    contract = {
        "spec_version": 1,
        "fields": {"id": "int64", "score": "float64"},
        "rules": [
            {"id": "id-non-null", "kind": "not_null", "field": "id"},
            {"id": "score-range", "kind": "range", "field": "score",
             "min": 0.0, "max": 1000.0},
        ],
    }
    snapshot = {
        "url": quality_server.url_for("/wide.parquet"),
        "etag": etag, "length": len(data), "data": data,
    }
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    from parquet_range import QualityAuditRequest, QualityContract, SnapshotManifest
    request = QualityAuditRequest(
        object=SnapshotManifest(snapshot["url"], etag, len(data)),
        contract=QualityContract.from_dict(contract),
        violations_path=str(out_dir / "v.jsonl"),
        report_path=str(out_dir / "r.json"),
        batch_rows=2048,
    )
    client = quality_client_factory(block_size=8192, max_merge_gap_blocks=0)
    report = QualityAuditEngine(client, request).run()
    wire = quality_server.physical_bytes("/wide.parquet")
    total = len(data)
    # id+score are ~16 of ~110 bytes/row; the audit must transfer clearly
    # less than the whole object and never touch the wide/nested chunks.
    assert 0 < wire < total * 0.55
    assert report["rows_scanned"] == 12_000
    assert report["resources"]["network_cache"]["wire_bytes"] == wire


def test_report_contents(
    quality_client_factory, quality_snapshot, make_quality_request
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    report = QualityAuditEngine(client, request).run()
    assert report["report_format"] == "pqr-quality-report"
    ident = report["identity"]
    assert ident["etag"] == quality_snapshot["etag"]
    assert ident["length"] == quality_snapshot["length"]
    assert ident["version_id"] == _identity(quality_snapshot)["version_id"]
    assert report["contract_sha256"] == QualityContract.from_dict(
        quality_contract()
    ).sha256
    assert report["rules"] == [r["id"] for r in quality_contract()["rules"]]
    matched = report["schema"]["matched_fields"]
    assert {f["name"] for f in matched} == set(
        quality_contract()["fields"]
    )
    assert report["schema"]["file_num_fields"] > len(matched)
    assert report["row_groups_scanned"] == 3
    assert report["row_groups_total"] == 3
    assert report["violations"]["records"] == report["total_violations"]
    assert report["violations"]["sha256"] == hashlib.sha256(
        open(request.violations_path, "rb").read()
    ).hexdigest()
    assert report["resources"]["peak"]["max_rss_bytes"] > 0
    assert report["resources"]["network_cache"]["physical_requests"] > 0


def test_offline_pagination_by_rule(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    quality_server.clear_log()

    all_violations = _load_violations(request.violations_path)
    # Page through one rule two records at a time; no network traffic at all.
    rule_id = "score-range"
    expected = [r for r in all_violations if r["rule_id"] == rule_id]
    collected = []
    offset = 0
    while True:
        out = read_violations(
            request.report_path, rule_id=rule_id, offset=offset, limit=2
        )
        collected.extend(out["page"])
        if offset + 2 >= out["page_info"]["matched"]:
            break
        offset += 2
    assert collected == expected
    assert quality_server.log == [], "offline reader must not touch the network"

    # Unfiltered pagination covers every record in file order.
    out = read_violations(request.report_path, limit=10_000)
    assert out["page"] == all_violations
    assert out["verified"] and out["order_stable"]


def test_offline_reader_detects_tampering(
    quality_client_factory, quality_snapshot, make_quality_request
):
    from parquet_range import AuditPublishError

    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    # Flip a byte in the violations file: verification must fail.
    with open(request.violations_path, "r+b") as fh:
        first = fh.read(1)
        fh.seek(0)
        fh.write(bytes([first[0] ^ 0x01]))
    with pytest.raises(AuditPublishError):
        read_violations(request.report_path)


def test_offline_reader_rejects_incomplete_or_foreign_report(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    from parquet_range import AuditPublishError

    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    # A report that is not a completed quality audit is refused.
    fake = tmp_path / "fake.json"
    fake.write_text(json.dumps({"status": "completed"}), encoding="utf-8")
    with pytest.raises(AuditPublishError):
        read_violations(str(fake))
    unfinished = tmp_path / "unfinished.json"
    unfinished.write_text(
        json.dumps({"report_format": "pqr-quality-report", "status": "failed"}),
        encoding="utf-8",
    )
    with pytest.raises(AuditPublishError):
        read_violations(str(unfinished))
    # A missing violations file is refused.
    report = json.load(open(request.report_path))
    report["violations"]["path"] = str(tmp_path / "does-not-exist.jsonl")
    broken = tmp_path / "broken.json"
    broken.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(AuditPublishError):
        read_violations(str(broken))
