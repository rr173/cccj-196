"""Quality-audit failure modes: every failure is loud and never half-publishes.

Each test asserts the precise exception class, the absence of formal output,
and that a previously published successful result stays byte-identical.
"""

from __future__ import annotations

import hashlib
import os
import threading

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import (
    AuditSchemaError,
    QualityAuditEngine,
    QualityAuditRequest,
    QualityContract,
    RetryExhaustedError,
    SnapshotManifest,
    VersionMismatchError,
)
from parquet_range.exceptions import (
    RangeProtocolError,
    TruncatedResponseError,
)

from quality_conftest import (
    etag_of,
    make_quality_rows,
    quality_contract,
    write_quality_parquet,
)


def _no_formal_output(request):
    assert not os.path.exists(request.violations_path)
    assert not os.path.exists(request.report_path)


def _seed_result(client, request):
    """A known-good published result that failures must never overwrite."""
    QualityAuditEngine(client, request).run()
    return (
        open(request.violations_path, "rb").read(),
        open(request.report_path, "rb").read(),
    )


def test_schema_mismatch_missing_field_and_wrong_type(
    quality_client_factory, quality_snapshot, make_quality_request
):
    # Declared field missing from the file.
    contract = quality_contract()
    contract["fields"]["ghost_col"] = "int64"
    contract["rules"].append(
        {"id": "ghost-rule", "kind": "not_null", "field": "ghost_col"}
    )
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot, contract=contract)
    with pytest.raises(AuditSchemaError) as exc:
        QualityAuditEngine(client, request).run()
    assert any(d["field"] == "ghost_col" and "missing" in d["problem"]
               for d in exc.value.diagnostics)
    _no_formal_output(request)

    # Declared type disagrees with the file's Arrow type.
    contract2 = quality_contract()
    contract2["fields"]["score"] = "float32"  # file has float64
    request2 = make_quality_request(
        quality_snapshot, contract=contract2,
        violations_name="v2.jsonl", report_name="r2.json",
    )
    with pytest.raises(AuditSchemaError) as exc2:
        QualityAuditEngine(client, request2).run()
    assert any(d["field"] == "score" and "type mismatch" in d["problem"]
               for d in exc2.value.diagnostics)
    _no_formal_output(request2)


def test_field_order_in_file_does_not_matter(
    quality_client_factory, quality_server, make_quality_request, tmp_path
):
    """Same fields in a different file order must still match exactly."""
    import random

    from quality_conftest import QUALITY_SCHEMA, make_clean_rows

    rows = make_clean_rows(20)
    order = list(QUALITY_SCHEMA.names)
    random.Random(1234).shuffle(order)
    shuffled_schema = pa.schema([QUALITY_SCHEMA.field(n) for n in order])
    path = str(tmp_path / "reordered.parquet")
    table = pa.table(
        {name: [r[name] for r in rows] for name in QUALITY_SCHEMA.names},
        schema=QUALITY_SCHEMA,
    ).select(order)
    pq.write_table(table.cast(shuffled_schema), path, row_group_size=10)
    data = open(path, "rb").read()
    etag = etag_of(data)
    quality_server.put_object("/reordered.parquet", data, etag)
    snapshot = {
        "url": quality_server.url_for("/reordered.parquet"),
        "etag": etag, "length": len(data), "data": data,
    }
    client = quality_client_factory()
    request = make_quality_request(snapshot)
    report = QualityAuditEngine(client, request).run()
    assert report["status"] == "completed"
    assert report["total_violations"] == 0


def test_bad_identity_fails_before_any_data_read(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    from dataclasses import replace as _dc_replace

    client = quality_client_factory()
    # Wrong entity length.
    bad_len = _dc_replace(
        make_quality_request(quality_snapshot),
        object=SnapshotManifest(
            quality_snapshot["url"], quality_snapshot["etag"],
            quality_snapshot["length"] + 10,
        ),
    )
    with pytest.raises(VersionMismatchError):
        QualityAuditEngine(client, bad_len).run()
    _no_formal_output(bad_len)
    # Wrong ETag (server has no such version -> conditional 412).
    bad_etag = _dc_replace(
        make_quality_request(quality_snapshot),
        object=SnapshotManifest(
            quality_snapshot["url"], '"deadbeefdeadbeefdeadbeefdeadbeef"',
            quality_snapshot["length"],
        ),
    )
    with pytest.raises(VersionMismatchError):
        QualityAuditEngine(client, bad_etag).run()
    _no_formal_output(bad_etag)
    # No data GET ever happened for these (only conditional HEAD/probe).
    assert quality_server.range_gets("/quality.parquet") == [] or all(
        e.range_header == "bytes=0-0"
        for e in quality_server.range_gets("/quality.parquet")
    )


def test_etag_swap_mid_audit_fails_and_preserves_old_result(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory(block_size=4096)
    request = make_quality_request(quality_snapshot)
    good_violations, good_report = _seed_result(client, request)

    barrier = quality_server.barrier
    name = "GET:/quality.parquet"
    barrier.arm(name)
    outcome = {}

    def runner():
        try:
            outcome["r"] = QualityAuditEngine(client, request).run()
        except BaseException as exc:  # noqa: BLE001
            outcome["e"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert barrier.wait_for_arrivals(name, 1, timeout=15)
    # Swap to a new version and DROP history: the still-pinned conditional
    # GETs of the running audit now receive a real 412.
    new_data = quality_snapshot["data"] + b"\x00" * 31
    new_etag = '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"'
    quality_server.swap_version("/quality.parquet", new_data, new_etag)
    barrier.release(name)
    t.join(timeout=30)
    assert not t.is_alive()
    assert isinstance(outcome.get("e"), VersionMismatchError), outcome
    # The previously published pair is byte-identical.
    assert open(request.violations_path, "rb").read() == good_violations
    assert open(request.report_path, "rb").read() == good_report


def test_412_data_get_fails_audit(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    quality_server.fault.force_412 = True
    with pytest.raises(VersionMismatchError):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)


def test_truncated_response_fails_audit(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory(max_retries=1)
    request = make_quality_request(quality_snapshot)
    quality_server.fault.truncate_body = 7
    with pytest.raises((TruncatedResponseError, RangeProtocolError,
                        RetryExhaustedError)):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)


def test_bad_content_range_fails_audit(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    quality_server.fault.bad_content_range = True
    with pytest.raises(RangeProtocolError):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)


def test_retry_exhaustion_fails_audit(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory(max_retries=1)
    request = make_quality_request(quality_snapshot)
    quality_server.fault.fail_times = 50  # far beyond the retry budget
    with pytest.raises(RetryExhaustedError):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)


def test_200_whole_object_fallback_fails_audit(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    quality_server.fault.force_200 = True
    with pytest.raises(RangeProtocolError):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)


def test_failure_never_overwrites_prior_success(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    good_violations, good_report = _seed_result(client, request)
    # Break the live identity; the rerun must fail and leave the seed intact.
    new_data = quality_snapshot["data"] + b"\x01"
    quality_server.swap_version(
        "/quality.parquet", new_data,
        '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"',
    )
    with pytest.raises(VersionMismatchError):
        QualityAuditEngine(client, request).run()
    assert open(request.violations_path, "rb").read() == good_violations
    assert open(request.report_path, "rb").read() == good_report


def test_weak_etag_object_rejected(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    quality_server.fault.weak_etag = True
    from parquet_range import WeakOrMissingETagError

    with pytest.raises((WeakOrMissingETagError, VersionMismatchError)):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)


def test_missing_etag_object_rejected(
    quality_client_factory, quality_snapshot, make_quality_request,
    quality_server,
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    quality_server.fault.missing_etag = True
    from parquet_range import WeakOrMissingETagError

    with pytest.raises((WeakOrMissingETagError, VersionMismatchError,
                        RangeProtocolError)):
        QualityAuditEngine(client, request).run()
    _no_formal_output(request)
