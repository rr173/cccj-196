"""Diff failure modes: every invalid input fails loudly with no formal output.

These run against the real Range server and assert both the precise exception
class and that neither the JSONL facts nor the report are published (and any
pre-existing successful result stays byte-identical).
"""

from __future__ import annotations

import hashlib
import os

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import (
    DiffEngine,
    DiffRequest,
    DuplicateKeyError,
    ManifestError,
    NullKeyError,
    RetryExhaustedError,
    SchemaDriftError,
    SnapshotManifest,
    UnsupportedKeyTypeError,
    VersionMismatchError,
)
from parquet_range.exceptions import (
    RangeProtocolError,
    SegmentIntegrityError,
    TruncatedResponseError,
)

from diff_conftest import KEY_COLUMN, write_diff_parquet, _make_rows, baseline_spec


def _no_formal_output(request):
    assert not os.path.exists(request.diff_path)
    assert not os.path.exists(request.report_path)


def _publish_seed(client, request):
    """A known-good result that later failures must never overwrite."""
    DiffEngine(client, request).run()
    facts = open(request.diff_path, "rb").read()
    report = open(request.report_path, "rb").read()
    return facts, report


def test_duplicate_key_reports_identity_row_group_and_row(
    diff_client_factory, diff_server, tmp_path
):
    rows = _make_rows(baseline_spec())
    lp = str(tmp_path / "dup.parquet")
    data = write_diff_parquet(lp, rows, groups=4, duplicate_key=rows[0]["kid"])
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/dup.parquet", data, etag)
    # A normal right side.
    from diff_conftest import target_spec
    rp = str(tmp_path / "ok.parquet")
    rdata = write_diff_parquet(rp, _make_rows(target_spec()), groups=3)
    retag = '"' + hashlib.sha256(rdata).hexdigest()[:32] + '"'
    diff_server.put_object("/ok.parquet", rdata, retag)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/dup.parquet"), etag, len(data)),
        right=SnapshotManifest(diff_server.url_for("/ok.parquet"), retag, len(rdata)),
        key_column=KEY_COLUMN,
        compare_columns=("code", "tags", "blob"),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    client = diff_client_factory()
    with pytest.raises(DuplicateKeyError) as exc:
        DiffEngine(client, request).run()
    occ = exc.value.occurrences
    assert len(occ) >= 2
    for item in occ:
        assert item["identity"]["etag"] == etag
        assert isinstance(item["row_group"], int) and item["row_group"] >= 0
        assert isinstance(item["row"], int) and item["row"] >= 0
        assert item["side"] == "left"
    _no_formal_output(request)


def test_null_key_is_rejected(diff_client_factory, diff_server, tmp_path):
    rows = _make_rows(baseline_spec())
    lp = str(tmp_path / "nullk.parquet")
    data = write_diff_parquet(lp, rows, groups=4, null_key=True)
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/nullk.parquet", data, etag)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/nullk.parquet"), etag, len(data)),
        right=SnapshotManifest(diff_server.url_for("/nullk.parquet"), etag, len(data)),
        key_column=KEY_COLUMN,
        compare_columns=("code",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    client = diff_client_factory()
    with pytest.raises(NullKeyError):
        DiffEngine(client, request).run()
    _no_formal_output(request)


def test_float_key_type_rejected(diff_client_factory, diff_server, tmp_path):
    schema = pa.schema([pa.field("fk", pa.float64()), pa.field("v", pa.string())])
    p = str(tmp_path / "float.parquet")
    pq.write_table(pa.table({"fk": [1.0, 2.0], "v": ["a", "b"]}, schema=schema), p)
    data = open(p, "rb").read()
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/float.parquet", data, etag)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/float.parquet"), etag, len(data)),
        right=SnapshotManifest(diff_server.url_for("/float.parquet"), etag, len(data)),
        key_column="fk",
        compare_columns=("v",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    client = diff_client_factory()
    with pytest.raises((UnsupportedKeyTypeError, SchemaDriftError)):
        DiffEngine(client, request).run()
    _no_formal_output(request)


def test_schema_type_drift_field_diagnostics_and_no_publish(
    diff_client_factory, diff_server, tmp_path
):
    s_l = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.string())])
    s_r = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.int32())])
    lp = str(tmp_path / "dl.parquet")
    rp = str(tmp_path / "dr.parquet")
    pq.write_table(pa.table({"kid": [1, 2], "v": ["a", "b"]}, schema=s_l), lp)
    pq.write_table(pa.table({"kid": [1, 2], "v": [1, 2]}, schema=s_r), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le = '"' + hashlib.sha256(ld).hexdigest()[:32] + '"'
    re_ = '"' + hashlib.sha256(rd).hexdigest()[:32] + '"'
    diff_server.put_object("/dl.parquet", ld, le)
    diff_server.put_object("/dr.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/dl.parquet"), le, len(ld)),
        right=SnapshotManifest(diff_server.url_for("/dr.parquet"), re_, len(rd)),
        key_column="kid",
        compare_columns=("v",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    client = diff_client_factory()
    with pytest.raises(SchemaDriftError) as exc:
        DiffEngine(client, request).run()
    diag = exc.value.diagnostics
    assert any(d["field"] == "v" and d["problem"].startswith("type mismatch")
               for d in diag)
    _no_formal_output(request)


def test_missing_field_drift_not_silently_null_filled(
    diff_client_factory, diff_server, tmp_path
):
    s_l = pa.schema([pa.field("kid", pa.int64()), pa.field("a", pa.string()),
                     pa.field("b", pa.string())])
    s_r = pa.schema([pa.field("kid", pa.int64()), pa.field("a", pa.string())])
    lp = str(tmp_path / "ml.parquet")
    rp = str(tmp_path / "mr.parquet")
    pq.write_table(pa.table({"kid": [1], "a": ["x"], "b": ["y"]}, schema=s_l), lp)
    pq.write_table(pa.table({"kid": [1], "a": ["x"]}, schema=s_r), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le = '"' + hashlib.sha256(ld).hexdigest()[:32] + '"'
    re_ = '"' + hashlib.sha256(rd).hexdigest()[:32] + '"'
    diff_server.put_object("/ml.parquet", ld, le)
    diff_server.put_object("/mr.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/ml.parquet"), le, len(ld)),
        right=SnapshotManifest(diff_server.url_for("/mr.parquet"), re_, len(rd)),
        key_column="kid",
        compare_columns=("a", "b"),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    client = diff_client_factory()
    with pytest.raises(SchemaDriftError) as exc:
        DiffEngine(client, request).run()
    assert any(d["field"] == "b" and "missing on right" in d["problem"]
               for d in exc.value.diagnostics)
    _no_formal_output(request)


def test_bad_manifest_identity_length_and_etag_fail_before_fetch(
    diff_client_factory, diff_snapshots, make_request, tmp_path
):
    lurl, le, llen, _ = diff_snapshots["left"]
    rurl, re_, rlen, _ = diff_snapshots["right"]
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    client = diff_client_factory()
    # Wrong length.
    bad_len = DiffRequest(
        left=SnapshotManifest(lurl, le, llen + 999),
        right=SnapshotManifest(rurl, re_, rlen),
        key_column="kid",
        compare_columns=("code",),
        diff_path=str(out / "f1.jsonl"),
        report_path=str(out / "r1.json"),
        memory_limit=4 << 20,
    )
    with pytest.raises(VersionMismatchError):
        DiffEngine(client, bad_len).run()
    _no_formal_output(bad_len)
    # Wrong ETag (server has no such retained version -> conditional 412).
    bad_etag = DiffRequest(
        left=SnapshotManifest(lurl, '"deadbeefdeadbeefdeadbeefdeadbeef"', llen),
        right=SnapshotManifest(rurl, re_, rlen),
        key_column="kid",
        compare_columns=("code",),
        diff_path=str(out / "f2.jsonl"),
        report_path=str(out / "r2.json"),
        memory_limit=4 << 20,
    )
    with pytest.raises(VersionMismatchError):
        DiffEngine(client, bad_etag).run()
    _no_formal_output(bad_etag)


def test_malformed_manifest_rejected(tmp_path):
    with pytest.raises(ManifestError):
        DiffRequest.from_dict({"left": {"url": "http://h/x"}, "right": {}})
    with pytest.raises(ManifestError):
        DiffRequest.from_dict({"left": {"url": "http://h/l", "etag": '"e"',
                                        "length": 10},
                               "right": {"url": "http://h/r", "etag": '"e"',
                                         "length": 10},
                               "key_column": "k",
                               "compare_columns": [],
                               "diff_path": "a", "report_path": "b"})


def test_weak_etag_manifest_rejected(tmp_path):
    with pytest.raises(ManifestError):
        SnapshotManifest.from_dict(
            {"url": "http://h/x", "etag": 'W/"weak"', "length": 10}, side="left"
        )


def test_failure_never_overwrites_prior_success(
    diff_client_factory, diff_snapshots, make_request, tmp_path, diff_server
):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    good_facts, good_report = _publish_seed(client, request)
    # Now break the live identity; rerun must fail and leave the seed intact.
    lurl, le, llen, ldata = diff_snapshots["left"]
    diff_server.swap_version(
        "/base.parquet", ldata + b"\x01",
        '"' + hashlib.sha256(ldata + b"\x01").hexdigest()[:32] + '"',
    )
    with pytest.raises(VersionMismatchError):
        DiffEngine(client, request).run()
    assert open(request.diff_path, "rb").read() == good_facts
    assert open(request.report_path, "rb").read() == good_report


def test_etag_swap_mid_diff_cannot_mix_versions(
    diff_client_factory, diff_server, tmp_path
):
    """Swap one ETag after footer/probe but during data GETs -> whole diff fails.

    The retained-history server keeps the old version addressable by ETag, but
    a swap that DROPS history makes the still-pinned conditional data GETs 412,
    proving a half-old/half-new merge can never be published.
    """
    from diff_conftest import target_spec

    rows = _make_rows(baseline_spec())
    lp = str(tmp_path / "swap.parquet")
    data = write_diff_parquet(lp, rows, groups=4)
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/swap.parquet", data, etag)
    rp = str(tmp_path / "swapr.parquet")
    rdata = write_diff_parquet(rp, _make_rows(target_spec()), groups=3)
    retag = '"' + hashlib.sha256(rdata).hexdigest()[:32] + '"'
    diff_server.put_object("/swapr.parquet", rdata, retag)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/swap.parquet"), etag, len(data)),
        right=SnapshotManifest(diff_server.url_for("/swapr.parquet"), retag,
                                len(rdata)),
        key_column=KEY_COLUMN,
        compare_columns=("code", "tags", "blob"),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    # Drop history on swap so the old ETag becomes unresolvable (real 412).
    new_data = data + b"\x00" * 31
    new_etag = '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"'
    barrier = diff_server.barrier
    name = "GET:/swap.parquet"
    barrier.arm(name)
    client = diff_client_factory(block_size=8192)

    import threading

    outcome = {}

    def runner():
        try:
            outcome["r"] = DiffEngine(client, request).run()
        except BaseException as exc:  # noqa: BLE001
            outcome["e"] = exc

    t = threading.Thread(target=runner, daemon=True)
    t.start()
    assert barrier.wait_for_arrivals(name, 1, timeout=15)
    diff_server.swap_version("/swap.parquet", new_data, new_etag)
    barrier.release(name)
    t.join(timeout=30)
    assert not t.is_alive()
    assert isinstance(outcome.get("e"), VersionMismatchError), outcome
    _no_formal_output(request)


def test_412_data_get_fails_diff(diff_client_factory, diff_snapshots,
                                make_request, tmp_path, diff_server):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    # Probe/first contact stays clean; only data GETs are forced to 412.
    diff_server.fault.force_412 = True
    with pytest.raises(VersionMismatchError):
        DiffEngine(client, request).run()
    _no_formal_output(request)


def test_retry_exhaustion_fails_diff(diff_client_factory, diff_snapshots,
                                     make_request, tmp_path, diff_server):
    client = diff_client_factory(max_retries=1)
    request = make_request(diff_snapshots)
    diff_server.fault.fail_times = 50  # far beyond the retry budget
    with pytest.raises(RetryExhaustedError):
        DiffEngine(client, request).run()
    _no_formal_output(request)


def test_truncated_response_fails_diff(diff_client_factory, diff_snapshots,
                                       make_request, tmp_path, diff_server):
    client = diff_client_factory(max_retries=1)
    request = make_request(diff_snapshots)
    diff_server.fault.truncate_body = 7
    with pytest.raises((TruncatedResponseError, RangeProtocolError,
                        RetryExhaustedError, SegmentIntegrityError)):
        DiffEngine(client, request).run()
    _no_formal_output(request)
