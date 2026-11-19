"""Same-URL dual-identity pinning and schema-contract edge cases.

These prove the two snapshots of one diff are treated as two explicit
identities even when they share a URL, and that field *order* differences are
tolerated while real dictionary/type drift is refused.
"""

from __future__ import annotations

import hashlib
import json

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import (
    DiffEngine,
    DiffRequest,
    SchemaDriftError,
    SnapshotManifest,
)

from diff_oracle import oracle_expected


def _etag(data: bytes) -> str:
    return '"' + hashlib.sha256(data).hexdigest()[:32] + '"'


def test_same_url_two_versions_are_distinct_identities(
    diff_client_factory, diff_server, tmp_path
):
    """Both manifests point at one URL but pin two retained ETags.

    The retained-history server answers each conditional request with the
    matching version, so the diff must compare the two explicit versions rather
    than probing the current object twice and mixing time-distinct content.
    """
    schema = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.string())])
    v1_path = str(tmp_path / "same.parquet")
    v2_path = str(tmp_path / "same_v2.parquet")
    pq.write_table(pa.table({"kid": [1, 2], "v": ["a", "b"]}, schema=schema),
                   v1_path)
    pq.write_table(pa.table({"kid": [2, 3], "v": ["b", "c"]}, schema=schema),
                   v2_path)
    v1 = open(v1_path, "rb").read()
    v2 = open(v2_path, "rb").read()
    e1, e2 = _etag(v1), _etag(v2)
    url = diff_server.url_for("/same.parquet")
    # v1 is current but BOTH versions are retained and addressable by ETag.
    diff_server.put_object("/same.parquet", v1, e1)
    diff_server.swap_version("/same.parquet", v2, e2, retain_history=True)

    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(url, e1, len(v1)),
        right=SnapshotManifest(url, e2, len(v2)),
        key_column="kid",
        compare_columns=("v",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    client = diff_client_factory()
    report = DiffEngine(client, request).run()
    assert report["counts"] == {"insert": 1, "delete": 1, "update": 0,
                                "no-change": 1}
    # Both pinned identities are recorded even though the URL is identical.
    assert report["left_identity"]["url"] == report["right_identity"]["url"]
    assert report["left_identity"]["etag"] == e1
    assert report["right_identity"]["etag"] == e2
    # The service log shows conditional requests for BOTH ETags on one path.
    gets = diff_server.range_gets("/same.parquet")
    used = {e.if_match for e in gets}
    assert e1 in used and e2 in used
    assert all(e.status == 206 for e in gets)


def test_field_reordering_is_not_misjudged_as_drift_or_inequality(
    diff_client_factory, diff_server, tmp_path
):
    """Same fields in a different order: diff proceeds and matches the oracle."""
    s_l = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("a", pa.string()),
        pa.field("b", pa.int32()),
    ])
    s_r = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("b", pa.int32()),
        pa.field("a", pa.string()),
    ])
    lp, rp = str(tmp_path / "ol.parquet"), str(tmp_path / "or.parquet")
    pq.write_table(pa.table({"kid": [1, 2], "a": ["x", "y"], "b": [10, 20]},
                            schema=s_l), lp)
    pq.write_table(pa.table({"kid": [2, 1], "b": [20, 10], "a": ["y", "x"]},
                            schema=s_r), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le, re_ = _etag(ld), _etag(rd)
    diff_server.put_object("/ol.parquet", ld, le)
    diff_server.put_object("/or.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/ol.parquet"), le, len(ld)),
        right=SnapshotManifest(diff_server.url_for("/or.parquet"), re_, len(rd)),
        key_column="kid",
        compare_columns=("a", "b"),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    report = DiffEngine(diff_client_factory(), request).run()
    # Reordered-but-equal rows are all no-change; ordering of fields never
    # causes a false update.
    assert report["counts"]["update"] == 0
    assert report["counts"]["no-change"] == 2
    facts = [json.loads(line) for line in open(request.diff_path)]
    expected = oracle_expected(ld, rd, "kid", ["a", "b"])
    assert [f["kind"] for f in facts] == [e[1]["kind"] for e in expected]


def test_dictionary_index_width_drift_is_refused_with_diagnostics(
    diff_client_factory, diff_server, tmp_path
):
    """Same logical string values but a different dictionary type is drift."""
    s_l = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("c", pa.dictionary(pa.int8(), pa.string())),
    ])
    s_r = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("c", pa.dictionary(pa.int32(), pa.string())),
    ])
    lp, rp = str(tmp_path / "dl.parquet"), str(tmp_path / "dr.parquet")

    def dict_array(index_type):
        dictionary = pa.array(["x"])
        indices = pa.array([0], type=index_type)
        return pa.DictionaryArray.from_arrays(indices, dictionary)

    pq.write_table(
        pa.table({"kid": pa.array([1], pa.int64()),
                  "c": dict_array(pa.int8())}, schema=s_l), lp)
    pq.write_table(
        pa.table({"kid": pa.array([1], pa.int64()),
                  "c": dict_array(pa.int32())}, schema=s_r), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le, re_ = _etag(ld), _etag(rd)
    diff_server.put_object("/dl.parquet", ld, le)
    diff_server.put_object("/dr.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/dl.parquet"), le, len(ld)),
        right=SnapshotManifest(diff_server.url_for("/dr.parquet"), re_, len(rd)),
        key_column="kid",
        compare_columns=("c",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    with pytest.raises(SchemaDriftError) as exc:
        DiffEngine(diff_client_factory(), request).run()
    assert any(d["field"] == "c" for d in exc.value.diagnostics)
    assert not (out / "f.jsonl").exists()


def test_dictionary_vs_plain_same_value_type_is_drift(
    diff_client_factory, diff_server, tmp_path
):
    s_l = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("c", pa.dictionary(pa.int8(), pa.string())),
    ])
    s_r = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("c", pa.string()),
    ])
    lp, rp = str(tmp_path / "dl.parquet"), str(tmp_path / "dr.parquet")
    pq.write_table(
        pa.table({"kid": [1], "c": pa.array(["x"]).dictionary_encode()},
                 schema=s_l), lp)
    pq.write_table(pa.table({"kid": [1], "c": ["x"]}, schema=s_r), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le, re_ = _etag(ld), _etag(rd)
    diff_server.put_object("/dl.parquet", ld, le)
    diff_server.put_object("/dr.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/dl.parquet"), le, len(ld)),
        right=SnapshotManifest(diff_server.url_for("/dr.parquet"), re_, len(rd)),
        key_column="kid",
        compare_columns=("c",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    with pytest.raises(SchemaDriftError):
        DiffEngine(diff_client_factory(), request).run()
    assert not (out / "f.jsonl").exists()


def test_service_log_records_conditional_headers_ranges_and_status(
    diff_client_factory, diff_snapshots, make_request, tmp_path, diff_server
):
    """The real server log is the evidence for the protocol contract."""
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    DiffEngine(client, request).run()
    gets = diff_server.range_gets()
    assert gets
    # Conditional header, precise interval, byte accounting and 206 status.
    for e in gets:
        assert e.status == 206
        assert e.if_match and e.if_match.startswith('"')
        assert e.range_header and e.range_header.startswith("bytes=")
        assert e.bytes_sent > 0
        assert e.content_range and e.content_range.startswith("bytes ")
        # Range header interval size equals bytes actually sent.
        spec = e.range_header[len("bytes="):]
        start_s, end_s = spec.split("-")
        assert int(end_s) - int(start_s) + 1 == e.bytes_sent
    # No whole-object fallback ever appears.
    all_gets = [e for e in diff_server.log if e.method == "GET"]
    assert all(e.status != 200 for e in all_gets)
    assert all(e.range_header not in (None, "bytes=0-") for e in all_gets)
