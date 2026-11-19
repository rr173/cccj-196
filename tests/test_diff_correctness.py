"""Diff correctness vs an independent local-PyArrow oracle (real HTTP server).

Covers multiple row groups, reordered records, insert/delete/update/no-change,
nested list/struct, dictionary, null, empty list/struct, binary and the
determinism of the published ordering.
"""

from __future__ import annotations

import json

from parquet_range import DiffEngine, DiffRequest, SnapshotManifest, explain_diff
from parquet_range.diff_key import encode_scalar

from diff_conftest import COMPARE_COLUMNS, KEY_COLUMN
from diff_oracle import oracle_expected


def _load_facts(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _canon_hex(kid: int) -> str:
    import pyarrow as pa

    return encode_scalar(pa.scalar(kid, pa.int64())).hex()


def test_diff_matches_independent_oracle_line_by_line(
    diff_client_factory, diff_snapshots, make_request, tmp_path
):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    report = DiffEngine(client, request).run()

    facts = _load_facts(request.diff_path)
    expected = oracle_expected(
        diff_snapshots["left"][3],
        diff_snapshots["right"][3],
        KEY_COLUMN,
        list(COMPARE_COLUMNS),
    )
    # Same number of facts, same canonical ordering.
    assert len(facts) == len(expected)
    assert [bytes.fromhex(f["key"]["hex"]) for f in facts] == sorted(
        bytes.fromhex(f["key"]["hex"]) for f in facts
    )
    for fact, (kid, exp) in zip(facts, expected):
        assert fact["kind"] == exp["kind"]
        assert fact["key"]["hex"] == _canon_hex(kid)
        assert fact["key"]["value"] == kid
        # before/after presence matches the kind contract.
        if fact["kind"] == "insert":
            assert "before" not in fact and "after" in fact
            assert fact["after"] == exp["after"]
        elif fact["kind"] == "delete":
            assert "after" not in fact and "before" in fact
            assert fact["before"] == exp["before"]
        else:
            assert "before" in fact and "after" in fact
            assert fact["before"] == exp["before"]
            assert fact["after"] == exp["after"]
        # Fixed record field order.
        assert list(fact.keys())[0:2] == ["kind", "key"]

    # Report counts agree with the facts and are internally consistent.
    counts = report["counts"]
    actual = {k: 0 for k in ("insert", "delete", "update", "no-change")}
    for f in facts:
        actual[f["kind"]] += 1
    assert counts == actual
    assert report["status"] == "completed"
    assert report["input_rows"] == {
        "left": len(diff_snapshots["left_rows"]),
        "right": len(diff_snapshots["right_rows"]),
    }
    assert report["unique_keys"]["left"] == len(diff_snapshots["left_rows"])
    assert report["unique_keys"]["right"] == len(diff_snapshots["right_rows"])
    # Identities recorded exactly as pinned.
    assert report["left_identity"]["etag"] == diff_snapshots["left"][1]
    assert report["right_identity"]["etag"] == diff_snapshots["right"][1]


def test_diff_repeat_runs_are_byte_stable(
    diff_client_factory, diff_snapshots, make_request, tmp_path
):
    client = diff_client_factory()
    r1 = make_request(diff_snapshots, diff_name="a.jsonl",
                      report_name="a.json")
    DiffEngine(client, r1).run()
    r2 = make_request(diff_snapshots, diff_name="b.jsonl",
                      report_name="b.json")
    DiffEngine(client, r2).run()
    assert open(r1.diff_path, "rb").read() == open(r2.diff_path, "rb").read()


def test_explain_reads_artifacts_without_network(
    diff_client_factory, diff_snapshots, make_request, tmp_path, diff_server
):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    DiffEngine(client, request).run()
    diff_server.clear_log()
    explanation = explain_diff(request.report_path)
    assert explanation["facts_verified"] is True
    assert explanation["canonical_order_stable"] is True
    # No service contact happened during explain.
    assert diff_server.range_gets() == []
    assert all(e.method != "HEAD" for e in diff_server.log)


def test_diff_distinguishes_null_empty_container_dictionary_and_binary(
    diff_client_factory, diff_snapshots, make_request, tmp_path
):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    DiffEngine(client, request).run()
    facts = {f["key"]["value"]: f for f in _load_facts(request.diff_path)}
    # key 2: pick-0 row has null tags/meta/cat/blob and gets its code updated.
    upd = facts[2]
    assert upd["kind"] == "update"
    assert upd["before"]["tags"] is None and upd["after"]["tags"] is None
    # key 3: pick-1 row has EMPTY list/struct (struct flags=[]) distinct from null.
    nc = facts[3]
    assert nc["kind"] == "no-change"
    # The empty list keeps its full Arrow type tag (name may differ across the
    # Parquet round-trip: item -> element) and an explicit empty value list.
    assert nc["before"]["tags"]["value"] == []
    assert "list<" in nc["before"]["tags"]["type"]
    assert "string" in nc["before"]["tags"]["type"]
    assert nc["before"]["meta"]["value"]["flags"]["value"] == []
    assert "list<" in nc["before"]["meta"]["value"]["flags"]["type"]
    # null container and empty container are never treated as equal:
    # key 2 (null tags) vs key 3 (empty tags) must both appear distinctly.
    assert facts[2]["before"]["tags"] is None
    # dictionary-decoded value surfaces as a plain string.
    assert nc["before"]["cat"] == "alpha"
    # empty binary is hex-tagged with an empty payload (distinct from null).
    assert nc["before"]["blob"] == {"type": "binary", "value": ""}
    assert facts[2]["before"]["blob"] is None
    # key 4 (pick 3) carries real binary bytes, rendered as hex.
    assert facts[4]["before"]["blob"]["value"] == "00ff00ff"


def test_only_declared_columns_and_row_groups_are_fetched(
    diff_client_factory, diff_snapshots, make_request, tmp_path, diff_server
):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    DiffEngine(client, request).run()
    gets = diff_server.range_gets()
    # Every data GET is a precise 206 with a conditional header; never a 200.
    assert gets
    for entry in gets:
        assert entry.status == 206
        assert entry.if_match is not None and entry.if_match.startswith('"')
        assert entry.range_header and entry.range_header.startswith("bytes=")
        assert entry.range_header != "bytes=0-"

    # Column-subset proof on a deliberately WIDE object: the undeclared filler
    # columns dominate the file, so reading only the key+compare columns must
    # transfer a clear minority of the object bytes (real server log evidence).
    import hashlib
    import pyarrow as pa
    import pyarrow.parquet as pq

    wide_schema = pa.schema([
        pa.field("kid", pa.int64()),
        pa.field("code", pa.string()),
        *[pa.field(f"wide_{i}", pa.float64()) for i in range(16)],
    ])
    n = 4000
    import random

    rng = random.Random(12345)
    cols = {
        "kid": list(range(n)),
        "code": [f"c-{i}" for i in range(n)],
        # Incompressible pseudo-random doubles: snappy cannot shrink them, so
        # the 16 undeclared columns genuinely dominate the object's bytes.
        **{f"wide_{i}": [rng.random() * 1e9 for _ in range(n)]
           for i in range(16)},
    }
    wp = str(tmp_path / "wide.parquet")
    pq.write_table(pa.table(cols, schema=wide_schema), wp,
                   row_group_size=1000, compression="snappy")
    wdata = open(wp, "rb").read()
    wetag = '"' + hashlib.sha256(wdata).hexdigest()[:32] + '"'
    diff_server.put_object("/wide.parquet", wdata, wetag)
    diff_server.clear_log()
    # Fine-grained blocks so sparse column selection is measured precisely.
    wide_client = diff_client_factory(block_size=8192)
    wreq = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/wide.parquet"), wetag, len(wdata)),
        right=SnapshotManifest(diff_server.url_for("/wide.parquet"), wetag, len(wdata)),
        key_column="kid",
        compare_columns=("code",),
        diff_path=str(tmp_path / "out" / "wfacts.jsonl"),
        report_path=str(tmp_path / "out" / "wreport.json"),
        memory_limit=4 << 20,
    )
    DiffEngine(wide_client, wreq).run()
    wgets = diff_server.range_gets("/wide.parquet")
    assert all(e.status == 206 and e.if_match == wetag for e in wgets)
    wire = sum(e.bytes_sent for e in wgets)
    # 16 undeclared incompressible wide columns dwarf the 2 narrow selected ones.
    assert wire < len(wdata) // 3, (wire, len(wdata))
    # Conditional HEADs were performed for both identities.
    heads = [e for e in diff_server.log if e.method == "HEAD"]
    assert len(heads) >= 2
    assert all(e.if_match == wetag for e in heads)
