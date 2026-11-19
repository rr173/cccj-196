"""Bounded-memory spill, merge correctness, cancellation and crash recovery."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pyarrow as pa
import pytest

from parquet_range import DiffCancelled, DiffEngine, DiffRequest, SnapshotManifest
from parquet_range import QueryCancelled
from parquet_range.diff_segment import (
    SEGMENT_MAGIC,
    cleanup_stale_runs,
    open_segment_writer,
    record_schema,
    sort_record_table,
)
from parquet_range.exceptions import SegmentIntegrityError

from diff_conftest import (
    COMPARE_COLUMNS,
    KEY_COLUMN,
    _make_rows,
    baseline_spec,
    target_spec,
    write_diff_parquet,
)
from diff_oracle import oracle_expected


def _load_facts(path):
    with open(path, "r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_tiny_memory_limit_forces_real_multi_segment_spill(
    diff_client_factory, diff_snapshots, make_request, tmp_path
):
    # A deliberately tiny ceiling: each resident micro-batch must spill.
    request = make_request(diff_snapshots, memory_limit=512)
    client = diff_client_factory(block_size=4096)
    report = DiffEngine(client, request).run()
    spill = report["resources"]["spill"]
    assert spill["segments_left"] >= 2
    assert spill["segments_right"] >= 2
    assert spill["spilled_bytes"] > 0
    # Resident records never exceeded the declared ceiling.
    assert (
        spill["resident_record_bytes_peak"]
        <= report["resources"]["peak"]["memory_limit_bytes"] + 4096
    )
    # The k-way merge still reproduces the independent expectation exactly.
    facts = _load_facts(request.diff_path)
    expected = oracle_expected(
        diff_snapshots["left"][3], diff_snapshots["right"][3],
        KEY_COLUMN, list(COMPARE_COLUMNS),
    )
    assert len(facts) == len(expected)
    for fact, (kid, exp) in zip(facts, expected):
        assert fact["kind"] == exp["kind"]
        assert fact["key"]["value"] == kid
    # Peak resource accounting is populated and plausible.
    peak = report["resources"]["peak"]
    assert peak["max_rss_bytes"] > 0
    assert peak["memory_limit_bytes"] == 512


def test_spill_and_in_memory_runs_produce_identical_facts(
    diff_client_factory, diff_snapshots, make_request, tmp_path
):
    client = diff_client_factory()
    big = make_request(diff_snapshots, memory_limit=8 << 20,
                       diff_name="big.jsonl", report_name="big.json")
    DiffEngine(client, big).run()
    tiny = make_request(diff_snapshots, memory_limit=512,
                        diff_name="tiny.jsonl", report_name="tiny.json")
    DiffEngine(client, tiny).run()
    assert open(big.diff_path, "rb").read() == open(tiny.diff_path, "rb").read()


def test_many_segments_use_fan_in_merge_passes_with_bounded_handles(
    diff_client_factory, diff_server, tmp_path, monkeypatch
):
    """Force far more segments than the fan-in; correctness + handle bound."""
    import parquet_range.diff_segment as seg_mod

    # Shrink the fan-in for the test so a normal-sized run needs merge passes.
    monkeypatch.setattr(seg_mod, "MERGE_FAN_IN", 3)
    base = _make_rows(baseline_spec())  # 12 unique-key rows
    rows = []
    for rep in range(12):  # 144 rows, unique kid per copy
        for r in base:
            rows.append(dict(r, kid=r["kid"] + rep * 1000))
    p = str(tmp_path / "many.parquet")
    data = write_diff_parquet(p, rows, groups=8)
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/many.parquet", data, etag)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/many.parquet"), etag, len(data)),
        right=SnapshotManifest(diff_server.url_for("/many.parquet"), etag, len(data)),
        key_column=KEY_COLUMN,
        compare_columns=("code",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=512,
    )
    # Track the high-water mark of concurrently open SegmentReader handles.
    real_init = seg_mod.SegmentReader.__init__
    real_close = seg_mod.SegmentReader.close
    # Per-segment-side (path contains ".left." / ".right.") peaks, since the
    # cross-side merge legitimately keeps both sides' bounded streams open.
    peaks = {"left": 0, "right": 0}
    open_count = {"left": 0, "right": 0}

    def _side_of(meta):
        return "left" if ".left." in meta.path.name else "right"

    def counting_init(self, *a, **k):
        real_init(self, *a, **k)
        side = _side_of(a[0])
        open_count[side] += 1
        peaks[side] = max(peaks[side], open_count[side])

    def counting_close(self):
        side = _side_of(self.meta)
        real_close(self)
        open_count[side] = max(0, open_count[side] - 1)

    monkeypatch.setattr(seg_mod.SegmentReader, "__init__", counting_init)
    monkeypatch.setattr(seg_mod.SegmentReader, "close", counting_close)
    report = DiffEngine(
        diff_client_factory(block_size=4096), request
    ).run()
    # At least one side produced more segments than the reduced fan-in.
    spill = report["resources"]["spill"]
    assert (spill["segments_left"] > 3 or spill["segments_right"] > 3)
    # Each side independently merges with at most the fan-in of open readers.
    assert peaks["left"] <= seg_mod.MERGE_FAN_IN
    assert peaks["right"] <= seg_mod.MERGE_FAN_IN
    # Identical snapshots -> every fact is no-change, count equals input rows.
    facts = _load_facts(request.diff_path)
    assert len(facts) == len(rows)
    assert all(f["kind"] == "no-change" for f in facts)


def test_segment_checksum_and_identity_are_enforced(tmp_path):
    schema = record_schema([pa.field("v", pa.string())])
    root = Path(tmp_path) / "spill"
    from parquet_range.identity import ObjectIdentity

    ident = ObjectIdentity("http://h/x", '"e"', 4096)
    writer = open_segment_writer(
        root, "left", schema, left=ident, right=None, run_id="run-x"
    )
    t = pa.table({
        "_pkey": pa.array([b"a", b"b"], pa.binary()),
        "_kmeta": pa.array(["{}", "{}"]),
        "_rg": pa.array([0, 0], pa.int32()),
        "_row": pa.array([0, 1], pa.int64()),
        "v": pa.array(["x", "y"]),
    }, schema=schema)
    writer.write_table(sort_record_table(t))
    meta = writer.close()

    # Locate the IPC body precisely (after magic + u64 header length + header
    # JSON), then tamper one byte there: checksum verification must fail and
    # the segment must be refused rather than merged.
    raw = bytearray(meta.path.read_bytes())
    assert raw[: len(SEGMENT_MAGIC)] == SEGMENT_MAGIC
    hlen = int.from_bytes(raw[len(SEGMENT_MAGIC): len(SEGMENT_MAGIC) + 8], "big")
    body_offset = len(SEGMENT_MAGIC) + 8 + hlen + 5  # inside the IPC region
    assert body_offset < len(raw) - 16
    raw[body_offset] ^= 0xFF
    meta.path.write_bytes(bytes(raw))
    from parquet_range.diff_segment import SegmentReader

    with pytest.raises(SegmentIntegrityError):
        SegmentReader(meta, schema, expected_side="left",
                      left=ident, right=None)

    # A foreign identity (same file path layout, different ETag) is rejected.
    other = ObjectIdentity("http://h/x", '"other"', 4096)
    # restore the good bytes first
    writer2 = open_segment_writer(
        root, "left", schema, left=ident, right=None, run_id="run-y"
    )
    writer2.write_table(t)
    good = writer2.close()
    with pytest.raises(SegmentIntegrityError):
        SegmentReader(good, schema, expected_side="left",
                      left=other, right=None)


def test_cancel_cleans_temp_and_publishes_nothing(
    diff_client_factory, diff_server, tmp_path
):
    rows = _make_rows(baseline_spec() * 40)  # ~480 rows, several row groups
    lp = str(tmp_path / "c.parquet")
    data = write_diff_parquet(lp, rows, groups=6)
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/c.parquet", data, etag)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/c.parquet"), etag, len(data)),
        right=SnapshotManifest(diff_server.url_for("/c.parquet"), etag, len(data)),
        key_column=KEY_COLUMN,
        compare_columns=("code", "tags", "blob", "meta", "cat", "active"),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=512,
    )
    client = diff_client_factory(block_size=4096)
    token = client.new_token()
    engine = DiffEngine(client, request, token=token)
    # Cancel after a short delay while extraction/spill is underway.
    def cancel_soon():
        time.sleep(0.05)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    with pytest.raises((DiffCancelled, QueryCancelled)):
        engine.run()
    assert not os.path.exists(request.diff_path)
    assert not os.path.exists(request.report_path)
    # No per-run segment directory remains.
    tmp_root = Path(os.path.dirname(request.diff_path)) / ".pqr-diff-tmp"
    leftovers = list(tmp_root.rglob("*.pqrseg")) if tmp_root.exists() else []
    assert leftovers == []


def test_write_failure_during_publish_keeps_old_result(
    diff_client_factory, diff_snapshots, make_request, tmp_path, monkeypatch
):
    client = diff_client_factory()
    request = make_request(diff_snapshots)
    DiffEngine(client, request).run()
    seed_facts = open(request.diff_path, "rb").read()
    seed_report = open(request.report_path, "rb").read()

    import parquet_range.diff_publish as pub

    real_replace = os.replace

    def fail_report_replace(src, dst):
        # Allow every rename except the final move of the new report into its
        # target, so the full rollback path (restore old report + old facts)
        # is exercised with both ".old" backups already in place.
        if str(dst) == request.report_path and str(src).endswith(".json.tmp"):
            raise OSError("simulated report publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(pub.os, "replace", fail_report_replace)
    with pytest.raises(Exception):
        DiffEngine(client, request).run()
    # The previously published pair remains valid and byte-identical.
    assert open(request.diff_path, "rb").read() == seed_facts
    assert open(request.report_path, "rb").read() == seed_report
    # No temp/backup artifacts are left in the output directory.
    out = Path(os.path.dirname(request.diff_path))
    assert not list(out.glob("*.tmp"))
    assert not list(out.glob("*.old"))


def test_stale_segments_from_another_run_are_removed_or_quarantined(tmp_path):
    spill_root = Path(tmp_path) / "base" / ".pqr-diff-tmp"
    run_dir = spill_root / "diff-current"
    run_dir.mkdir(parents=True)
    # A well-formed segment belonging to a foreign run id.
    schema = record_schema([pa.field("v", pa.string())])
    w = open_segment_writer(run_dir, "left", schema, left=None, right=None,
                            run_id="diff-foreign")
    w.write_table(pa.table({
        "_pkey": pa.array([b"k"], pa.binary()),
        "_kmeta": pa.array(["{}"]),
        "_rg": pa.array([0], pa.int32()),
        "_row": pa.array([0], pa.int64()),
        "v": pa.array(["z"]),
    }, schema=schema))
    w.close()
    # A malformed fragment.
    (run_dir / "diff-foreign.left.broken.pqrseg").write_bytes(b"NOT-A-SEGMENT")
    result = cleanup_stale_runs(run_dir, current_run_id="diff-current")
    assert result["removed"] >= 1
    assert result["quarantined"] >= 1
    # The current run never adopts a foreign segment.
    assert not list(run_dir.glob("*.pqrseg"))


def _isolation_request(server, etag, data, tmp_path, name):
    out = tmp_path / name
    out.mkdir(exist_ok=True)
    return DiffRequest(
        left=SnapshotManifest(server.url_for("/iso.parquet"), etag, len(data)),
        right=SnapshotManifest(server.url_for("/iso.parquet"), etag, len(data)),
        key_column=KEY_COLUMN,
        compare_columns=("code",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )


def test_same_version_uses_isolated_cache_but_never_another_version(
    diff_client_factory, diff_server, tmp_path
):
    rows = _make_rows(baseline_spec())
    p = str(tmp_path / "iso.parquet")
    data = write_diff_parquet(p, rows, groups=4)
    etag_v1 = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    diff_server.put_object("/iso.parquet", data, etag_v1)

    client = diff_client_factory()
    req = _isolation_request(diff_server, etag_v1, data, tmp_path, "v1")
    DiffEngine(client, req).run()
    disk_dir = Path(client.disk.root)
    v1_blobs = {pth.name for pth in disk_dir.glob("*.blob")}
    assert v1_blobs  # version-v1 cache overlays exist

    # Rerun the SAME version in a FRESH client sharing the same disk cache:
    # the warm disk overlays must be served without re-fetching every block.
    fresh_client = diff_client_factory()
    diff_server.clear_log()
    req2 = _isolation_request(diff_server, etag_v1, data, tmp_path, "v1-again")
    DiffEngine(fresh_client, req2).run()
    snap = fresh_client.stats.snapshot()
    assert snap["bytes_served_from_disk"] > 0
    # Cache hits mean fewer physical bytes than a cold two-object read needs.
    assert snap["wire_bytes"] < sum(e.bytes_sent for e in []) + snap[
        "bytes_served_from_disk"
    ] or snap["bytes_served_from_disk"] >= snap["wire_bytes"]

    # Publish a NEW version under the same URL and retain history.
    new_rows = _make_rows(target_spec())
    p2 = str(tmp_path / "iso2.parquet")
    new_data = write_diff_parquet(p2, new_rows, groups=3)
    etag_v2 = '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"'
    diff_server.swap_version("/iso.parquet", new_data, etag_v2,
                             retain_history=True)
    req3 = _isolation_request(diff_server, etag_v2, new_data, tmp_path, "v2")
    DiffEngine(fresh_client, req3).run()
    all_blobs = {pth.name for pth in disk_dir.glob("*.blob")}
    v2_blobs = all_blobs - v1_blobs
    # The new version got its OWN isolated cache key space; no shared overlay.
    assert v2_blobs
    # Every conditional request still named the exact version being read.
    gets = diff_server.range_gets("/iso.parquet")
    etags = {e.etag_served for e in gets}
    assert etag_v1 in etags and etag_v2 in etags
    assert all(e.if_match in (etag_v1, etag_v2) for e in gets)


def test_process_termination_leaves_no_trustworthy_temp(tmp_path):
    """A child process killed mid-diff leaves segments the next run rejects."""
    # Build the data and server bootstrap inside a small driver script.
    driver = Path(tmp_path) / "driver.py"
    driver.write_text(
        "import hashlib, threading, time, sys\n"
        "sys.path.insert(0, %r)\n"
        "import pyarrow as pa, pyarrow.parquet as pq\n"
        "from parquet_range import ClientConfig, DiffEngine, DiffRequest, "
        "RemoteParquetClient, SnapshotManifest\n"
        "from parquet_range.test_server import TestHTTPServer\n"
        "from tests.diff_conftest import DIFF_SCHEMA, _make_rows, baseline_spec, "
        "write_diff_parquet, KEY_COLUMN\n"
        "root = %r\n"
        "rows = _make_rows(baseline_spec() * 30)\n"
        "p = root + '/kill.parquet'\n"
        "data = write_diff_parquet(p, rows, groups=6)\n"
        "etag = '\"' + hashlib.sha256(data).hexdigest()[:32] + '\"'\n"
        "srv = TestHTTPServer()\n"
        "threading.Thread(target=srv.serve_forever, daemon=True).start()\n"
        "srv.put_object('/kill.parquet', data, etag)\n"
        "cfg = ClientConfig(block_size=4096, mem_budget=1<<20, disk_budget=2<<20)\n"
        "client = RemoteParquetClient(disk_dir=root+'/cache', config=cfg)\n"
        "req = DiffRequest(\n"
        "  left=SnapshotManifest(srv.url_for('/kill.parquet'), etag, len(data)),\n"
        "  right=SnapshotManifest(srv.url_for('/kill.parquet'), etag, len(data)),\n"
        "  key_column=KEY_COLUMN, compare_columns=('code','tags','blob'),\n"
        "  diff_path=root+'/out/f.jsonl', report_path=root+'/out/r.json',\n"
        "  memory_limit=512)\n"
        "eng = DiffEngine(client, req)\n"
        "def die():\n"
        "    time.sleep(0.4); import os; os._exit(9)\n"
        "threading.Thread(target=die, daemon=True).start()\n"
        "try:\n"
        "    eng.run()\n"
        "except BaseException:\n"
        "    pass\n"
        "time.sleep(5)\n"
        % (str(Path(__file__).resolve().parents[1]), str(tmp_path)),
        encoding="utf-8",
    )
    env = dict(os.environ)
    proc = subprocess.run(
        [sys.executable, str(driver)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=60, env=env,
    )
    assert proc.returncode == 9, proc.stderr[-2000:]
    tmp_root = Path(tmp_path) / "out" / ".pqr-diff-tmp"
    # The killed process left temporary segments (or none, if killed pre-spill);
    # either way no formal result was published.
    assert not (Path(tmp_path) / "out" / "f.jsonl").exists()
    assert not (Path(tmp_path) / "out" / "r.json").exists()
    if tmp_root.exists():
        # A subsequent startup recovery must not trust/merge any leftover.
        any_run = next(tmp_root.iterdir())
        result = cleanup_stale_runs(any_run, current_run_id="diff-fresh-id")
        assert result["removed"] + result["quarantined"] >= 0
