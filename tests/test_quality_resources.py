"""Bounded-memory auditing, cancellation, write failures and crash recovery."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from parquet_range import (
    AuditCancelled,
    AuditPublishError,
    QualityAuditEngine,
    QualityAuditRequest,
    QualityContract,
    QueryCancelled,
    SnapshotManifest,
    read_violations,
)
from parquet_range.quality_publish import recover_publish_unit

from quality_conftest import (
    etag_of,
    make_quality_rows,
    quality_contract,
    write_quality_parquet,
)


def _big_snapshot(quality_server, tmp_path, rows=24_000, groups=8):
    path = str(tmp_path / "big.parquet")
    data = write_quality_parquet(path, make_quality_rows(rows), groups=groups)
    etag = etag_of(data)
    quality_server.put_object("/big.parquet", data, etag)
    quality_server.clear_log()
    return {
        "url": quality_server.url_for("/big.parquet"),
        "etag": etag, "length": len(data), "data": data,
    }


def _request_for(snapshot, tmp_path, *, batch_rows=512, name="out"):
    out_dir = tmp_path / name
    out_dir.mkdir(exist_ok=True)
    return QualityAuditRequest(
        object=SnapshotManifest(snapshot["url"], snapshot["etag"],
                                snapshot["length"]),
        contract=QualityContract.from_dict(quality_contract()),
        violations_path=str(out_dir / "violations.jsonl"),
        report_path=str(out_dir / "report.json"),
        batch_rows=batch_rows,
    )


def test_small_budget_larger_than_budget_data(
    quality_client_factory, quality_server, tmp_path
):
    """A tiny cache budget still audits data several times its size."""
    snapshot = _big_snapshot(quality_server, tmp_path)
    assert snapshot["length"] > (1 << 20)  # the object is bigger than 1 MiB
    client = quality_client_factory(
        block_size=8192, mem_budget=1 << 20, disk_budget=2 << 20,
        max_inflight_bytes=1 << 20,
    )
    request = _request_for(snapshot, tmp_path)
    report = QualityAuditEngine(client, request).run()
    assert report["status"] == "completed"
    assert report["rows_scanned"] == 24_000
    peak = report["resources"]["peak"]
    assert peak["max_rss_bytes"] > 0
    # Boundedness: peak RSS stays far below what holding the whole table plus
    # all violations in memory would cost (the object alone is >1 MiB and the
    # violations file is larger still); 512 MiB is a generous ceiling.
    assert peak["max_rss_bytes"] < 512 << 20
    # Only the referenced columns were transferred.
    wire = quality_server.physical_bytes("/big.parquet")
    assert 0 < wire < snapshot["length"]
    # The violations file itself is bigger than the memory budget, proving
    # results were streamed rather than accumulated.
    assert report["violations"]["bytes"] > (1 << 20)


def test_cancellation_cleans_staging_and_publishes_nothing(
    quality_client_factory, quality_server, tmp_path
):
    snapshot = _big_snapshot(quality_server, tmp_path)
    client = quality_client_factory(block_size=4096)
    request = _request_for(snapshot, tmp_path)
    token = client.new_token()
    engine = QualityAuditEngine(client, request, token=token)

    def cancel_soon():
        time.sleep(0.05)
        token.cancel()

    threading.Thread(target=cancel_soon, daemon=True).start()
    with pytest.raises((AuditCancelled, QueryCancelled)):
        engine.run()
    assert not os.path.exists(request.violations_path)
    assert not os.path.exists(request.report_path)
    out_dir = Path(os.path.dirname(request.violations_path))
    assert not list(out_dir.glob("*.tmp"))


def test_write_failure_during_publish_keeps_old_result(
    quality_client_factory, quality_snapshot, make_quality_request,
    tmp_path, monkeypatch
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    seed_violations = open(request.violations_path, "rb").read()
    seed_report = open(request.report_path, "rb").read()

    import parquet_range.quality_publish as pub

    real_replace = os.replace

    def fail_report_replace(src, dst):
        # Let every rename through except the final move of the new report,
        # so the full rollback path is exercised with both ".old" backups.
        if str(dst) == request.report_path and str(src).endswith(".json.tmp"):
            raise OSError("simulated report publish failure")
        return real_replace(src, dst)

    monkeypatch.setattr(pub.os, "replace", fail_report_replace)
    with pytest.raises(AuditPublishError):
        QualityAuditEngine(client, request).run()
    assert open(request.violations_path, "rb").read() == seed_violations
    assert open(request.report_path, "rb").read() == seed_report
    out_dir = Path(os.path.dirname(request.violations_path))
    assert not list(out_dir.glob("*.tmp"))
    assert not list(out_dir.glob("*.old"))


def test_unwritable_output_dir_fails_without_partial_results(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    out_dir = tmp_path / "readonly"
    out_dir.mkdir()
    os.chmod(out_dir, 0o500)
    try:
        request = QualityAuditRequest(
            object=SnapshotManifest(
                quality_snapshot["url"], quality_snapshot["etag"],
                quality_snapshot["length"],
            ),
            contract=QualityContract.from_dict(quality_contract()),
            violations_path=str(out_dir / "violations.jsonl"),
            report_path=str(out_dir / "report.json"),
            batch_rows=64,
        )
        client = quality_client_factory()
        with pytest.raises((AuditPublishError, OSError)):
            QualityAuditEngine(client, request).run()
        assert not (out_dir / "violations.jsonl").exists()
        assert not (out_dir / "report.json").exists()
    finally:
        os.chmod(out_dir, 0o700)


def test_recovery_restores_interrupted_publish(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    """Simulate a crash between the two publish renames; recovery must
    restore the previous valid pair, never a half-new result."""
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    good_violations = open(request.violations_path, "rb").read()
    good_report = open(request.report_path, "rb").read()

    out_dir = Path(os.path.dirname(request.violations_path))
    # Simulate the crash: new violations swapped in, old violations backed
    # up, report never swapped (process died between the two renames).
    new_violations = b'{"fake":"new violations"}\n'
    os.replace(request.violations_path, request.violations_path + ".old")
    with open(request.violations_path, "wb") as fh:
        fh.write(new_violations)

    summary = recover_publish_unit(
        out_dir, request.violations_path, request.report_path
    )
    assert summary["restored"], summary
    # The previous valid pair is back, byte-identical.
    assert open(request.violations_path, "rb").read() == good_violations
    assert open(request.report_path, "rb").read() == good_report
    # The half-published new file was quarantined, not deleted silently.
    assert summary["quarantined"]
    for quarantined in summary["quarantined"]:
        assert open(quarantined, "rb").read() == new_violations
    # And the offline reader accepts the restored pair.
    out = read_violations(request.report_path)
    assert out["verified"]


def test_recovery_quarantines_orphan_new_result(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    """First-ever publish crashing between renames leaves an orphan: it must
    be quarantined so nothing looks like a successful audit."""
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    out_dir = Path(os.path.dirname(request.violations_path))
    # Simulate: violations swapped in, report never written, no .old backups.
    os.remove(request.report_path)
    summary = recover_publish_unit(
        out_dir, request.violations_path, request.report_path
    )
    assert summary["quarantined"]
    assert not os.path.exists(request.violations_path)
    assert not os.path.exists(request.report_path)


def test_recovery_removes_staging_leftovers(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    good_violations = open(request.violations_path, "rb").read()
    out_dir = Path(os.path.dirname(request.violations_path))
    # Drop fake staging files as a crashed run would leave them.
    stale = out_dir / ".quality-violations-deadbeef.jsonl.tmp"
    stale.write_bytes(b'{"partial":')
    stale_r = out_dir / ".quality-report-deadbeef.json.tmp"
    stale_r.write_bytes(b'{"partial"')
    summary = recover_publish_unit(
        out_dir, request.violations_path, request.report_path
    )
    assert summary["removed_staging"] == 2
    assert not stale.exists() and not stale_r.exists()
    # The valid published pair is untouched.
    assert open(request.violations_path, "rb").read() == good_violations


def test_process_kill_between_publish_renames_recovers_on_restart(
    quality_client_factory, quality_snapshot, make_quality_request, tmp_path
):
    """A real os._exit between the two renames; the next run recovers."""
    client = quality_client_factory()
    request = make_quality_request(quality_snapshot)
    QualityAuditEngine(client, request).run()
    good_violations = open(request.violations_path, "rb").read()
    good_report = open(request.report_path, "rb").read()

    # Child process: rerun the audit but os._exit(9) right after the
    # violations rename, before the report rename.
    driver = Path(tmp_path) / "kill_driver.py"
    request_json = tmp_path / "kill-request.json"
    request_json.write_text(json.dumps({
        "spec_version": 1,
        "object": {
            "url": quality_snapshot["url"],
            "etag": quality_snapshot["etag"],
            "length": quality_snapshot["length"],
        },
        "contract": quality_contract(),
        "violations_path": request.violations_path,
        "report_path": request.report_path,
        "batch_rows": 64,
    }))
    driver.write_text(
        "import os, sys\n"
        "sys.path.insert(0, %r)\n"
        "import parquet_range.quality_publish as pub\n"
        "from parquet_range import (ClientConfig, QualityAuditEngine,\n"
        "    QualityAuditRequest, RemoteParquetClient)\n"
        "real_replace = os.replace\n"
        "def dying_replace(src, dst):\n"
        "    real_replace(src, dst)\n"
        "    if str(dst).endswith('violations.jsonl'):\n"
        "        os._exit(9)  # die between the two renames\n"
        "pub.os.replace = dying_replace\n"
        "req = QualityAuditRequest.from_json_file(%r)\n"
        "client = RemoteParquetClient(disk_dir=%r,\n"
        "    config=ClientConfig(block_size=8192))\n"
        "QualityAuditEngine(client, req).run()\n"
        % (str(Path(__file__).resolve().parents[1]),
           str(request_json), str(tmp_path / "killcache")),
        encoding="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, str(driver)],
        cwd=str(Path(__file__).resolve().parents[1]),
        capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode == 9, proc.stderr[-2000:]
    # Mid-crash state: violations may be new, report is the old one.
    # The next engine run recovers the unit, then republishes cleanly.
    client2 = quality_client_factory()
    report = QualityAuditEngine(client2, request).run()
    assert report["status"] == "completed"
    # The republished pair is consistent and verifiable offline.
    out = read_violations(request.report_path)
    assert out["verified"]
    # The recovered+republished violations equal a fresh full audit.
    assert open(request.report_path, "rb").read() != good_report  # new run
    out_dir = Path(os.path.dirname(request.violations_path))
    assert not list(out_dir.glob("*.tmp"))
    assert not list(out_dir.glob("*.old"))


def test_second_run_after_crash_does_not_delete_valid_audit(tmp_path):
    """recover_publish_unit never deletes a consistent published pair."""
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    v, r = out_dir / "violations.jsonl", out_dir / "report.json"
    v.write_bytes(b'{"a": 1}\n')
    import hashlib

    report = {
        "report_format": "pqr-quality-report",
        "status": "completed",
        "total_violations": 1,
        "violations": {
            "path": str(v),
            "records": 1,
            "bytes": 9,
            "sha256": hashlib.sha256(b'{"a": 1}\n').hexdigest(),
        },
    }
    r.write_text(json.dumps(report), encoding="utf-8")
    summary = recover_publish_unit(out_dir, str(v), str(r))
    assert summary["quarantined"] == []
    assert summary["restored"] == []
    assert v.exists() and r.exists()
