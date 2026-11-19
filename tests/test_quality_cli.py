"""CLI acceptance for quality-audit / quality-explain: real exit statuses."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from quality_conftest import (
    etag_of,
    make_quality_rows,
    quality_contract,
    write_quality_parquet,
)

PQR = str(Path(__file__).resolve().parents[1] / "bin" / "pqr.py")


def _run_cli(args, **kwargs):
    return subprocess.run(
        [sys.executable, PQR, *args],
        capture_output=True, text=True, timeout=180, **kwargs,
    )


@pytest.fixture()
def cli_snapshot(quality_server, tmp_path):
    path = str(tmp_path / "cli.parquet")
    data = write_quality_parquet(path, make_quality_rows(111), groups=3)
    etag = etag_of(data)
    quality_server.put_object("/cli.parquet", data, etag)
    quality_server.clear_log()
    quality_server.fault.reset()
    yield {
        "url": quality_server.url_for("/cli.parquet"),
        "etag": etag,
        "length": len(data),
        "data": data,
    }
    quality_server.fault.reset()


def _write_request(tmp_path, snapshot, *, contract=None, name="req.json"):
    out_dir = tmp_path / "cli-out"
    out_dir.mkdir(exist_ok=True)
    request = {
        "spec_version": 1,
        "object": {
            "url": snapshot["url"],
            "etag": snapshot["etag"],
            "length": snapshot["length"],
        },
        "contract": contract or quality_contract(),
        "violations_path": str(out_dir / "violations.jsonl"),
        "report_path": str(out_dir / "report.json"),
        "batch_rows": 32,
    }
    path = tmp_path / name
    path.write_text(json.dumps(request), encoding="utf-8")
    return path, out_dir


def test_cli_audit_success_exit0_and_server_log(
    quality_server, cli_snapshot, tmp_path
):
    req_path, out_dir = _write_request(tmp_path, cli_snapshot)
    proc = _run_cli(["quality-audit", str(req_path)])
    assert proc.returncode == 0, proc.stderr
    report = json.loads((out_dir / "report.json").read_text())
    assert report["status"] == "completed"
    assert report["total_violations"] > 0
    # Server log: every data GET was conditional on the pinned ETag and 206.
    gets = quality_server.range_gets("/cli.parquet")
    assert gets, "expected real Range GETs"
    assert all(e.if_match == cli_snapshot["etag"] for e in gets)
    assert all(e.status == 206 for e in gets)
    assert all(e.etag_served == cli_snapshot["etag"] for e in gets)
    # Conditional HEAD also carried the pinned ETag.
    heads = [e for e in quality_server.log if e.method == "HEAD"]
    assert heads and all(e.if_match == cli_snapshot["etag"] for e in heads)


def test_cli_audit_with_flag_overrides(quality_server, cli_snapshot, tmp_path):
    contract_path = tmp_path / "contract.json"
    contract_path.write_text(json.dumps(quality_contract()), encoding="utf-8")
    out_dir = tmp_path / "flags-out"
    out_dir.mkdir(exist_ok=True)
    proc = _run_cli([
        "quality-audit",
        "--object-url", cli_snapshot["url"],
        "--object-etag", cli_snapshot["etag"],
        "--object-length", str(cli_snapshot["length"]),
        "--contract-file", str(contract_path),
        "--violations-out", str(out_dir / "v.jsonl"),
        "--report-out", str(out_dir / "r.json"),
        "--audit-batch-rows", "32",
    ])
    assert proc.returncode == 0, proc.stderr
    assert (out_dir / "v.jsonl").exists()
    assert (out_dir / "r.json").exists()


def test_cli_bad_contract_exit3(quality_server, cli_snapshot, tmp_path):
    contract = quality_contract()
    contract["rules"].append(
        {"id": "bad", "kind": "pattern", "field": "name", "pattern": "(?=x)y"}
    )
    req_path, _ = _write_request(tmp_path, cli_snapshot, contract=contract)
    proc = _run_cli(["quality-audit", str(req_path)])
    assert proc.returncode == 3, (proc.returncode, proc.stderr)
    assert "unsupported pattern" in proc.stderr


def test_cli_schema_mismatch_exit4(quality_server, cli_snapshot, tmp_path):
    contract = quality_contract()
    contract["fields"]["score"] = "float32"  # file has float64
    req_path, _ = _write_request(tmp_path, cli_snapshot, contract=contract)
    proc = _run_cli(["quality-audit", str(req_path)])
    assert proc.returncode == 4, (proc.returncode, proc.stderr)
    assert "type mismatch" in proc.stderr


def test_cli_version_mismatch_exit2(quality_server, cli_snapshot, tmp_path):
    req_path, out_dir = _write_request(tmp_path, cli_snapshot)
    # Swap the object so the pinned identity no longer matches.
    new_data = cli_snapshot["data"] + b"\x00"
    quality_server.swap_version(
        "/cli.parquet", new_data, etag_of(new_data)
    )
    proc = _run_cli(["quality-audit", str(req_path)])
    assert proc.returncode == 2, (proc.returncode, proc.stderr)
    assert not (out_dir / "violations.jsonl").exists()
    assert not (out_dir / "report.json").exists()


def test_cli_explain_offline(quality_server, cli_snapshot, tmp_path):
    req_path, out_dir = _write_request(tmp_path, cli_snapshot)
    proc = _run_cli(["quality-audit", str(req_path)])
    assert proc.returncode == 0, proc.stderr
    quality_server.clear_log()
    proc = _run_cli([
        "quality-explain", str(out_dir / "report.json"),
        "--rule", "score-range", "--offset", "0", "--limit", "3",
    ])
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["verified"] is True
    assert payload["page_info"]["rule_id"] == "score-range"
    assert len(payload["page"]) == 3
    assert all(r["rule_id"] == "score-range" for r in payload["page"])
    # quality-explain performed no network I/O at all.
    assert quality_server.log == []


def test_cli_explain_rejects_tampered_report(
    quality_server, cli_snapshot, tmp_path
):
    req_path, out_dir = _write_request(tmp_path, cli_snapshot)
    proc = _run_cli(["quality-audit", str(req_path)])
    assert proc.returncode == 0, proc.stderr
    # Corrupt one violations line: the offline verifier must refuse.
    vpath = out_dir / "violations.jsonl"
    content = vpath.read_bytes()
    vpath.write_bytes(content[:-2] + b"XX")
    proc = _run_cli(["quality-explain", str(out_dir / "report.json")])
    assert proc.returncode == 2, (proc.returncode, proc.stderr)


def test_cli_existing_commands_still_work(quality_server, cli_snapshot):
    """inspect/scan/cache-clear keep their existing behavior."""
    url = cli_snapshot["url"]
    proc = _run_cli(["inspect", url])
    assert proc.returncode == 0, proc.stderr
    info = json.loads(proc.stdout)
    assert info["etag"] == cli_snapshot["etag"]
    proc = _run_cli([
        "--block-size", "8192", "scan", url,
        "--columns", "id,score", "--row-groups", "0", "--format", "json",
    ])
    assert proc.returncode == 0, proc.stderr
    rows = json.loads(proc.stdout)
    assert rows and set(rows[0]) == {"id", "score"}
    proc = _run_cli(["cache-clear", url])
    assert proc.returncode == 0, proc.stderr
