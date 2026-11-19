"""End-to-end CLI tests for ``diff`` and ``diff-explain`` subcommands.

These shell out to the real ``bin/pqr.py`` entry point so the argparse layer,
exit codes and JSON request-file handling are covered, not just the library.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CLI = str(Path(__file__).resolve().parents[1] / "bin" / "pqr.py")


def _run_cli(args, cwd):
    return subprocess.run(
        [sys.executable, CLI, *args],
        cwd=cwd, capture_output=True, text=True, timeout=120,
    )


def _write_pair(server, tmp_path, name, rows_l, rows_r, schema):
    lp = str(tmp_path / f"{name}_l.parquet")
    rp = str(tmp_path / f"{name}_r.parquet")
    pq.write_table(pa.table(rows_l, schema=schema), lp)
    pq.write_table(pa.table(rows_r, schema=schema), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le = '"' + hashlib.sha256(ld).hexdigest()[:32] + '"'
    re_ = '"' + hashlib.sha256(rd).hexdigest()[:32] + '"'
    server.put_object(f"/{name}_l.parquet", ld, le)
    server.put_object(f"/{name}_r.parquet", rd, re_)
    return le, re_, len(ld), len(rd)


def test_cli_diff_via_request_file_and_explain(diff_server, tmp_path):
    schema = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.string())])
    le, re_, llen, rlen = _write_pair(
        diff_server, tmp_path, "cli",
        {"kid": [1, 2], "v": ["a", "b"]},
        {"kid": [2, 3], "v": ["b", "c"]},
        schema,
    )
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = {
        "left": {"url": diff_server.url_for("/cli_l.parquet"), "etag": le,
                 "length": llen},
        "right": {"url": diff_server.url_for("/cli_r.parquet"), "etag": re_,
                  "length": rlen},
        "key_column": "kid",
        "compare_columns": ["v"],
        "diff_path": str(out / "f.jsonl"),
        "report_path": str(out / "r.json"),
        "memory_limit": 1 << 20,
    }
    req_path = tmp_path / "request.json"
    req_path.write_text(json.dumps(request), encoding="utf-8")
    proc = _run_cli(
        ["--cache-dir", str(tmp_path / "cache"), "diff", str(req_path)],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(open(out / "r.json").read())
    assert report["counts"] == {"insert": 1, "delete": 1, "update": 0,
                                "no-change": 1}
    # diff-explain reads only the local artifacts.
    diff_server.clear_log()
    expl = _run_cli(
        ["diff-explain", str(out / "r.json")],
        cwd=str(Path(__file__).resolve().parents[1]),
    )
    assert expl.returncode == 0, expl.stderr
    payload = json.loads(expl.stdout)
    assert payload["facts_verified"] is True
    assert diff_server.range_gets() == []


def test_cli_diff_via_flags(diff_server, tmp_path):
    schema = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.string())])
    le, re_, llen, rlen = _write_pair(
        diff_server, tmp_path, "flag",
        {"kid": [1], "v": ["a"]},
        {"kid": [1], "v": ["a"]},
        schema,
    )
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    proc = _run_cli([
        "--cache-dir", str(tmp_path / "cache"), "diff",
        "--left-url", diff_server.url_for("/flag_l.parquet"),
        "--left-etag", le, "--left-length", str(llen),
        "--right-url", diff_server.url_for("/flag_r.parquet"),
        "--right-etag", re_, "--right-length", str(rlen),
        "--key-column", "kid", "--compare-columns", "v",
        "--diff-out", str(out / "f.jsonl"),
        "--report-out", str(out / "r.json"),
        "--diff-memory-limit", str(1 << 20),
    ], cwd=str(Path(__file__).resolve().parents[1]))
    assert proc.returncode == 0, proc.stderr
    report = json.loads(open(out / "r.json").read())
    assert report["counts"]["no-change"] == 1


def test_cli_bad_identity_exit_code_and_no_output(diff_server, tmp_path):
    schema = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.string())])
    le, re_, llen, rlen = _write_pair(
        diff_server, tmp_path, "bad",
        {"kid": [1], "v": ["a"]}, {"kid": [1], "v": ["a"]}, schema,
    )
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    proc = _run_cli([
        "--cache-dir", str(tmp_path / "cache"), "diff",
        "--left-url", diff_server.url_for("/bad_l.parquet"),
        "--left-etag", '"00000000000000000000000000000000"',
        "--left-length", str(llen),
        "--right-url", diff_server.url_for("/bad_r.parquet"),
        "--right-etag", re_, "--right-length", str(rlen),
        "--key-column", "kid", "--compare-columns", "v",
        "--diff-out", str(out / "f.jsonl"),
        "--report-out", str(out / "r.json"),
    ], cwd=str(Path(__file__).resolve().parents[1]))
    # Identity/version failure maps to the distinct diff exit code (2/3/4...),
    # never 0, and leaves no formal artifacts.
    assert proc.returncode != 0
    assert not (out / "f.jsonl").exists()
    assert not (out / "r.json").exists()
    assert "VersionMismatchError" in proc.stderr


def test_cli_drift_exit_code(diff_server, tmp_path):
    s_l = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.string())])
    s_r = pa.schema([pa.field("kid", pa.int64()), pa.field("v", pa.int32())])
    lp, rp = str(tmp_path / "dl.parquet"), str(tmp_path / "dr.parquet")
    pq.write_table(pa.table({"kid": [1], "v": ["x"]}, schema=s_l), lp)
    pq.write_table(pa.table({"kid": [1], "v": [1]}, schema=s_r), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le = '"' + hashlib.sha256(ld).hexdigest()[:32] + '"'
    re_ = '"' + hashlib.sha256(rd).hexdigest()[:32] + '"'
    diff_server.put_object("/dl.parquet", ld, le)
    diff_server.put_object("/dr.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    proc = _run_cli([
        "--cache-dir", str(tmp_path / "cache"), "diff",
        "--left-url", diff_server.url_for("/dl.parquet"),
        "--left-etag", le, "--left-length", str(len(ld)),
        "--right-url", diff_server.url_for("/dr.parquet"),
        "--right-etag", re_, "--right-length", str(len(rd)),
        "--key-column", "kid", "--compare-columns", "v",
        "--diff-out", str(out / "f.jsonl"),
        "--report-out", str(out / "r.json"),
    ], cwd=str(Path(__file__).resolve().parents[1]))
    assert proc.returncode == 4  # SchemaDriftError family
    assert "SchemaDriftError" in proc.stderr
    assert not (out / "f.jsonl").exists()
