#!/usr/bin/env python3
"""Command-line interface for the sparse remote Parquet reader.

Subcommands:
    inspect      URL                       show pinned identity + Parquet metadata
    scan         URL --columns a,b --row-groups 0,2 [--format json|arrow|csv]
    cache-clear  [URL]                     drop unpinned cached overlays
    diff         REQUEST.json              run a version audit between two snapshots
    diff-explain REPORT.json [--diff F]    explain an already published diff offline
    stats                                 show resource statistics of a run
    serve        DIR --port PORT          run the real Range test HTTP server
    generate     OUT.parquet [options]    build a large nested/dict/nulls file
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from parquet_range import (  # noqa: E402
    ClientConfig,
    DiffEngine,
    DiffRequest,
    ParquetRangeError,
    RemoteParquetClient,
    explain_diff,
)
from parquet_range.test_server import TestHTTPServer  # noqa: E402


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip() != ""]


def _parse_str_list(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip() != ""]


def cmd_inspect(args: argparse.Namespace) -> int:
    with RemoteParquetClient(disk_dir=args.cache_dir, config=_config(args)) as client:
        obj = client.open(args.url)
        info = obj.inspect()
        payload = info.to_dict()
        payload["physical_requests_so_far"] = client.stats.snapshot()
        print(json.dumps(payload, indent=2, default=str))
    return 0


def cmd_scan(args: argparse.Namespace) -> int:
    columns = _parse_str_list(args.columns) if args.columns else []
    row_groups = _parse_int_list(args.row_groups)
    with RemoteParquetClient(disk_dir=args.cache_dir, config=_config(args)) as client:
        obj = client.open(args.url)
        table = obj.scan(columns=columns, row_groups=row_groups)
        if args.format == "json":
            print(json.dumps(table.to_pylist(), indent=2, default=str))
        elif args.format == "csv":
            import pyarrow.csv as pacsv

            pacsv.write_csv(table, sys.stdout.buffer)
        else:
            print(table)
        print("\n---", file=sys.stderr)
        print(client.stats.render(), file=sys.stderr)
        print(
            f"rows={table.num_rows} columns={table.column_names} "
            f"version={obj.version_id[:12]}",
            file=sys.stderr,
        )
    return 0


def cmd_cache_clear(args: argparse.Namespace) -> int:
    with RemoteParquetClient(disk_dir=args.cache_dir, config=_config(args)) as client:
        freed = client.cache_clear(args.url)
        print(json.dumps(freed, indent=2))
    return 0


def _build_diff_request(args: argparse.Namespace) -> DiffRequest:
    if args.request:
        request = DiffRequest.from_json_file(args.request)
        raw = request.as_json()
    else:
        raw = {"spec_version": 1}
    # Explicit CLI flags override (or fully supply) the request document.
    if args.left_url:
        raw["left"] = {
            "url": args.left_url,
            "etag": args.left_etag,
            "length": args.left_length,
        }
    if args.right_url:
        raw["right"] = {
            "url": args.right_url,
            "etag": args.right_etag,
            "length": args.right_length,
        }
    if args.key_column:
        raw["key_column"] = args.key_column
    if args.compare_columns:
        raw["compare_columns"] = _parse_str_list(args.compare_columns)
    if args.diff_out:
        raw["diff_path"] = args.diff_out
    if args.report_out:
        raw["report_path"] = args.report_out
    if args.diff_memory_limit is not None:
        raw["memory_limit"] = args.diff_memory_limit
    return DiffRequest.from_dict(raw)


def cmd_diff(args: argparse.Namespace) -> int:
    request = _build_diff_request(args)
    with RemoteParquetClient(disk_dir=args.cache_dir, config=_config(args)) as client:
        engine = DiffEngine(client, request, spill_dir=args.spill_dir)
        report = engine.run()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print(
        f"\ndiff published: {request.diff_path}\nreport: {request.report_path}",
        file=sys.stderr,
    )
    return 0


def cmd_diff_explain(args: argparse.Namespace) -> int:
    explanation = explain_diff(args.report, args.diff)
    print(json.dumps(explanation, indent=2, ensure_ascii=False))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    import hashlib

    server = TestHTTPServer(("127.0.0.1", args.port))
    for name in sorted(os.listdir(args.directory)):
        full = os.path.join(args.directory, name)
        if os.path.isfile(full):
            data = open(full, "rb").read()
            etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
            server.put_object("/" + name, data, etag)
            print(f"serving /{name} ({len(data)} bytes, {etag})")
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(f"Range server listening on {server.base_url()} (Ctrl-C to stop)")
    try:
        thread.join()
    except KeyboardInterrupt:
        server.shutdown()
    return 0


def cmd_generate(args: argparse.Namespace) -> int:
    # Local import keeps CLI startup cheap for non-generate subcommands.
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from generate_large_parquet import build_file

    build_file(
        args.out,
        rows_per_group=args.rows_per_group,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    size = os.path.getsize(args.out)
    print(f"wrote {args.out}: {size} bytes, {args.num_groups} row groups")
    return 0


def _config(args: argparse.Namespace) -> ClientConfig:
    return ClientConfig(
        block_size=args.block_size,
        mem_budget=args.mem_budget,
        disk_budget=args.disk_budget,
        max_inflight_bytes=args.inflight_budget,
        max_inflight_requests=args.inflight_requests,
        max_retries=args.max_retries,
    )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sparse HTTP Range Parquet reader")
    p.add_argument("--cache-dir", default=".parquet_range_cache")
    p.add_argument("--block-size", type=int, default=1 << 20)
    p.add_argument("--mem-budget", type=int, default=64 << 20)
    p.add_argument("--disk-budget", type=int, default=512 << 20)
    p.add_argument("--inflight-budget", type=int, default=16 << 20)
    p.add_argument("--inflight-requests", type=int, default=8)
    p.add_argument("--max-retries", type=int, default=3)

    sub = p.add_subparsers(dest="command", required=True)

    sp = sub.add_parser("inspect", help="validate object and print Parquet metadata")
    sp.add_argument("url")
    sp.set_defaults(func=cmd_inspect)

    sp = sub.add_parser("scan", help="scan selected columns/row groups")
    sp.add_argument("url")
    sp.add_argument("--columns", default="", help="comma-separated top-level columns")
    sp.add_argument("--row-groups", required=True, help="comma-separated row group ids")
    sp.add_argument("--format", choices=["arrow", "json", "csv"], default="arrow")
    sp.set_defaults(func=cmd_scan)

    sp = sub.add_parser("cache-clear", help="clear unpinned cache overlays")
    sp.add_argument("url", nargs="?")
    sp.set_defaults(func=cmd_cache_clear)

    dp = sub.add_parser(
        "diff",
        help="version audit between two pinned snapshots (JSONL facts + report)",
    )
    dp.add_argument(
        "request",
        nargs="?",
        help="JSON request file; every field can be overridden by the flags below",
    )
    dp.add_argument("--left-url")
    dp.add_argument("--left-etag")
    dp.add_argument("--left-length", type=int)
    dp.add_argument("--right-url")
    dp.add_argument("--right-etag")
    dp.add_argument("--right-length", type=int)
    dp.add_argument("--key-column")
    dp.add_argument("--compare-columns", help="comma-separated top-level columns")
    dp.add_argument("--diff-out", help="final JSONL facts path")
    dp.add_argument("--report-out", help="final JSON audit report path")
    dp.add_argument(
        "--diff-memory-limit",
        type=int,
        help="per-side resident record byte ceiling (forces spill when exceeded)",
    )
    dp.add_argument("--spill-dir", help="explicit controlled temp-storage directory")
    dp.set_defaults(func=cmd_diff)

    ep = sub.add_parser(
        "diff-explain",
        help="read/verify a published diff report and facts without any network I/O",
    )
    ep.add_argument("report")
    ep.add_argument("--diff", help="facts path (defaults to the report's facts.path)")
    ep.set_defaults(func=cmd_diff_explain)

    sp = sub.add_parser("serve", help="run the real Range HTTP test server")
    sp.add_argument("directory")
    sp.add_argument("--port", type=int, default=0)
    sp.set_defaults(func=cmd_serve)

    sp = sub.add_parser("generate", help="generate a large nested Parquet file")
    sp.add_argument("out")
    sp.add_argument("--rows-per-group", type=int, default=200_000)
    sp.add_argument("--num-groups", type=int, default=8)
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_generate)
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ParquetRangeError as exc:
        # Distinct, scriptable exit statuses for the diff acceptance cases.
        from parquet_range import (
            DiffCancelled,
            DiffPublishError,
            DuplicateKeyError,
            ManifestError,
            NullKeyError,
            SchemaDriftError,
            SegmentIntegrityError,
            UnsupportedKeyTypeError,
        )

        if isinstance(args, argparse.Namespace) and getattr(args, "command", "") == "diff":
            if isinstance(exc, (ManifestError,)):
                code = 3
            elif isinstance(exc, (SchemaDriftError, UnsupportedKeyTypeError, NullKeyError,
                                  DuplicateKeyError)):
                code = 4
            elif isinstance(exc, DiffPublishError):
                code = 5
            elif isinstance(exc, SegmentIntegrityError):
                code = 6
            elif isinstance(exc, DiffCancelled):
                code = 7
            else:
                code = 2
        else:
            code = 2
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        detail = getattr(exc, "diagnostics", None) or getattr(exc, "occurrences", None)
        if detail:
            print(json.dumps(detail, indent=2, ensure_ascii=False), file=sys.stderr)
        return code


if __name__ == "__main__":
    raise SystemExit(main())
