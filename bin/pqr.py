#!/usr/bin/env python3
"""Command-line interface for the sparse remote Parquet reader.

Subcommands:
    inspect      URL                       show pinned identity + Parquet metadata
    scan         URL --columns a,b --row-groups 0,2 [--format json|arrow|csv]
    cache-clear  [URL]                     drop unpinned cached overlays
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

import pyarrow as pa  # noqa: E402

from parquet_range import (  # noqa: E402
    ClientConfig,
    ParquetRangeError,
    RemoteParquetClient,
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
        print(f"error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
