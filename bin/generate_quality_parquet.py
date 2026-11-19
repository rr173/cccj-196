#!/usr/bin/env python3
"""Generate a large multi-row-group Parquet file plus a quality-audit request.

The file mixes every type the quality contract supports (signed/unsigned
ints, floats with NaN, strings, booleans, timezone-aware and naive
timestamps, dates) with nested and wide columns that no contract rule ever
references, so projection sparsity is visible in the server log.  A
ready-to-use contract JSON and audit request JSON (with the real strong ETag
and entity length filled in) are written next to it.

Running this script writes ONLY generated data; it never calls the product
audit engine, so the same files can feed the independent oracle in tests.

Generated artifacts are excluded from version control via .gitignore.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field("small_i", pa.int8()),
    pa.field("big_u", pa.uint64()),
    pa.field("byte_u", pa.uint8()),
    pa.field("score", pa.float64()),
    pa.field("ratio", pa.float32()),
    pa.field("flag", pa.bool_()),
    pa.field("name", pa.string()),
    pa.field("ccy", pa.string()),
    pa.field("created", pa.timestamp("us", tz="UTC")),
    pa.field("updated", pa.timestamp("us", tz="UTC")),
    pa.field("ts_ns", pa.timestamp("ns")),
    pa.field("day", pa.date32()),
    pa.field("meta", pa.struct([
        pa.field("x", pa.int32()),
        pa.field("label", pa.string()),
    ])),
    pa.field("tags", pa.list_(pa.string())),
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(8)],
])

_BASE_TS = datetime.datetime(2022, 1, 1, tzinfo=datetime.timezone.utc)
_BASE_NS = datetime.datetime(2022, 1, 1)
_BASE_DAY = datetime.date(2022, 1, 1)
_CCY = ["USD", "EUR", "JPY", "GBP", "CHF"]


def _row(rng: random.Random, i: int) -> dict:
    return {
        "id": None if i % 13 == 0 else i,
        "small_i": (i % 200) - 100,
        "big_u": i if i % 5 == 0 else (2 ** 64 - 1 - i),
        "byte_u": i % 256,
        "score": float("nan") if i % 11 == 0 else float((i * 37) % 2000),
        "ratio": ((i % 40) - 20) / 10.0,
        "flag": i % 2 == 0,
        "name": f"bad name {i}" if i % 3 == 0 else f"Name_{i}",
        "ccy": _CCY[i % len(_CCY)],
        "created": _BASE_TS + datetime.timedelta(seconds=i),
        "updated": _BASE_TS + datetime.timedelta(seconds=i + (i % 7) - 3),
        "ts_ns": (
            _BASE_NS - datetime.timedelta(days=400)
            if i % 17 == 0
            else _BASE_NS + datetime.timedelta(seconds=i * 30)
        ),
        "day": (
            datetime.date(2018, 6, 1)
            if i % 19 == 0
            else _BASE_DAY + datetime.timedelta(days=i % 800)
        ),
        "meta": None if i % 4 == 0 else {"x": i, "label": f"m{i}"},
        "tags": None if i % 6 == 0 else [f"t{i}", "z"],
        **{f"wide_{k}": rng.random() * 1e6 for k in range(8)},
    }


def contract_document() -> dict:
    """The quality contract shipped next to the generated data."""
    return {
        "spec_version": 1,
        "fields": {
            "id": "int64",
            "small_i": "int8",
            "big_u": "uint64",
            "byte_u": "uint8",
            "score": "float64",
            "ratio": "float32",
            "flag": "bool",
            "name": "string",
            "ccy": "string",
            "created": "timestamp[us, tz=UTC]",
            "updated": "timestamp[us, tz=UTC]",
            "ts_ns": "timestamp[ns]",
            "day": "date32",
        },
        "rules": [
            {"id": "id-non-null", "kind": "not_null", "field": "id"},
            {"id": "small-range", "kind": "range", "field": "small_i",
             "min": -50, "max": 50},
            {"id": "big-u-range", "kind": "range", "field": "big_u",
             "min": 0, "max": 2 ** 64 - 100},
            {"id": "byte-u-range", "kind": "range", "field": "byte_u",
             "min": 10, "max": 250,
             "min_inclusive": False, "max_inclusive": False},
            {"id": "score-range", "kind": "range", "field": "score",
             "min": 0.0, "max": 1000.0, "nan_is_violation": True},
            {"id": "ratio-range", "kind": "range", "field": "ratio",
             "min": -1.5, "max": 1.5},
            {"id": "flag-enum", "kind": "enum", "field": "flag",
             "values": [True]},
            {"id": "ccy-enum", "kind": "enum", "field": "ccy",
             "values": ["USD", "EUR", "JPY"]},
            {"id": "name-pattern", "kind": "pattern", "field": "name",
             "pattern": "^[A-Z][a-z0-9_]{2,15}$"},
            {"id": "ts-order", "kind": "order",
             "left": "created", "right": "updated"},
            {"id": "ts-ns-range", "kind": "range", "field": "ts_ns",
             "min": "2021-01-01T00:00:00", "max": "2030-01-01T00:00:00"},
            {"id": "day-range", "kind": "range", "field": "day",
             "min": "2020-01-01", "max": "2030-12-31"},
        ],
    }


def build_file(out_dir: str, *, rows_per_group: int, num_groups: int,
               seed: int) -> dict:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)
    parquet_path = os.path.join(out_dir, "quality.parquet")
    with pq.ParquetWriter(parquet_path, SCHEMA, compression="zstd",
                          use_dictionary=True) as writer:
        for g in range(num_groups):
            rows = [_row(rng, g * rows_per_group + i)
                    for i in range(rows_per_group)]
            table = pa.table(
                {name: [r[name] for r in rows] for name in SCHEMA.names},
                schema=SCHEMA,
            )
            writer.write_table(table, row_group_size=rows_per_group)
    with open(parquet_path, "rb") as fh:
        data = fh.read()
    etag = '"' + hashlib.sha256(data).hexdigest()[:32] + '"'
    summary = {
        "path": parquet_path,
        "etag": etag,
        "length": len(data),
        "rows": rows_per_group * num_groups,
        "row_groups": num_groups,
    }
    contract_path = os.path.join(out_dir, "quality-contract.json")
    with open(contract_path, "w", encoding="utf-8") as fh:
        json.dump(contract_document(), fh, indent=2)
        fh.write("\n")
    summary["contract_path"] = contract_path
    return summary


def write_request_template(summary: dict, out_dir: str, base_url: str,
                           batch_rows: int) -> str:
    base_url = base_url.rstrip("/")
    request = {
        "spec_version": 1,
        "object": {
            "url": f"{base_url}/quality.parquet",
            "etag": summary["etag"],
            "length": summary["length"],
        },
        "contract_path": summary["contract_path"],
        "violations_path": os.path.join(out_dir, "results", "violations.jsonl"),
        "report_path": os.path.join(out_dir, "results", "report.json"),
        "batch_rows": batch_rows,
    }
    path = os.path.join(out_dir, "quality-audit-request.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(request, fh, indent=2)
        fh.write("\n")
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="directory for the generated data + request")
    ap.add_argument("--rows-per-group", type=int, default=20_000)
    ap.add_argument("--num-groups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="HTTP base URL embedded in the generated request template",
    )
    ap.add_argument("--batch-rows", type=int, default=4096)
    args = ap.parse_args()

    summary = build_file(
        args.out_dir,
        rows_per_group=args.rows_per_group,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    request_path = write_request_template(
        summary, args.out_dir, args.base_url, args.batch_rows
    )
    print(json.dumps(summary, indent=2))
    print(f"\nrequest template: {request_path}")
    print("serve with: python3 bin/pqr.py serve "
          f"{args.out_dir} --port 8765")
    print("audit with: python3 bin/pqr.py quality-audit "
          f"{request_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
