#!/usr/bin/env python3
"""Generate a large Parquet file exercising nested, dictionary and null cases.

Schema (deliberately wider than any single query so sparse selection pays
off):

    id        int64
    events    list<struct<ts: timestamp[us], code: int16, msg: string>>
    meta      struct<source: string, score: double, flags: list<int32>>
    category  dictionary<int8, string>
    payload   binary
    filler_1..filler_6  float64   # wide, rarely selected columns

Every row group independently contains nulls at every nesting level, empty
lists, and dictionary-encoded strings.  The output path is printed with size
and row-group layout for use against the Range test server.
"""

from __future__ import annotations

import argparse
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("id", pa.int64()),
    pa.field(
        "events",
        pa.list_(
            pa.struct([
                pa.field("ts", pa.timestamp("us")),
                pa.field("code", pa.int16()),
                pa.field("msg", pa.string()),
            ])
        ),
    ),
    pa.field(
        "meta",
        pa.struct([
            pa.field("source", pa.string()),
            pa.field("score", pa.float64()),
            pa.field("flags", pa.list_(pa.int32())),
        ]),
    ),
    pa.field("category", pa.dictionary(pa.int8(), pa.string())),
    pa.field("payload", pa.binary()),
    *[pa.field(f"filler_{i}", pa.float64()) for i in range(1, 7)],
])

CATEGORIES = ["alpha", "beta", "gamma", "delta", "epsilon"]
SOURCES = ["sensor-a", "sensor-b", "manual", "replay"]


def _row(rng: random.Random, row_id: int) -> dict:
    if rng.random() < 0.08:
        events = None
    elif rng.random() < 0.1:
        events = []
    else:
        events = [
            {
                "ts": 1_700_000_000_000_000 + row_id * 1000 + j * 7,
                "code": rng.randint(-100, 100),
                "msg": None if rng.random() < 0.2 else f"msg-{rng.randint(0, 999)}",
            }
            for j in range(rng.randint(0, 4))
        ]
    if rng.random() < 0.07:
        meta = None
    else:
        meta = {
            "source": None if rng.random() < 0.15 else rng.choice(SOURCES),
            "score": None if rng.random() < 0.1 else rng.random() * 100,
            "flags": None if rng.random() < 0.12 else [rng.randint(0, 9) for _ in range(rng.randint(0, 5))],
        }
    category = None if rng.random() < 0.06 else rng.choice(CATEGORIES)
    payload = None if rng.random() < 0.2 else os.urandom(rng.randint(8, 64))
    row = {
        "id": row_id,
        "events": events,
        "meta": meta,
        "category": category,
        "payload": payload,
    }
    for i in range(1, 7):
        row[f"filler_{i}"] = None if rng.random() < 0.05 else rng.random() * 1e6
    return row


def build_table(rows: int, base_id: int, seed: int) -> pa.Table:
    rng = random.Random(seed)
    rows_data = [_row(rng, base_id + i) for i in range(rows)]
    arrays = {}
    for name in SCHEMA.names:
        arrays[name] = pa.array(
            [r[name] for r in rows_data], type=SCHEMA.field(name).type
        )
    return pa.table(arrays, schema=SCHEMA)


def build_file(
    path: str,
    *,
    rows_per_group: int = 200_000,
    num_groups: int = 8,
    seed: int = 7,
) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with pq.ParquetWriter(path, SCHEMA, compression="zstd", use_dictionary=True) as w:
        for g in range(num_groups):
            table = build_table(rows_per_group, g * rows_per_group, seed + g)
            w.write_table(table, row_group_size=rows_per_group)
    md = pq.read_metadata(path)
    print(
        f"generated {path}: {os.path.getsize(path):,} bytes, "
        f"{md.num_row_groups} row groups, {md.num_rows:,} rows, "
        f"{md.num_columns} leaf columns"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out")
    ap.add_argument("--rows-per-group", type=int, default=200_000)
    ap.add_argument("--num-groups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    build_file(
        args.out,
        rows_per_group=args.rows_per_group,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
