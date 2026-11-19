#!/usr/bin/env python3
"""Generate a large baseline/target Parquet pair for diff acceptance runs.

The two files share one schema and a configurable scalar key type, are spread
over many row groups, and the *target* is deliberately produced by:

  * reordering the surviving records (the diff must not assume sorted input),
  * updating a deterministic subset (chosen comparison fields change),
  * deleting a deterministic subset,
  * inserting brand-new keys,

so an independent check can predict every insert/delete/update/no-change.

Running this script writes ONLY generated data plus a ready-to-use diff
request JSON (the strong ETags and entity lengths are filled in from the
actual bytes).  It never calls the product diff engine, so it can also feed the
independent PyArrow oracle in tests and manual audits.

Generated artifacts are excluded from version control via .gitignore.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random

import pyarrow as pa
import pyarrow.parquet as pq

SCHEMA = pa.schema([
    pa.field("kid", pa.int64()),
    pa.field("code", pa.string()),
    pa.field("active", pa.bool_()),
    pa.field("tags", pa.list_(pa.string())),
    pa.field("meta", pa.struct([
        pa.field("score", pa.float64()),
        pa.field("label", pa.string()),
        pa.field("flags", pa.list_(pa.int32())),
    ])),
    pa.field("cat", pa.dictionary(pa.int16(), pa.string())),
    pa.field("blob", pa.binary()),
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(8)],
])

CATEGORIES = ["alpha", "beta", "gamma", "delta", "epsilon"]
COMPARE_COLUMNS = ["code", "active", "tags", "meta", "cat", "blob"]


def _row(rng: random.Random, kid: int, *, mutate: bool = False) -> dict:
    tags = None if kid % 11 == 0 else (
        [] if kid % 7 == 0 else [f"t{kid % 5}", "z"]
    )
    # Deterministic per-key values (independent of any per-run RNG) so the
    # baseline and target render a row equal unless ``mutate`` is set; this is
    # what makes the generator's expected counts an exact independent oracle.
    meta = None if kid % 13 == 0 else {
        "score": round((kid % 1000) / 1000.0, 6),
        "label": None if kid % 4 == 0 else f"label-{kid % 97}",
        "flags": None if kid % 5 == 0 else [kid % 4, (kid + 1) % 4],
    }
    cat = None if kid % 9 == 0 else CATEGORIES[kid % len(CATEGORIES)]
    blob = None if kid % 6 == 0 else bytes((kid * 7 + j) & 0xFF
                                           for j in range(8 + kid % 24))
    return {
        "kid": kid,
        "code": f"code-{kid}" + ("-CHANGED" if mutate else ""),
        "active": (kid % 2 == 0) != mutate,
        "tags": tags,
        "meta": meta,
        "cat": cat,
        "blob": blob,
        # Deterministic wide columns too, so they never perturb the comparison.
        **{f"wide_{i}": ((kid * 31 + i * 17) % 10000) / 100.0
           for i in range(8)},
    }


def _to_table(rows: list[dict]) -> pa.Table:
    return pa.table({name: [r[name] for r in rows] for name in SCHEMA.names},
                    schema=SCHEMA)


def _write(path: str, rows: list[dict], groups: int, rows_per_group: int) -> None:
    with pq.ParquetWriter(path, SCHEMA, compression="zstd",
                          use_dictionary=True) as writer:
        for g in range(groups):
            part = rows[g * rows_per_group:(g + 1) * rows_per_group]
            if part:
                writer.write_table(_to_table(part), row_group_size=len(part))


def build_pair(out_dir: str, *, rows_per_group: int, num_groups: int,
               seed: int) -> dict:
    rng = random.Random(seed)
    total = rows_per_group * num_groups
    baseline_ids = list(range(1, total + 1))
    baseline = [_row(random.Random(seed + k), k) for k in baseline_ids]

    delete_set = {k for k in baseline_ids if k % 17 == 0}
    # Updates must be disjoint from deletes (a deleted row cannot be updated).
    update_set = {k for k in baseline_ids
                  if k % 23 == 0 and k not in delete_set}
    kept = [k for k in baseline_ids if k not in delete_set]
    rng.shuffle(kept)  # reorder surviving records on purpose

    target = []
    for k in kept:
        target.append(_row(random.Random(seed + k), k,
                           mutate=(k in update_set)))
    next_id = total + 1
    insert_set = set()
    for _ in range(max(1, total // 50)):
        insert_set.add(next_id)
        target.append(_row(random.Random(seed + next_id), next_id))
        next_id += 1
    rng.shuffle(target)

    os.makedirs(out_dir, exist_ok=True)
    left_path = os.path.join(out_dir, "baseline.parquet")
    right_path = os.path.join(out_dir, "target.parquet")
    _write(left_path, baseline, num_groups, rows_per_group)
    _write(right_path, target, num_groups, rows_per_group)

    def etag_of(path: str) -> tuple[str, bytes]:
        data = open(path, "rb").read()
        return '"' + hashlib.sha256(data).hexdigest()[:32] + '"', data

    le, ldata = etag_of(left_path)
    re_, rdata = etag_of(right_path)
    md_l = pq.read_metadata(left_path)
    md_r = pq.read_metadata(right_path)
    summary = {
        "baseline": {
            "path": left_path, "etag": le, "length": len(ldata),
            "rows": md_l.num_rows, "row_groups": md_l.num_row_groups,
        },
        "target": {
            "path": right_path, "etag": re_, "length": len(rdata),
            "rows": md_r.num_rows, "row_groups": md_r.num_row_groups,
        },
        "expected_counts": {
            "insert": len(insert_set),
            "delete": len(delete_set),
            "update": len(update_set),
            "no-change": len(kept) - len(update_set),
        },
        "key_column": "kid",
        "compare_columns": COMPARE_COLUMNS,
    }
    with open(os.path.join(out_dir, "pair-summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def write_request_template(summary: dict, out_dir: str, base_url: str,
                           memory_limit: int) -> str:
    base_url = base_url.rstrip("/")
    request = {
        "spec_version": 1,
        "left": {
            "url": f"{base_url}/baseline.parquet",
            "etag": summary["baseline"]["etag"],
            "length": summary["baseline"]["length"],
        },
        "right": {
            "url": f"{base_url}/target.parquet",
            "etag": summary["target"]["etag"],
            "length": summary["target"]["length"],
        },
        "key_column": summary["key_column"],
        "compare_columns": summary["compare_columns"],
        "diff_path": os.path.join(out_dir, "results", "facts.jsonl"),
        "report_path": os.path.join(out_dir, "results", "report.json"),
        "memory_limit": memory_limit,
    }
    path = os.path.join(out_dir, "diff-request.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(request, fh, indent=2)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="directory for generated pair + request")
    ap.add_argument("--rows-per-group", type=int, default=20_000)
    ap.add_argument("--num-groups", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument(
        "--base-url",
        default="http://127.0.0.1:8765",
        help="HTTP base URL embedded in the generated request template",
    )
    ap.add_argument("--memory-limit", type=int, default=8 << 20)
    args = ap.parse_args()

    summary = build_pair(
        args.out_dir,
        rows_per_group=args.rows_per_group,
        num_groups=args.num_groups,
        seed=args.seed,
    )
    request_path = write_request_template(
        summary, args.out_dir, args.base_url, args.memory_limit
    )
    print(json.dumps(summary, indent=2))
    print(f"\nrequest template: {request_path}")
    print(
        "serve with: python3 bin/pqr.py serve "
        f"{args.out_dir} --port 8765"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
