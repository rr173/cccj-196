#!/usr/bin/env python3
"""Generate a skewed temporal-enrich dataset plus a ready-to-run request.

The generated dataset is deliberately hostile and reproducible:

* one hot entity key carries at least 100k window revisions and 100k facts
  (defaults: 1000 windows x 100 revisions; 100k facts), with heavily
  overlapping windows, withdrawals and re-opens, and registration times
  shuffled across shards and row groups;
* a few thousand normal keys with small window/fact counts fill the rest;
* fact/revision shards carry wide unused columns so a correct
  implementation's network traffic is a small fraction of the objects.

The generator writes the Parquet shards, a request JSON with the real
strong ETags/lengths, and a sample-expectations JSON: for a deterministic
sample of output rows it computes the expected status/winner-window ids by
brute force from the generated revisions (the product is never involved),
so any run can be checked against ground truth.

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

FACT_SCHEMA = pa.schema([
    pa.field("entity_key", pa.string()),
    pa.field("event_at", pa.int64()),
    pa.field("known_at", pa.int64()),
    pa.field("fact_id", pa.int64()),
    pa.field("payload", pa.string()),
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(8)],
])

REVISION_SCHEMA = pa.schema([
    pa.field("entity_key", pa.string()),
    pa.field("window_id", pa.string()),
    pa.field("recorded_at", pa.int64()),
    pa.field("op", pa.string()),
    pa.field("valid_from", pa.int64()),
    pa.field("valid_to", pa.int64()),
    pa.field("priority", pa.int64()),
    pa.field("service_area", pa.string()),
    pa.field("owner", pa.string()),
    pa.field("quota", pa.int64()),
    pa.field("note", pa.string()),
    *[pa.field(f"wide_{i}", pa.float64()) for i in range(4)],
])

HOT_KEY = "entity-hot"
FACT_COLUMNS = ["fact_id", "payload"]
DIM_COLUMNS = ["service_area", "owner", "quota"]
TIME_BASE = 1_000_000


def hot_revisions(n_windows: int, per_window: int) -> list[dict]:
    """Deterministic hot-key window revisions (overlapping, moving windows)."""
    rows = []
    for i in range(n_windows):
        wid = f"w-{i:04d}"
        for j in range(per_window):
            recorded_at = TIME_BASE + j * 1000 + (i % 500)
            vf = i * 100 + (j % 5) * 10
            vt = vf + 5000
            if j > 0 and j % 17 == 0:
                rows.append({
                    "entity_key": HOT_KEY, "window_id": wid,
                    "recorded_at": recorded_at, "op": "withdraw",
                    "valid_from": None, "valid_to": None, "priority": None,
                    "service_area": None, "owner": None, "quota": None,
                })
            else:
                rows.append({
                    "entity_key": HOT_KEY, "window_id": wid,
                    "recorded_at": recorded_at, "op": "upsert",
                    "valid_from": vf, "valid_to": vt,
                    "priority": (i + j) % 7 - 3,
                    "service_area": f"area-{(i * 31 + j) % 50}",
                    "owner": f"owner-{i % 20}",
                    "quota": i * 1000 + j,
                })
    return rows


def hot_facts(n_facts: int) -> list[dict]:
    rows = []
    for k in range(n_facts):
        rows.append({
            "entity_key": HOT_KEY,
            "event_at": (k * 37) % 105_000,
            "known_at": TIME_BASE + (k * 997) % 100_000,
            "fact_id": k,
            "payload": f"payload-{k % 1000}",
        })
    return rows


def normal_revisions(n_keys: int, rng: random.Random) -> list[dict]:
    rows = []
    for k in range(n_keys):
        key = f"entity-{k:06d}"
        for w in range(1 + k % 3):
            wid = f"nw-{k}-{w}"
            vf = rng.randint(0, 500)
            for r in range(1 + (k + w) % 3):
                rows.append({
                    "entity_key": key, "window_id": wid,
                    "recorded_at": TIME_BASE + k * 10 + r,
                    "op": "upsert",
                    "valid_from": vf + r, "valid_to": vf + 400 + r,
                    "priority": (k + w + r) % 5,
                    "service_area": f"area-{k % 7}",
                    "owner": f"owner-{k % 5}",
                    "quota": k * 10 + w,
                })
    return rows


def normal_facts(n_keys: int, rng: random.Random) -> list[dict]:
    rows = []
    fid = 0
    for k in range(n_keys):
        for f in range(1 + k % 4):
            rows.append({
                "entity_key": f"entity-{k:06d}",
                "event_at": rng.randint(0, 1000),
                "known_at": TIME_BASE + k * 10 + rng.randint(0, 5),
                "fact_id": 10_000_000 + fid,
                "payload": f"np-{k}-{f}",
            })
            fid += 1
    return rows


def _with_wide_fact(row: dict, i: int) -> dict:
    return {
        **row,
        **{f"wide_{c}": float((i * 13 + c * 7) % 1000) / 3.0 for c in range(8)},
    }


def _with_wide_rev(row: dict, i: int) -> dict:
    return {
        "note": f"note-{i % 100}",
        **row,
        **{f"wide_{c}": float((i * 11 + c * 5) % 1000) / 7.0 for c in range(4)},
    }


def _write_shards(rows: list[dict], schema: pa.Schema, out_dir: str,
                  stem: str, n_shards: int, rows_per_group: int,
                  rng: random.Random) -> list[dict]:
    """Shuffle rows, split into n_shards parquet files, return manifests."""
    rows = list(rows)
    rng.shuffle(rows)
    shards = []
    per = (len(rows) + n_shards - 1) // n_shards
    for s in range(n_shards):
        part = rows[s * per:(s + 1) * per]
        if not part:
            continue
        path = os.path.join(out_dir, f"{stem}-{s}.parquet")
        with pq.ParquetWriter(path, schema, compression="snappy",
                              use_dictionary=False) as writer:
            for start in range(0, len(part), rows_per_group):
                chunk = part[start:start + rows_per_group]
                table = pa.table(
                    {name: [r[name] for r in chunk] for name in schema.names},
                    schema=schema,
                )
                writer.write_table(table, row_group_size=len(chunk))
        data = open(path, "rb").read()
        shards.append({
            "shard_id": f"{stem}-{s}",
            "path": path,
            "etag": '"' + hashlib.sha256(data).hexdigest()[:32] + '"',
            "length": len(data),
            "rows": len(part),
            "row_order": [r.get("fact_id") for r in part] if stem == "facts" else None,
        })
    return shards


def _expected_for_fact(revisions: list[dict], event_at: int,
                       known_at: int) -> dict:
    """Brute-force expected result for one fact (no product code involved)."""
    state: dict[str, dict] = {}
    for rev in revisions:
        if rev["recorded_at"] <= known_at:
            cur = state.get(rev["window_id"])
            if cur is None or rev["recorded_at"] > cur["recorded_at"]:
                state[rev["window_id"]] = rev
    candidates = []
    for rev in state.values():
        if rev["op"] != "upsert":
            continue
        vt = rev["valid_to"]
        if rev["valid_from"] <= event_at and (vt is None or event_at < vt):
            candidates.append(rev)
    if not candidates:
        return {"status": "unmatched", "count": 0, "dim": None, "windows": []}
    top = max(r["priority"] for r in candidates)
    tied = sorted((r for r in candidates if r["priority"] == top),
                  key=lambda r: r["window_id"])
    if len(tied) == 1:
        win = tied[0]
        return {
            "status": "matched", "count": 1,
            "dim": {c: win[c] for c in DIM_COLUMNS},
            "windows": [win["window_id"]],
        }
    return {
        "status": "ambiguous", "count": len(tied), "dim": None,
        "windows": [tied[0]["window_id"], tied[1]["window_id"]],
    }


def build_dataset(out_dir: str, *, hot_n_facts: int, hot_n_windows: int,
                  hot_revs_per_window: int, normal_keys: int, n_shards: int,
                  rows_per_group: int, seed: int,
                  sample_size: int) -> dict:
    rng = random.Random(seed)
    os.makedirs(out_dir, exist_ok=True)

    revisions = hot_revisions(hot_n_windows, hot_revs_per_window)
    revisions += normal_revisions(normal_keys, rng)
    facts = hot_facts(hot_n_facts) + normal_facts(normal_keys, rng)

    rev_rows = [_with_wide_rev(r, i) for i, r in enumerate(revisions)]
    fact_rows = [_with_wide_fact(r, i) for i, r in enumerate(facts)]

    rev_shards = _write_shards(rev_rows, REVISION_SCHEMA, out_dir, "revisions",
                               n_shards, rows_per_group, rng)
    fact_shards = _write_shards(fact_rows, FACT_SCHEMA, out_dir, "facts",
                                n_shards, rows_per_group, rng)

    # Output row order = manifest shard order + physical row order; the
    # generator knows exactly which fact_id lands on which output row.
    output_fact_ids: list[int] = []
    for shard in fact_shards:
        output_fact_ids.extend(shard["row_order"])
    for shard in fact_shards:
        shard.pop("row_order", None)

    # Brute-force expected results for a deterministic sample of output rows.
    hot_rev_rows = [r for r in revisions if r["entity_key"] == HOT_KEY]
    normal_rev_by_key: dict[str, list[dict]] = {}
    for r in revisions:
        if r["entity_key"] != HOT_KEY:
            normal_rev_by_key.setdefault(r["entity_key"], []).append(r)
    fact_by_id = {f["fact_id"]: f for f in facts}
    sample_indices = sorted({
        (i * 9973) % len(output_fact_ids) for i in range(sample_size)
    } | {0, len(output_fact_ids) // 2, len(output_fact_ids) - 1})
    samples = []
    for out_row in sample_indices:
        fid = output_fact_ids[out_row]
        fact = fact_by_id[fid]
        revs = (hot_rev_rows if fact["entity_key"] == HOT_KEY
                else normal_rev_by_key.get(fact["entity_key"], []))
        exp = _expected_for_fact(revs, fact["event_at"], fact["known_at"])
        samples.append({
            "output_row": out_row,
            "fact_id": fid,
            "entity_key": fact["entity_key"],
            "event_at": fact["event_at"],
            "known_at": fact["known_at"],
            "expected": exp,
        })

    summary = {
        "seed": seed,
        "hot_key": HOT_KEY,
        "hot": {
            "facts": hot_n_facts,
            "revisions": hot_n_windows * hot_revs_per_window,
            "windows": hot_n_windows,
        },
        "normal_keys": normal_keys,
        "totals": {"facts": len(facts), "revisions": len(revisions)},
        "fact_columns": FACT_COLUMNS,
        "dim_columns": DIM_COLUMNS,
        "fact_shards": fact_shards,
        "revision_shards": rev_shards,
        "sample_expectations": samples,
    }
    with open(os.path.join(out_dir, "temporal-summary.json"), "w",
              encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    return summary


def write_request_template(summary: dict, out_dir: str, base_url: str,
                           memory_limit: int, spill_dir: str | None) -> str:
    base_url = base_url.rstrip("/")
    request = {
        "spec_version": 1,
        "fact_shards": [
            {"shard_id": s["shard_id"],
             "url": f"{base_url}/{os.path.basename(s['path'])}",
             "etag": s["etag"], "length": s["length"]}
            for s in summary["fact_shards"]
        ],
        "revision_shards": [
            {"shard_id": s["shard_id"],
             "url": f"{base_url}/{os.path.basename(s['path'])}",
             "etag": s["etag"], "length": s["length"]}
            for s in summary["revision_shards"]
        ],
        "fact_columns": summary["fact_columns"],
        "dim_columns": summary["dim_columns"],
        "output_path": os.path.join(out_dir, "results", "wide.parquet"),
        "memory_limit": memory_limit,
    }
    if spill_dir:
        request["spill_dir"] = spill_dir
    path = os.path.join(out_dir, "temporal-request.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(request, fh, indent=2)
    return path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("out_dir", help="directory for generated shards + request")
    ap.add_argument("--hot-facts", type=int, default=100_000)
    ap.add_argument("--hot-windows", type=int, default=1000)
    ap.add_argument("--hot-revisions-per-window", type=int, default=100)
    ap.add_argument("--normal-keys", type=int, default=2000)
    ap.add_argument("--shards", type=int, default=3)
    ap.add_argument("--rows-per-group", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sample-size", type=int, default=48)
    ap.add_argument("--memory-limit", type=int, default=4 << 20)
    ap.add_argument("--spill-dir", default=None)
    ap.add_argument(
        "--base-url", default="http://127.0.0.1:8765",
        help="HTTP base URL embedded in the generated request template",
    )
    args = ap.parse_args()

    summary = build_dataset(
        args.out_dir,
        hot_n_facts=args.hot_facts,
        hot_n_windows=args.hot_windows,
        hot_revs_per_window=args.hot_revisions_per_window,
        normal_keys=args.normal_keys,
        n_shards=args.shards,
        rows_per_group=args.rows_per_group,
        seed=args.seed,
        sample_size=args.sample_size,
    )
    request_path = write_request_template(
        summary, args.out_dir, args.base_url, args.memory_limit, args.spill_dir
    )
    print(json.dumps({k: v for k, v in summary.items()
                      if k != "sample_expectations"}, indent=2)[:2000])
    print(f"\nrequest template: {request_path}")
    print(f"sample expectations: {len(summary['sample_expectations'])} rows")
    print(f"serve with: python3 bin/pqr.py serve {args.out_dir} --port 8765")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
