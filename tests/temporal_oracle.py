"""Independent temporal-enrich oracle: brute-force expected results.

This module never imports the product's matching engine.  It decodes the
generated shards with local PyArrow (or consumes the fixture's in-memory
rows) and computes, for every fact, the expected status / dimension values /
window count / source coordinates by literally applying the bitemporal
contract per fact:

* per window, the visible revision is the last one with
  ``recorded_at <= known_at`` (none or ``withdraw`` -> window absent);
* a visible ``upsert`` covers iff ``valid_from <= event_at < valid_to``
  (``valid_to=None`` is unbounded);
* the answer is the maximum priority among covering windows: one top window
  -> ``matched`` with its dimension values; several -> ``ambiguous`` (never
  a winner); none -> ``unmatched``.

The oracle is deliberately O(revisions-of-key) per fact -- correctness by
brute force, exactly what the product must not do.
"""

from __future__ import annotations


def expected_for_fact(revisions, event_at, known_at, dim_columns):
    """Compute the expected outcome for one fact from positioned revisions.

    ``revisions`` is a list of ``(row_dict, shard_id, row_group, row)``.
    Returns a dict with status/count/dim/dim_sources.
    """
    state: dict[str, tuple] = {}   # window_id -> (row_dict, shard, rg, row)
    for rev, shard_id, rg, row in revisions:
        if rev["recorded_at"] > known_at:
            continue
        cur = state.get(rev["window_id"])
        if cur is None or rev["recorded_at"] > cur[0]["recorded_at"]:
            state[rev["window_id"]] = (rev, shard_id, rg, row)
    candidates = []
    for wid, (rev, shard_id, rg, row) in state.items():
        if rev["op"] != "upsert":
            continue
        vt = rev["valid_to"]
        if rev["valid_from"] <= event_at and (vt is None or event_at < vt):
            candidates.append((wid, rev, shard_id, rg, row))
    if not candidates:
        return {
            "status": "unmatched",
            "count": 0,
            "dim": None,
            "dim_sources": None,
        }
    top = max(rev["priority"] for _, rev, _, _, _ in candidates)
    tied = sorted(
        (c for c in candidates if c[1]["priority"] == top),
        key=lambda c: c[0].encode("utf-8"),
    )
    if len(tied) == 1:
        wid, rev, shard_id, rg, row = tied[0]
        return {
            "status": "matched",
            "count": 1,
            "dim": {c: rev[c] for c in dim_columns},
            "dim_sources": [{
                "shard_id": shard_id, "row_group": rg, "row": row,
                "window_id": wid,
            }],
        }
    return {
        "status": "ambiguous",
        "count": len(tied),
        "dim": None,
        "dim_sources": [
            {"shard_id": s, "row_group": rg, "row": row, "window_id": wid}
            for wid, _, s, rg, row in tied[:2]
        ],
    }


def oracle_enrich(positioned_facts, positioned_revisions, dim_columns):
    """Expected results for every fact, in fact (output) order."""
    by_key: dict[str, list] = {}
    for entry in positioned_revisions:
        by_key.setdefault(entry[0]["entity_key"], []).append(entry)
    results = []
    for fact, shard_id, rg, row in positioned_facts:
        exp = expected_for_fact(
            by_key.get(fact["entity_key"], []),
            fact["event_at"],
            fact["known_at"],
            dim_columns,
        )
        exp["fact_source"] = {
            "shard_id": shard_id, "row_group": rg, "row": row,
        }
        results.append(exp)
    return results
