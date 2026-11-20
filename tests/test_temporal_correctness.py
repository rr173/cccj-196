"""Temporal-enrich correctness tests (real Range server + independent oracle).

The oracle in ``temporal_oracle.py`` computes expected results by brute
force from the generated rows; the product's wide table is compared against
it row by row, including provenance coordinates and value/type fidelity of
every business column.
"""

from __future__ import annotations

import datetime
import random
from decimal import Decimal

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import TemporalEnrichEngine

from temporal_conftest import (
    DIM_COLUMNS,
    FACT_COLUMNS,
    FACT_SCHEMA,
    REVISION_SCHEMA,
    fact_row,
    publish_shards,
    rev_upsert,
    rev_withdraw,
)
from temporal_oracle import oracle_enrich


def _read_output(path):
    table = pq.read_table(path)
    return table, table.to_pylist()


def _check_row_against_oracle(out_row, expected, fact, fact_columns, dim_columns):
    # Selected fact columns carried through with exact values.
    for col in fact_columns:
        assert out_row[col] == fact[col], (col, out_row[col], fact[col])
    assert out_row["_match_status"] == expected["status"]
    assert out_row["_top_window_count"] == expected["count"]
    if expected["dim"] is None:
        assert out_row["_dim"] is None
    else:
        assert out_row["_dim"] is not None
        for col in dim_columns:
            assert out_row["_dim"][col] == expected["dim"][col], (
                col, out_row["_dim"][col], expected["dim"][col],
            )
    assert out_row["_fact_source"] == expected["fact_source"]
    assert out_row["_dim_sources"] == expected["dim_sources"]


def test_timeline_scenario_required_sequence(
    temporal_client_factory, timeline_dataset, make_temporal_request
):
    """The required bitemporal timeline: A's window moves/withdraws/reopens."""
    client = temporal_client_factory()
    request = make_temporal_request(
        timeline_dataset["fact_manifests"],
        timeline_dataset["revision_manifests"],
    )
    report = TemporalEnrichEngine(client, request).run()
    assert report["status"] == "completed"

    table, rows = _read_output(request.output_path)
    assert len(rows) == len(timeline_dataset["facts"])
    expected = oracle_enrich(
        timeline_dataset["facts"], timeline_dataset["revisions"], DIM_COLUMNS
    )
    for out_row, exp, (fact, _, _, _) in zip(
        rows, expected, timeline_dataset["facts"]
    ):
        _check_row_against_oracle(out_row, exp, fact, FACT_COLUMNS, DIM_COLUMNS)

    # The required sequence for event 25 of key A at known_at 19/20/30/40:
    # A's initial version, the low-priority window, the low-priority window,
    # A's reopened version.
    key_a_event25 = [r for r in rows
                     if r["fact_id"] in (0, 1, 2, 3)]
    key_a_event25.sort(key=lambda r: r["fact_id"])
    assert [r["_match_status"] for r in key_a_event25] == [
        "matched", "matched", "matched", "matched",
    ]
    assert [r["_dim"]["service_area"] for r in key_a_event25] == [
        "A1", "L", "L", "A3",
    ]
    # Key B: the high-priority W3 ties W1 only at known_at 19.
    key_b_event25 = [r for r in rows if r["fact_id"] in (7, 8, 9, 10)]
    key_b_event25.sort(key=lambda r: r["fact_id"])
    assert key_b_event25[0]["_match_status"] == "ambiguous"
    assert key_b_event25[0]["_top_window_count"] == 2
    assert key_b_event25[0]["_dim"] is None
    assert [s["window_id"] for s in key_b_event25[0]["_dim_sources"]] == [
        "W1", "W3",
    ]
    assert [r["_dim"]["service_area"] for r in key_b_event25[1:]] == [
        "H", "H", "A3",
    ]


def test_boundary_and_tie_positions(
    temporal_client_factory, timeline_dataset, make_temporal_request
):
    """Left-closed/right-open boundaries and tie provenance positions."""
    client = temporal_client_factory()
    request = make_temporal_request(
        timeline_dataset["fact_manifests"],
        timeline_dataset["revision_manifests"],
    )
    TemporalEnrichEngine(client, request).run()
    _, rows = _read_output(request.output_path)
    by_id = {r["fact_id"]: r for r in rows}

    # Key B boundary probes (fact_ids 11, 12, 13):
    # event 26 at K=19: W3 [25,26) does not cover (right-open) -> W1 wins.
    assert by_id[11]["_match_status"] == "matched"
    assert by_id[11]["_dim"]["service_area"] == "A1"
    # event 50 at K=20: W1 [50,100) covers its left edge -> A2.
    assert by_id[12]["_dim"]["service_area"] == "A2"
    # event 100 at K=40: W1 [0,100) does not cover its right edge -> L.
    assert by_id[13]["_dim"]["service_area"] == "L"

    # Ambiguous row: two smallest window_ids in UTF-8 byte order, with the
    # winning revisions' exact source coordinates.
    amb = by_id[7]
    sources = amb["_dim_sources"]
    assert len(sources) == 2
    assert sources[0]["window_id"] == "W1"
    assert sources[1]["window_id"] == "W3"
    # The W1 source points at the revision visible at K=19 (recorded_at=10).
    w1_source = sources[0]
    rev_rows = timeline_dataset["revisions"]
    positioned = {
        (s, rg, row): rev for rev, s, rg, row in rev_rows
    }
    src_rev = positioned[
        (w1_source["shard_id"], w1_source["row_group"], w1_source["row"])
    ]
    assert src_rev["recorded_at"] == 10
    assert src_rev["valid_from"] == 0 and src_rev["valid_to"] == 100


def test_output_order_and_duplicate_facts_preserved(
    temporal_client_factory, temporal_server, tmp_path, make_temporal_request
):
    """Facts are emitted in manifest shard order + physical row order."""
    facts = [
        fact_row(0, "k1", 10, 100),
        fact_row(1, "k2", 10, 100),
        fact_row(2, "k1", 20, 100),
        fact_row(1, "k2", 10, 100),   # exact duplicate of fact 1 (in shard 1)
        fact_row(3, "k3", 5, 50),
    ]
    revisions = [rev_upsert("k1", "w", 1, 0, 1000, 1, area="X")]
    fact_manifests, positioned_facts = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA,
        n_shards=2, rows_per_group=2,
    )
    rev_manifests, positioned_revs = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        n_shards=1, rows_per_group=10,
    )
    client = temporal_client_factory()
    request = make_temporal_request(fact_manifests, rev_manifests)
    TemporalEnrichEngine(client, request).run()
    _, rows = _read_output(request.output_path)
    # One output row per fact, duplicates preserved, manifest/physical order.
    assert [r["fact_id"] for r in rows] == [0, 1, 2, 1, 3]
    # Provenance coordinates are distinct even for identical facts.
    assert rows[1]["_fact_source"] != rows[3]["_fact_source"]
    assert rows[1]["_fact_source"]["shard_id"] == "facts-0"
    assert rows[3]["_fact_source"]["shard_id"] == "facts-1"
    assert rows[1]["_fact_source"]["row"] == 1
    assert rows[3]["_fact_source"]["row"] == 0
    # k3 has no revisions -> unmatched with empty-window count.
    assert rows[4]["_match_status"] == "unmatched"
    assert rows[4]["_top_window_count"] == 0
    assert rows[4]["_dim"] is None
    assert rows[4]["_dim_sources"] is None


def test_matched_row_with_null_business_column_is_not_confused_with_unmatched(
    temporal_client_factory, temporal_server, tmp_path, make_temporal_request
):
    facts = [fact_row(0, "k", 10, 100)]
    revisions = [
        rev_upsert("k", "w", 1, 0, 100, 1, area=None, owner="o", quota=None),
    ]
    # Null out the whole optional dim set except owner.
    rev = revisions[0]
    rev["service_area"] = None
    rev["quota"] = None
    rev["blob"] = None
    rev["rate"] = None
    rev["since"] = None
    rev["active"] = None
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA,
        n_shards=1, rows_per_group=10,
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        n_shards=1, rows_per_group=10,
    )
    client = temporal_client_factory()
    request = make_temporal_request(fact_manifests, rev_manifests)
    TemporalEnrichEngine(client, request).run()
    table, rows = _read_output(request.output_path)
    assert rows[0]["_match_status"] == "matched"
    # The struct itself is NOT null even though most fields are.
    dim_col = table.column("_dim")
    assert dim_col.null_count == 0
    assert rows[0]["_dim"]["service_area"] is None
    assert rows[0]["_dim"]["quota"] is None
    assert rows[0]["_dim"]["owner"] == "o"
    # Value type fidelity through the round trip.
    assert rows[0]["amount"] == Decimal("0.001")
    assert rows[0]["tag"] == bytes([0, 0, 0])
    assert rows[0]["moment"] == datetime.datetime(2022, 3, 1)
    # Schema metadata carries the shard identity map.
    meta = table.schema.metadata
    assert meta is not None and b"pqr.temporal_enrich" in meta
    import json as _json
    info = _json.loads(meta[b"pqr.temporal_enrich"].decode("utf-8"))
    assert info["shards"]["facts-0"]["etag"] == fact_manifests[0]["etag"]
    assert info["shards"]["revisions-0"]["length"] == rev_manifests[0]["length"]


def test_randomized_small_dataset_matches_oracle(
    temporal_client_factory, temporal_server, tmp_path, make_temporal_request
):
    """Full-output comparison against the brute-force oracle."""
    rng = random.Random(20261120)
    keys = [f"k{i:03d}" for i in range(40)]
    revisions = []
    for key in keys:
        n_windows = rng.randint(1, 4)
        for w in range(n_windows):
            wid = f"{key}-w{w}"
            moves = rng.randint(1, 5)
            # recorded_at values are unique per window (sorted then shuffled
            # into the global stream by the shard writer's shuffle).
            recs = sorted(rng.sample(range(0, 3000), moves))
            for m, rec in enumerate(recs):
                if m > 0 and rng.random() < 0.2:
                    revisions.append(rev_withdraw(key, wid, rec))
                else:
                    vf = rng.randint(0, 500)
                    vt = None if rng.random() < 0.2 else vf + rng.randint(1, 400)
                    revisions.append(rev_upsert(
                        key, wid, rec, vf, vt, rng.randint(-3, 5),
                        area=None if rng.random() < 0.1 else f"a{rng.randint(0, 9)}",
                    ))
        # A window that is never revised stays absent; also add revisions for
        # keys that may have no facts.
    rng.shuffle(revisions)
    facts = []
    fid = 0
    for key in keys:
        for _ in range(rng.randint(0, 6)):
            facts.append(fact_row(
                fid, key, rng.randint(0, 900), rng.randint(0, 3500),
            ))
            fid += 1
    # Duplicate a few facts verbatim (they must be preserved).
    assert len(facts) > 11
    facts += [facts[3], facts[3], facts[10]]
    rng.shuffle(facts)
    fact_manifests, positioned_facts = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA,
        n_shards=3, rows_per_group=17,
    )
    rev_manifests, positioned_revs = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        n_shards=2, rows_per_group=23,
    )
    client = temporal_client_factory()
    request = make_temporal_request(fact_manifests, rev_manifests)
    report = TemporalEnrichEngine(client, request).run()

    _, rows = _read_output(request.output_path)
    expected = oracle_enrich(positioned_facts, positioned_revs, DIM_COLUMNS)
    assert len(rows) == len(expected) == report["counts"]["facts"]
    counts = {"matched": 0, "ambiguous": 0, "unmatched": 0}
    for out_row, exp, (fact, _, _, _) in zip(rows, expected, positioned_facts):
        _check_row_against_oracle(out_row, exp, fact, FACT_COLUMNS, DIM_COLUMNS)
        counts[exp["status"]] += 1
    assert report["counts"]["matched"] == counts["matched"]
    assert report["counts"]["ambiguous"] == counts["ambiguous"]
    assert report["counts"]["unmatched"] == counts["unmatched"]
    # Determinism: the same run again produces byte-identical output.
    request2 = make_temporal_request(
        fact_manifests, rev_manifests, output_name="wide2.parquet"
    )
    TemporalEnrichEngine(client, request2).run()
    assert open(request.output_path, "rb").read() == open(
        request2.output_path, "rb").read()


def test_empty_inputs_produce_empty_wide_table(
    temporal_client_factory, temporal_server, tmp_path, make_temporal_request
):
    client = temporal_client_factory()
    request = make_temporal_request([], [])
    report = TemporalEnrichEngine(client, request).run()
    assert report["counts"]["facts"] == 0
    table = pq.read_table(request.output_path)
    assert table.num_rows == 0
    # The schema is complete even with zero rows.
    assert "_match_status" in table.column_names
    assert "_dim" in table.column_names


def test_unmatched_and_revision_only_keys(
    temporal_client_factory, temporal_server, tmp_path, make_temporal_request
):
    """Keys with only revisions are still validated; facts only -> unmatched."""
    facts = [fact_row(0, "only-facts", 5, 100)]
    revisions = [
        rev_upsert("only-revisions", "w", 1, 0, 10, 1),
        rev_upsert("only-revisions", "w", 2, 0, 20, 2),
    ]
    fact_manifests, _ = publish_shards(
        temporal_server, tmp_path, "facts", facts, FACT_SCHEMA,
        n_shards=1, rows_per_group=10,
    )
    rev_manifests, _ = publish_shards(
        temporal_server, tmp_path, "revisions", revisions, REVISION_SCHEMA,
        n_shards=1, rows_per_group=10,
    )
    client = temporal_client_factory()
    request = make_temporal_request(fact_manifests, rev_manifests)
    report = TemporalEnrichEngine(client, request).run()
    _, rows = _read_output(request.output_path)
    assert rows[0]["_match_status"] == "unmatched"
    assert report["counts"]["unmatched"] == 1
    m = report["resources"]["matching"]
    assert m["keys_fact_only"] == 1
    assert m["keys_revision_only"] == 1
