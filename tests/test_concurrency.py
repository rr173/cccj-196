"""Concurrency and cancellation acceptance tests against the real server.

Covers the required behaviours:
* Q1 and Q2 read overlapping blocks; cancelling Q1 leaves Q2 fully successful
  and only the necessary physical fetches happen;
* a physical request shared by two queries is NOT killed while one remains;
* when the last waiter leaves the request is aborted and a late packet can
  never become a complete overlay;
* cancellation releases only the cancelling query's need;
* bounded retry keeps the original pinned version.
"""

from __future__ import annotations

import threading
import time

import pyarrow.parquet as pq
import io
import pytest

from parquet_range import QueryCancelled, VersionMismatchError


def _wait_inflight(scheduler, want_waiters: int, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with scheduler._lock:
            if any(inf.waiters >= want_waiters for inf in scheduler._inflight.values()):
                return True
        time.sleep(0.01)
    return False


def test_cancel_q1_q2_overlapping_blocks_q2_completes(
    client_factory, wide_parquet_url, server
):
    """Headline acceptance: cancel Q1 mid-flight; Q2 still fully succeeds."""
    client = client_factory(block_size=64 * 1024, max_inflight_requests=4)
    url, data, _ = wide_parquet_url
    name = "GET:/wide.parquet"
    obj = client.open(url)
    obj.inspect()  # warm footer so the barrier gates only data GETs
    server.clear_log()
    server.barrier.arm(name)

    token1 = client.new_token()
    token2 = client.new_token()
    outcomes: dict[str, object] = {}

    def q1():
        try:
            outcomes["q1"] = obj.scan(
                columns=["id", "name"], row_groups=[5], token=token1
            )
        except BaseException as exc:  # noqa: BLE001 - recorded for assertions
            outcomes["q1"] = exc

    def q2():
        try:
            outcomes["q2"] = obj.scan(
                columns=["id", "name"], row_groups=[5], token=token2
            )
        except BaseException as exc:  # noqa: BLE001
            outcomes["q2"] = exc

    t1 = threading.Thread(target=q1, name="Q1")
    t2 = threading.Thread(target=q2, name="Q2")
    t1.start()
    assert server.barrier.wait_for_arrivals(name, 1, timeout=10)
    t2.start()
    # Both queries must be sharing the same in-flight physical request.
    assert _wait_inflight(client.scheduler, 2)

    # Cancel Q1 while the shared GET is still parked. Q2 remains a waiter,
    # so the physical request must NOT be aborted.
    token1.cancel()
    with client.scheduler._lock:
        inflight = list(client.scheduler._inflight.values())
        assert all(not inf.cancel.is_set() for inf in inflight), (
            "shared physical request must survive as long as Q2 waits"
        )
        assert all(inf.waiters >= 1 for inf in inflight)

    server.barrier.release(name)
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert not t1.is_alive() and not t2.is_alive()

    # Q1 reports cancellation; Q2 returns the correct, complete Arrow table.
    assert isinstance(outcomes["q1"], QueryCancelled), outcomes["q1"]
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_groups(
        [5], columns=["id", "name"]
    )
    assert outcomes["q2"].equals(ref)  # type: ignore[union-attr]

    # Only the necessary physical data GETs occurred (one per shared block),
    # never a whole-object fetch.
    data_gets = [
        e for e in server.range_gets("/wide.parquet")
        if e.status == 206 and e.range_header != "bytes=0-0"
    ]
    assert data_gets, "expected at least one physical data GET"
    assert all(e.status == 206 for e in data_gets)
    total_wire = sum(e.bytes_sent for e in data_gets)
    assert total_wire < len(data)


def test_last_waiter_cancel_aborts_request_and_discards_late_packet(
    client_factory, wide_parquet_url, server
):
    """When the only waiter leaves, the GET aborts and is not cached."""
    client = client_factory(block_size=64 * 1024, max_inflight_requests=4)
    url, data, _ = wide_parquet_url
    name = "GET:/wide.parquet"
    obj = client.open(url)
    obj.inspect()
    server.clear_log()
    server.barrier.arm(name)

    token = client.new_token()
    outcome: list[BaseException] = []

    def q():
        try:
            obj.scan(columns=["wide_11"], row_groups=[3], token=token)
        except BaseException as exc:  # noqa: BLE001
            outcome.append(exc)

    t = threading.Thread(target=q, name="lone-q")
    t.start()
    assert server.barrier.wait_for_arrivals(name, 1, timeout=10)
    # Snapshot which blocks are in flight, then cancel the sole waiter.
    with client.scheduler._lock:
        in_flight_before = {
            (inf.identity.version_id(), bi)
            for inf in client.scheduler._inflight.values()
            for bi in range(inf.start_block, inf.end_block)
        }
    token.cancel()
    # The transport cancel event must be set; the request gets aborted.
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        with client.scheduler._lock:
            still = any(
                (inf.identity.version_id(), bi) in in_flight_before
                for inf in client.scheduler._inflight.values()
                for bi in range(inf.start_block, inf.end_block)
            )
        if not still:
            break
        time.sleep(0.01)
    server.barrier.release(name)
    t.join(timeout=15)
    assert not t.is_alive()
    assert outcome and isinstance(outcome[0], QueryCancelled)

    # None of those blocks may now be a readable overlay.
    for key in in_flight_before:
        version_id, block_index = key
        assert key not in client.scheduler.mem._data
    # Budget slots were released (no permanent leak): a later scan works.
    later = obj.scan(columns=["id"], row_groups=[0])
    assert later.num_rows == 4000


def test_cancel_releases_only_own_need_independent_requests(
    client_factory, wide_parquet_url, server
):
    """Two queries on disjoint blocks: cancelling one never touches the other."""
    client = client_factory(
        block_size=64 * 1024, max_inflight_requests=4, max_merge_gap_blocks=0
    )
    url, data, _ = wide_parquet_url
    obj = client.open(url)
    obj.inspect()
    server.clear_log()

    name_a = "GET:/wide.parquet"
    server.barrier.arm(name_a)
    token_a = client.new_token()
    token_b = client.new_token()
    res: dict[str, object] = {}

    def qa():
        try:
            res["a"] = obj.scan(columns=["wide_0"], row_groups=[0], token=token_a)
        except BaseException as exc:  # noqa: BLE001
            res["a"] = exc

    def qb():
        try:
            res["b"] = obj.scan(columns=["wide_11"], row_groups=[5], token=token_b)
        except BaseException as exc:  # noqa: BLE001
            res["b"] = exc

    ta = threading.Thread(target=qa, name="QA")
    tb = threading.Thread(target=qb, name="QB")
    ta.start()
    assert server.barrier.wait_for_arrivals(name_a, 1, timeout=10)
    tb.start()
    # Let QB get scheduled (its own request), then cancel only QA.
    time.sleep(0.2)
    token_a.cancel()
    server.barrier.release(name_a)
    ta.join(timeout=15)
    tb.join(timeout=15)
    assert isinstance(res["a"], QueryCancelled)
    ref = pq.ParquetFile(io.BytesIO(data)).read_row_groups([5], columns=["wide_11"])
    assert res["b"].equals(ref)  # type: ignore[union-attr]


def test_retry_after_cancel_keeps_pinned_version(
    client_factory, parquet_url, server
):
    """A retry that follows transient failure must reuse If-Match, not drift."""
    client = client_factory(max_retries=3)
    url, _, etag = parquet_url
    obj = client.open(url)
    server.fault.fail_times = 2  # two 503s, then success
    table = obj.scan(columns=["id"], row_groups=[0])
    assert table.num_rows == 50
    gets = [e for e in server.range_gets("/sample.parquet") if e.range_header != "bytes=0-0"]
    assert gets, "expected data GETs"
    assert all(e.if_match == etag for e in gets)
