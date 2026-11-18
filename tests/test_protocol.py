"""Protocol-violation and object-version isolation tests against the real server."""

from __future__ import annotations

import hashlib

import pytest

from parquet_range import (
    RangeProtocolError,
    RetryExhaustedError,
    TruncatedResponseError,
    VersionMismatchError,
    WeakOrMissingETagError,
)


# ------------------------------------------------------- first-contact failures
def test_missing_etag_fails_open(client, parquet_url, server):
    url, _, _ = parquet_url
    server.fault.missing_etag = True
    with pytest.raises(WeakOrMissingETagError):
        client.open(url)


def test_weak_etag_fails_open(client, parquet_url, server):
    url, _, _ = parquet_url
    server.fault.weak_etag = True
    with pytest.raises(WeakOrMissingETagError):
        client.open(url)


def test_no_range_support_fails_open(client, parquet_url, server):
    url, _, _ = parquet_url
    server.fault.no_range_support = True
    with pytest.raises(RangeProtocolError):
        client.open(url)


def test_200_fallback_on_probe_fails_open(client, parquet_url, server):
    url, _, _ = parquet_url
    server.fault.force_200 = True
    with pytest.raises(RangeProtocolError, match="200"):
        client.open(url)


# ------------------------------------------------------- 206 response violations
def test_bad_content_range_fails_query(client, parquet_url, server):
    url, _, _ = parquet_url
    obj = client.open(url)  # probe is clean
    server.fault.bad_content_range = True
    with pytest.raises(RangeProtocolError, match="Content-Range"):
        obj.scan(columns=["id"], row_groups=[0])


def test_truncated_body_then_retry_exhaustion(client_factory, parquet_url, server):
    # truncation is deterministic; bounded retries must give up, not loop forever
    client = client_factory(max_retries=2)
    url, _, _ = parquet_url
    obj = client.open(url)
    server.fault.truncate_body = 10
    # Either the Content-Length/Content-Range mismatch or the short body is
    # detected; both are explicit RangeProtocolError failures, never a 200 mix.
    with pytest.raises((TruncatedResponseError, RangeProtocolError, RetryExhaustedError)):
        obj.scan(columns=["id"], row_groups=[0])
    snap = client.stats.snapshot()
    # initial attempt + at most max_retries, never an unbounded loop
    assert snap["retries"] <= 2


def test_etag_mid_206_causes_version_mismatch(client, wide_parquet_url, server):
    """ETag changes between two Range GETs: the pinned query must fail.

    Uses the wide multi-block file so row group 2's blocks are guaranteed
    uncached and force fresh conditional GETs against the swapped object.
    """
    url, data, old_etag = wide_parquet_url
    obj = client.open(url)
    # First successful query populates only row group 0's blocks.
    t1 = obj.scan(columns=["id"], row_groups=[0])
    assert t1.num_rows == 4000
    # Server swaps to a structurally valid new version (same URL, new bytes/ETag).
    from tests.conftest import write_wide_parquet_v2

    new_data = write_wide_parquet_v2("/tmp/pqr_v2_wide.parquet")
    new_etag = '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"'
    server.swap_version("/wide.parquet", new_data, new_etag)
    # The still-open pinned object must now fail on its conditional GETs.
    with pytest.raises(VersionMismatchError):
        obj.scan(columns=["wide_11"], row_groups=[2])
    # No 200 fallback happened; the conditional header was really sent.
    gets = [e for e in server.log if e.path == "/wide.parquet" and e.method == "GET"]
    assert all(e.if_match == old_etag for e in gets if e.if_match)
    assert all(e.status in (206, 412) for e in gets)


def test_412_precondition_fails_query(client, parquet_url, server):
    url, _, _ = parquet_url
    obj = client.open(url)
    server.fault.force_412 = True
    with pytest.raises(VersionMismatchError, match="412"):
        obj.scan(columns=["id"], row_groups=[0])
    assert client.stats.snapshot()["status_412"] >= 1


def test_new_query_accepts_new_version_while_old_stays_isolated(
    client, parquet_url, server
):
    url, data, old_etag = parquet_url
    old = client.open(url)
    old.scan(columns=["id"], row_groups=[0])
    old_version = old.version_id

    new_data = data + b"\x00" * 17
    new_etag = '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"'
    server.swap_version("/sample.parquet", new_data, new_etag)

    # Re-opening the same URL surfaces the change explicitly instead of
    # silently mixing; open_fresh accepts the new version.
    with pytest.raises(VersionMismatchError):
        client.open(url)
    new = client.open_fresh(url)
    assert new.version_id != old_version
    assert new.identity.etag == new_etag

    # Both versions keep independent cache keys.
    from parquet_range.identity import ObjectIdentity  # noqa: F401
    assert old.identity.block_key(0) != new.identity.block_key(0)


def test_cached_old_bytes_not_stitched_after_swap(client, wide_parquet_url, server):
    url, data, _ = wide_parquet_url
    obj = client.open(url)
    t_before = obj.scan(columns=["id", "name"], row_groups=[0])
    cached_version = obj.identity.version_id()

    from tests.conftest import write_wide_parquet_v2

    new_data = write_wide_parquet_v2("/tmp/pqr_v2_wide_b.parquet")
    new_etag = '"' + hashlib.sha256(new_data).hexdigest()[:32] + '"'
    server.swap_version("/wide.parquet", new_data, new_etag)
    fresh = client.open_fresh(url)
    # New version has its own empty overlay: a cold range goes to the wire.
    before = client.stats.snapshot()["physical_requests"]
    fresh.scan(columns=["id", "name"], row_groups=[0])
    after = client.stats.snapshot()["physical_requests"]
    assert after > before
    # Old identity remains cached and readable without any new GET.
    wire_before = client.stats.snapshot()["wire_bytes"]
    t_again = obj.scan(columns=["id", "name"], row_groups=[0])
    wire_after = client.stats.snapshot()["wire_bytes"]
    assert t_again.equals(t_before)
    assert wire_after == wire_before
    assert obj.identity.version_id() == cached_version


# ------------------------------------------------------- bounded retries
def test_transient_5xx_recovers_within_retry_budget(client_factory, parquet_url, server):
    client = client_factory(max_retries=3)
    url, _, _ = parquet_url
    server.fault.fail_times = 2
    obj = client.open(url)
    table = obj.scan(columns=["id"], row_groups=[0])
    assert table.num_rows == 50
    snap = client.stats.snapshot()
    assert snap["retries"] >= 2
    assert snap["retry_exhausted"] == 0


def test_retry_exhaustion_is_bounded(client_factory, parquet_url, server):
    client = client_factory(max_retries=1)
    url, _, _ = parquet_url
    obj = client.open(url)
    server.fault.fail_times = 10
    with pytest.raises(RetryExhaustedError):
        obj.scan(columns=["id"], row_groups=[0])
    snap = client.stats.snapshot()
    assert snap["retries"] == 1
    assert snap["retry_exhausted"] == 1
