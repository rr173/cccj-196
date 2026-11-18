"""Cache overlay, disk persistence/evidence and LRU eviction tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest

from parquet_range import ClientConfig, RemoteParquetClient
from parquet_range.disk_cache import DiskCache
from parquet_range.identity import ObjectIdentity
from parquet_range.stats import Stats


def _identity(url="http://h/f", etag='"v1"', length=10_000) -> ObjectIdentity:
    return ObjectIdentity(url, etag, length)


# ------------------------------------------------------------- disk evidence
def test_disk_cache_roundtrip(tmp_path):
    stats = Stats()
    dc = DiskCache(tmp_path / "cache", stats)
    ident = _identity()
    dc.put(ident, 3, 3 * 4096, 4 * 4096, b"x" * 4096)
    # evidence + blob both present
    files = {p.name for p in Path(dc.root).iterdir()}
    assert any(n.endswith(".blob") for n in files)
    assert any(n.endswith(".evidence") for n in files)
    # Fresh process: recovery verifies checksum before trusting anything.
    dc2 = DiskCache(tmp_path / "cache", Stats())
    assert dc2.get(ident, 3) == b"x" * 4096


def test_disk_cache_rejects_corrupt_blob(tmp_path):
    stats = Stats()
    root = tmp_path / "cache"
    dc = DiskCache(root, stats)
    ident = _identity()
    dc.put(ident, 1, 4096, 8192, b"y" * 4096)
    # Corrupt the blob behind the cache's back.
    blob = next(root.glob("*.blob"))
    blob.write_bytes(b"z" * 4096)
    dc2 = DiskCache(root, Stats())
    assert dc2.get(ident, 1) is None
    assert dc2.stats.snapshot()["disk_fragments_rejected"] >= 1
    # both evidence and blob were quarantined
    assert list(root.glob("*.blob")) == []
    assert list(root.glob("*.evidence")) == []


def test_disk_cache_rejects_partial_and_evidence_less(tmp_path):
    root = tmp_path / "cache"
    dc = DiskCache(root, Stats())
    ident = _identity()
    dc.put(ident, 2, 0, 100, b"a" * 100)
    # Simulate a power loss mid-write: leftover .partial + orphan blob.
    (root / f"{ident.version_id()}.b000000000009.blob.partial").write_bytes(b"p" * 50)
    (root / f"{ident.version_id()}.b000000000008.blob").write_bytes(b"orphan")
    dc2 = DiskCache(root, Stats())
    assert dc2.get(ident, 9) is None
    assert dc2.get(ident, 8) is None
    assert dc2.stats.snapshot()["disk_fragments_rejected"] >= 2
    # the good block survived
    assert dc2.get(ident, 2) == b"a" * 100


def test_disk_cache_version_isolation(tmp_path):
    dc = DiskCache(tmp_path / "cache", Stats())
    v1 = _identity(etag='"v1"')
    v2 = _identity(etag='"v2"')
    dc.put(v1, 0, 0, 10, b"0123456789")
    dc.put(v2, 0, 0, 10, b"9876543210")
    assert dc.get(v1, 0) == b"0123456789"
    assert dc.get(v2, 0) == b"9876543210"


# ------------------------------------------------------------- end-to-end persistence
def test_scan_cache_hit_survives_process_restart(tmp_path, parquet_url, client_factory):
    url, data, _ = parquet_url
    disk_dir = str(tmp_path / "persist")
    c1 = client_factory(disk_dir=disk_dir)
    obj1 = c1.open(url)
    t1 = obj1.scan(columns=["id", "meta"], row_groups=[0, 1])
    assert c1.stats.snapshot()["physical_requests"] > 0
    c1.close()

    # Brand-new client/process: memory is empty, disk must be verified then used.
    c2 = client_factory(disk_dir=disk_dir)
    obj2 = c2.open(url)
    t2 = obj2.scan(columns=["id", "meta"], row_groups=[0, 1])
    assert t1.equals(t2)
    snap = c2.stats.snapshot()
    assert snap["physical_requests"] == 0  # only HEAD/probe validation, no data GET
    assert snap["bytes_served_from_disk"] > 0


def test_disk_evidence_tampering_invalidates_block(tmp_path, parquet_url, client_factory):
    url, data, _ = parquet_url
    disk_dir = str(tmp_path / "tamper")
    c1 = client_factory(disk_dir=disk_dir)
    obj = c1.open(url)
    obj.scan(columns=["id"], row_groups=[0])
    c1.close()
    root = Path(disk_dir)
    evidence = next(root.glob("*.evidence"))
    ev = json.loads(evidence.read_text())
    ev["sha256"] = "0" * 64
    evidence.write_text(json.dumps(ev))
    c2 = client_factory(disk_dir=disk_dir)
    obj2 = c2.open(url)
    # Tampered evidence => rejected => block refetched over the wire, not trusted.
    obj2.scan(columns=["id"], row_groups=[0])
    assert c2.stats.snapshot()["disk_fragments_rejected"] >= 1
    assert c2.stats.snapshot()["physical_requests"] >= 1


# ------------------------------------------------------------- memory LRU
def test_memory_lru_eviction_and_budget(client_factory, wide_parquet_url):
    # Tiny 2-block memory budget forces eviction; correctness must survive.
    client = client_factory(mem_budget=2 * 64 * 1024, block_size=64 * 1024)
    url, data, _ = wide_parquet_url
    obj = client.open(url)
    t = obj.scan(columns=["id", "name"], row_groups=[0, 1, 2])
    ref = pq.ParquetFile(__import__("io").BytesIO(data)).read_row_groups(
        [0, 1, 2], columns=["id", "name"]
    )
    assert t.equals(ref)
    snap = client.stats.snapshot()
    assert snap["mem_evictions"] > 0
    # cached memory must never exceed budget by more than one in-flight block
    assert snap["mem_bytes_cached"] <= 2 * 64 * 1024 + 64 * 1024


def test_pinned_blocks_are_not_evicted(client_factory, wide_parquet_url):
    client = client_factory(block_size=64 * 1024, mem_budget=3 * 64 * 1024)
    url, _, _ = wide_parquet_url
    obj = client.open(url)
    # Pin blocks manually like an active reader would, then flood the cache.
    ident = obj.identity
    scheduler = client.scheduler
    pinned_indices = []
    first, last = scheduler.block_span(0, 2 * 64 * 1024)
    for idx in range(first, last):
        scheduler.mem.put(ident, idx, bytes([idx]) * 64 * 1024)
        scheduler.mem.pin(ident, idx)
        pinned_indices.append(idx)
    for idx in range(last, last + 20):
        end = min(ident.length, (idx + 1) * 64 * 1024)
        scheduler.mem.put(ident, idx, bytes([idx % 256]) * (end - idx * 64 * 1024))
    for idx in pinned_indices:
        assert scheduler.mem.get(ident, idx) is not None


def test_eviction_while_reader_active_keeps_pinned_ranges(client_factory, wide_parquet_url):
    """A range held by an active reader cannot be reclaimed from either tier."""
    client = client_factory(
        block_size=64 * 1024, mem_budget=3 * 64 * 1024, disk_budget=4 * 64 * 1024
    )
    url, _, _ = wide_parquet_url
    obj = client.open(url)
    ident = obj.identity
    sched = client.scheduler

    # Establish and pin the first two blocks (simulate an in-progress reader).
    sched.read(ident, 0, 2 * 64 * 1024)
    for idx in range(2):
        sched.mem.pin(ident, idx)
        client.disk.pin(ident, idx)
    pinned_mem_before = sched.mem.get(ident, 0)
    pinned_disk_before = client.disk.get(ident, 0)
    assert pinned_mem_before is not None and pinned_disk_before is not None

    # Flood both tiers well past their budgets.
    for idx in range(2, 30):
        end = min(ident.length, (idx + 1) * 64 * 1024)
        sched.mem.put(ident, idx, bytes([idx % 256]) * (end - idx * 64 * 1024))
        client.disk.evict_lru(1 << 30, pinned={(ident.version_id(), idx) for idx in range(2)})

    # The pinned ranges survive in both tiers; unpinned ones may be gone.
    assert sched.mem.get(ident, 0) == pinned_mem_before
    assert sched.mem.get(ident, 1) is not None
    assert client.disk.get(ident, 0) == pinned_disk_before
    for idx in range(2):
        sched.mem.unpin(ident, idx)
        client.disk.unpin(ident, idx)


# ------------------------------------------------------------- cache clear
def test_cache_clear_frees_unpinned(client, parquet_url, tmp_path):
    url, _, _ = parquet_url
    obj = client.open(url)
    obj.scan(columns=["id"], row_groups=[0])
    freed = client.cache_clear(url)
    assert freed["memory_bytes_freed"] > 0 or freed["disk_bytes_freed"] > 0
    snap = client.stats.snapshot()
    assert snap["mem_bytes_cached"] == 0
    assert (client.disk.cached_bytes() if client.disk else 0) == 0


def test_coalesced_concurrent_reads_share_one_physical_get(
    client_factory, wide_parquet_url, server
):
    import threading
    import time

    client = client_factory(block_size=64 * 1024, max_inflight_requests=4)
    url, data, _ = wide_parquet_url
    name = "GET:/wide.parquet"
    obj = client.open(url)
    # Warm + cache the footer first so the barrier only gates column data GETs.
    obj.inspect()

    # Pick an uncached row group (4) and its first block's in-flight key.
    server.clear_log()
    server.barrier.arm(name)
    errors: list[BaseException] = []
    results: dict[str, object] = {}

    def run(key: str, row_group: int) -> None:
        try:
            results[key] = obj.scan(columns=["id"], row_groups=[row_group])
        except BaseException as exc:  # captured, never silently swallowed
            errors.append(exc)

    t1 = threading.Thread(target=run, args=("q1", 4))
    t2 = threading.Thread(target=run, args=("q2", 4))
    t1.start()
    # The single shared physical GET parks in the server barrier.
    assert server.barrier.wait_for_arrivals(name, 1, timeout=10), "Q1 GET never arrived"

    t2.start()
    # Wait until Q2 has joined Q1's in-flight request (two waiters, one GET).
    deadline = time.monotonic() + 10
    joined = False
    while time.monotonic() < deadline:
        with client.scheduler._lock:
            joined = any(inf.waiters >= 2 for inf in client.scheduler._inflight.values())
        if joined:
            break
        time.sleep(0.01)
    assert joined, "Q2 never coalesced onto Q1's in-flight GET"

    server.barrier.release(name)  # one-shot: later block GETs pass through
    t1.join(timeout=15)
    t2.join(timeout=15)
    assert not t1.is_alive() and not t2.is_alive()
    assert errors == []
    assert results["q1"].equals(results["q2"])  # type: ignore[union-attr]

    physical = [
        e for e in server.range_gets("/wide.parquet")
        if e.status == 206 and e.range_header != "bytes=0-0"
    ]
    # Both queries read the same blocks yet only one physical data GET per block.
    assert len(physical) == 1, [e.range_header for e in physical]
    assert client.stats.snapshot()["coalesced_hits"] >= 1
