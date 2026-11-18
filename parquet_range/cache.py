"""Block overlay cache + in-flight request scheduler.

Design
------
* The object is address space split into fixed-size blocks (default 1 MiB).
  Every cache key embeds the object version, so versions never share bytes.
* A block becomes a *readable overlay* only after a full validated payload
  arrives.  Partial/in-flight data is kept in the in-flight table, never in
  the readable overlay.
* Concurrent queries needing the same block join one physical GET
  (in-flight coalescing).  A query's cancellation drops only its own need;
  the physical GET continues while another waiter exists.  When the last
  waiter leaves the GET is aborted -- and a late packet can never mark the
  block complete.
* Adjacent missing blocks may be merged into one GET with a *bounded* gap:
  at most ``max_merge_gap`` blocks of hole may be bridged, so a huge empty
  span never gets downloaded to avoid an overlap.
* Budgets: total memory bytes, total disk bytes, max in-flight bytes and
  max in-flight requests.  Pinned blocks (held by active readers) are never
  evicted.
"""

from __future__ import annotations

import dataclasses
import threading
from collections import OrderedDict

from .disk_cache import DiskCache
from .exceptions import BudgetExceeded, QueryCancelled, TruncatedResponseError
from .identity import ObjectIdentity
from .stats import Stats
from .transport import HttpTransport


@dataclasses.dataclass(slots=True)
class _Inflight:
    identity: ObjectIdentity
    start_block: int
    end_block: int           # exclusive
    waiters: int
    event: threading.Event
    payload: bytes | None = None
    error: BaseException | None = None
    cancel: threading.Event = dataclasses.field(default_factory=threading.Event)
    bytes_held: int = 0      # counted against in-flight byte budget


class MemoryOverlay:
    """LRU byte-budgeted memory cache of complete validated blocks."""

    def __init__(self, stats: Stats, budget_bytes: int) -> None:
        self.stats = stats
        self.budget_bytes = budget_bytes
        self._lock = threading.Lock()
        self._data: OrderedDict[tuple[str, int], bytes] = OrderedDict()
        self._pins: dict[tuple[str, int], int] = {}

    def get(self, identity: ObjectIdentity, index: int) -> bytes | None:
        key = (identity.version_id(), index)
        with self._lock:
            payload = self._data.get(key)
            if payload is not None:
                self._data.move_to_end(key)  # LRU touch
            return payload

    def put(self, identity: ObjectIdentity, index: int, payload: bytes) -> None:
        key = (identity.version_id(), index)
        size = len(payload)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self.stats.add("mem_bytes_cached", -len(old))
            self._data[key] = payload
            self.stats.add("mem_bytes_cached", size)
            self._enforce_locked()

    def pin(self, identity: ObjectIdentity, index: int) -> None:
        key = (identity.version_id(), index)
        with self._lock:
            self._pins[key] = self._pins.get(key, 0) + 1

    def unpin(self, identity: ObjectIdentity, index: int) -> None:
        key = (identity.version_id(), index)
        with self._lock:
            n = self._pins[key] - 1
            if n <= 0:
                self._pins.pop(key, None)
            else:
                self._pins[key] = n

    def is_pinned(self, key: tuple[str, int]) -> bool:
        with self._lock:
            return key in self._pins

    def _enforce_locked(self) -> None:
        """Evict unpinned oldest blocks until within budget.

        Blocks pinned by active readers are skipped; if everything is pinned
        the budget may be exceeded transiently rather than corrupting a read.
        """
        while self.stats and self._mem_bytes_locked() > self.budget_bytes and self._data:
            for key in self._data:  # oldest first (OrderedDict)
                if key not in self._pins:
                    break
            else:
                return  # all blocks are in use; cannot evict
            payload = self._data.pop(key)
            self.stats.inc("mem_evictions")
            self.stats.add("mem_evicted_bytes", len(payload))
            self.stats.add("mem_bytes_cached", -len(payload))

    def _mem_bytes_locked(self) -> int:
        return sum(len(b) for b in self._data.values())

    def bytes_cached(self) -> int:
        with self._lock:
            return self._mem_bytes_locked()

    def clear(self, identity: ObjectIdentity | None = None) -> int:
        removed = 0
        with self._lock:
            for key in list(self._data):
                if identity is not None and key[0] != identity.version_id():
                    continue
                if key in self._pins:
                    continue
                payload = self._data.pop(key)
                removed += len(payload)
                self.stats.add("mem_bytes_cached", -len(payload))
        return removed


class BlockScheduler:
    """Owns overlay, disk tier, in-flight table, budgets and worker threads."""

    def __init__(
        self,
        transport: HttpTransport,
        stats: Stats,
        *,
        block_size: int = 1 << 20,
        mem_budget: int = 64 << 20,
        disk_budget: int = 512 << 20,
        max_inflight_bytes: int = 16 << 20,
        max_inflight_requests: int = 8,
        max_merge_gap_blocks: int = 2,
        disk_cache: DiskCache | None = None,
        workers: int = 8,
    ) -> None:
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.transport = transport
        self.stats = stats
        self.block_size = block_size
        self.disk_budget = disk_budget
        self.max_inflight_bytes = max_inflight_bytes
        self.max_inflight_requests = max_inflight_requests
        self.max_merge_gap = max_merge_gap_blocks
        self.mem = MemoryOverlay(stats, mem_budget)
        self.disk = disk_cache
        self._lock = threading.Lock()
        self._inflight: dict[tuple[str, int], _Inflight] = {}
        self._req_slots = threading.Semaphore(max_inflight_requests)
        self._byte_slots = threading.Semaphore(max_inflight_bytes)
        self._workers = workers
        self._pool = threading.local()
        self._threads: list[threading.Thread] = []
        self._closed = False

    # ------------------------------------------------------------- helpers
    def block_span(self, start: int, end: int) -> tuple[int, int]:
        """Return [first_block, last_block) covering byte range [start, end)."""
        return start // self.block_size, (end - 1) // self.block_size + 1

    def block_bounds(self, identity: ObjectIdentity, index: int) -> tuple[int, int]:
        start = index * self.block_size
        end = min(identity.length, start + self.block_size)
        return start, end

    # ------------------------------------------------------------- reading
    def read(
        self,
        identity: ObjectIdentity,
        start: int,
        length: int,
        *,
        token: "QueryToken | None" = None,
    ) -> bytes:
        """Read exactly ``length`` bytes at ``start`` for a pinned version."""
        if token is not None and token.cancelled:
            raise QueryCancelled("query was cancelled before read")
        if start < 0 or length < 0 or start + length > identity.length:
            raise IndexError(
                f"read [{start}, {start + length}) outside object of {identity.length}"
            )
        if length == 0:
            return b""
        first, last = self.block_span(start, start + length)
        out = bytearray()
        for index in range(first, last):
            if token is not None and token.cancelled:
                raise QueryCancelled("query was cancelled during read")
            block = self._read_block(identity, index, token)
            bstart, bend = self.block_bounds(identity, index)
            lo = max(start, bstart) - bstart
            hi = min(start + length, bend) - bstart
            out.extend(block[lo:hi])
        return bytes(out)

    def _read_block(
        self, identity: ObjectIdentity, index: int, token: "QueryToken | None"
    ) -> bytes:
        key = (identity.version_id(), index)
        # 1. memory overlay
        payload = self.mem.get(identity, index)
        if payload is not None:
            self.stats.inc("bytes_served_from_mem", len(payload))
            return payload
        # 2. disk tier
        if self.disk is not None:
            payload = self.disk.get(identity, index)
            if payload is not None:
                self.stats.inc("bytes_served_from_disk", len(payload))
                # Promote; put() handles LRU eviction.
                self.mem.put(identity, index, payload)
                return payload
        # 3. fetch (coalesced), with bounded adjacent merging
        return self._fetch_coalesced(identity, index, token)

    def _fetch_coalesced(
        self, identity: ObjectIdentity, index: int, token: "QueryToken | None"
    ) -> bytes:
        """Join an in-flight GET or start one, possibly merging neighbors."""
        key = (identity.version_id(), index)
        join: _Inflight | None = None
        starter = False
        with self._lock:
            existing = self._inflight.get(key)
            if existing is not None:
                existing.waiters += 1
                join = existing
            else:
                # Bounded merge: extend across adjacent *missing* blocks while
                # the hole stays within max_merge_gap blocks.
                end_block = index + 1
                gap = 0
                probe = index + 1
                while probe < self._nblocks(identity):
                    pkey = (identity.version_id(), probe)
                    if (
                        self.mem.get(identity, probe) is None
                        and (self.disk is None or self.disk.get(identity, probe) is None)
                        and pkey not in self._inflight
                    ):
                        gap += 1
                        if gap <= self.max_merge_gap:
                            end_block = probe + 1
                        else:
                            break
                    else:
                        break
                    probe += 1
                bstart = index * self.block_size
                bend = min(identity.length, end_block * self.block_size)
                inf = _Inflight(
                    identity=identity,
                    start_block=index,
                    end_block=end_block,
                    waiters=1,
                    event=threading.Event(),
                    bytes_held=bend - bstart,
                )
                for bi in range(index, end_block):
                    self._inflight[(identity.version_id(), bi)] = inf
                join = inf
                starter = True
        if not starter:
            self.stats.inc("coalesced_hits")
        else:
            self._launch(inf, bstart, bend)
        return self._wait_block(join, identity, index, token)

    def _drop_waiter(
        self, inf: _Inflight, identity: ObjectIdentity, index: int,
        token: "QueryToken",
    ) -> None:
        """Cancellation hook retained for external callers/tests."""
        token._drop_once(inf, identity, index)

    def _wait_block(
        self, inf: _Inflight, identity: ObjectIdentity, index: int,
        token: "QueryToken | None",
    ) -> bytes:
        if token is not None:
            token._register(inf, identity, index)
        try:
            while not inf.event.is_set():
                if inf.event.wait(timeout=0.05):
                    break
                if token is not None and token.cancelled:
                    token._drop_once(inf, identity, index)
                    raise QueryCancelled("query cancelled while waiting on block")
        finally:
            if token is not None:
                token._unregister(inf, identity, index)
        if inf.error is not None:
            raise inf.error
        assert inf.payload is not None
        bstart, bend = self.block_bounds(identity, index)
        lo = bstart - (inf.start_block * self.block_size)
        hi = lo + (bend - bstart)
        return inf.payload[lo:hi]

    # ------------------------------------------------------------- workers
    def _launch(self, inf: _Inflight, start: int, end: int) -> None:
        """Acquire in-flight budgets, then run the GET on a daemon thread."""
        size = end - start
        # Budget accounting: wait outside the global lock.
        self.stats.inc("inflight_budget_waits")
        self._req_slots.acquire()
        acquired = 0
        while acquired < size:
            chunk = min(1 << 20, size - acquired)
            self._byte_slots.acquire()
            acquired += chunk
        t = threading.Thread(
            target=self._run_physical,
            args=(inf, start, end),
            name=f"range-get-{start}",
            daemon=True,
        )
        with self._lock:
            if self._closed:
                self._release_budgets(size)
                inf.error = BudgetExceeded("scheduler closed")
                inf.event.set()
                return
            # Prune finished workers so the thread list cannot grow without bound.
            self._threads = [th for th in self._threads if th.is_alive()]
            self._threads.append(t)
        t.start()

    def _run_physical(self, inf: _Inflight, start: int, end: int) -> None:
        size = end - start
        try:
            payload = self.transport.get_range(
                inf.identity, start, end - 1, cancel_event=inf.cancel
            )
            # A cancellation can still land after a fully received body; only
            # publish when at least one waiter remains and data is complete.
            with self._lock:
                if inf.cancel.is_set() or inf.waiters <= 0:
                    inf.error = QueryCancelled("last waiter left; result discarded")
                    inf.event.set()
                    return
                if len(payload) != size:  # defensive; transport already checks
                    inf.error = TruncatedResponseError("internal length mismatch")
                    inf.event.set()
                    return
                inf.payload = payload
            # Publish (memory + disk overlays) BEFORE waking waiters, so a
            # returning reader or close() never sees an unfinished overlay.
            self._publish(inf, payload)
            with self._lock:
                inf.event.set()
        except BaseException as exc:  # noqa: BLE001 - surface to all waiters
            with self._lock:
                inf.error = exc
                inf.event.set()
        finally:
            self._release_budgets(size)
            with self._lock:
                for bi in range(inf.start_block, inf.end_block):
                    self._inflight.pop((inf.identity.version_id(), bi), None)

    def _release_budgets(self, size: int) -> None:
        self._req_slots.release()
        remaining = size
        while remaining:
            chunk = min(1 << 20, remaining)
            self._byte_slots.release()
            remaining -= chunk

    def _publish(self, inf: _Inflight, payload: bytes) -> None:
        """Split a coalesced GET payload into complete validated block overlays."""
        base = inf.start_block * self.block_size
        for bi in range(inf.start_block, inf.end_block):
            bstart, bend = self.block_bounds(inf.identity, bi)
            part = payload[bstart - base: bend - base]
            # A block is complete when it reaches its planned bounds; the
            # final block of an object is legitimately shorter than block_size.
            if len(part) != bend - bstart:
                continue
            self.mem.put(inf.identity, bi, part)
            if self.disk is None:
                continue
            size = len(part)
            if self.disk.cached_bytes() + size > self.disk_budget:
                # Make room via unpinned LRU eviction on disk, then retry once.
                need = self.disk.cached_bytes() + size - self.disk_budget
                self.disk.evict_lru(need, pinned=set())
            try:
                self.disk.put(inf.identity, bi, bstart, bend, part)
            except OSError:
                # Disk is a best-effort tier; the memory overlay stays correct.
                pass

    def _nblocks(self, identity: ObjectIdentity) -> int:
        return (identity.length + self.block_size - 1) // self.block_size

    # ------------------------------------------------------------- lifecycle
    def close(self) -> None:
        """Signal cancellation to every in-flight GET and wait for workers.

        Joining guarantees that by the time close() returns no worker is still
        writing a disk overlay, so a reopened process can trust the evidence
        files and no ``.partial`` fragment is left behind by a clean shutdown.
        """
        with self._lock:
            if self._closed:
                threads = []
            else:
                self._closed = True
                for inf in self._inflight.values():
                    inf.cancel.set()
                threads = list(self._threads)
        for thread in threads:
            if thread is not threading.current_thread():
                thread.join(timeout=30.0)

    def clear_caches(self, identity: ObjectIdentity | None = None) -> dict[str, int]:
        mem_freed = self.mem.clear(identity)
        disk_freed = self.disk.clear_version(identity) if self.disk else 0
        return {"memory_bytes_freed": mem_freed, "disk_bytes_freed": disk_freed}


class QueryToken:
    """Per-query cancellation handle.

    Tracks the in-flight requests this query is waiting on so cancellation
    removes exactly its own waiters.  A physical request keeps running while
    any other query still depends on it; when the last waiter leaves, the
    scheduler aborts the transport and any late packet is discarded instead
    of being published as a complete overlay.
    """

    def __init__(self, scheduler: "BlockScheduler") -> None:
        self.scheduler = scheduler
        self.cancelled = False
        self._lock = threading.Lock()
        self._waiting: list[tuple[_Inflight, ObjectIdentity, int]] = []
        # In-flights for which this token already dropped its waiter, so a
        # racing cancel()/wait-loop cannot decrement the waiter count twice.
        self._dropped: set[int] = set()

    def cancel(self) -> None:
        with self._lock:
            if self.cancelled:
                return
            self.cancelled = True
            waiting = list(self._waiting)
        # Drop exactly this query's waiter counts.  The scheduler aborts the
        # physical GET only when no other query is still waiting.
        for inf, identity, index in waiting:
            self._drop_once(inf, identity, index)

    def _drop_once(self, inf: _Inflight, identity: ObjectIdentity, index: int) -> None:
        with self._lock:
            if id(inf) in self._dropped:
                return
            self._dropped.add(id(inf))
            inf.waiters -= 1
            last = inf.waiters <= 0
        if last:
            inf.cancel.set()  # abort transport; late packets are discarded

    def _register(self, inf: _Inflight, identity: ObjectIdentity, index: int) -> None:
        with self._lock:
            self._waiting.append((inf, identity, index))

    def _unregister(self, inf: _Inflight, identity: ObjectIdentity, index: int) -> None:
        with self._lock:
            try:
                self._waiting.remove((inf, identity, index))
            except ValueError:
                pass
