"""Resource statistics: physical requests, bytes on the wire, cache budgets.

Counters are thread-safe because the transport is driven from a thread pool.
"""

from __future__ import annotations

import dataclasses
import threading


@dataclasses.dataclass
class Stats:
    """Mutable counters shared by the transport and cache layers."""

    lock: threading.Lock = dataclasses.field(default_factory=threading.Lock, repr=False)

    # Physical HTTP layer
    physical_requests: int = 0          # GET requests actually sent
    head_requests: int = 0
    probe_requests: int = 0
    retries: int = 0
    retry_exhausted: int = 0
    status_412: int = 0
    status_other_errors: int = 0
    cancelled_requests: int = 0        # aborted because last waiter left

    # Bytes
    wire_bytes: int = 0                # bytes received in 206 bodies
    range_bytes_requested: int = 0     # sum of logical requested range sizes
    bytes_served_from_mem: int = 0
    bytes_served_from_disk: int = 0
    coalesced_hits: int = 0           # reads served by an in-flight physical GET

    # Cache state
    mem_bytes_cached: int = 0
    mem_evictions: int = 0
    mem_evicted_bytes: int = 0
    disk_bytes_cached: int = 0
    disk_evictions: int = 0
    disk_evicted_bytes: int = 0
    disk_fragments_rejected: int = 0  # leftover temp/evidence files on reopen

    # Budget rejections (should stay zero in normal operation)
    inflight_budget_waits: int = 0

    def snapshot(self) -> dict[str, int]:
        with self.lock:
            return {
                f.name: getattr(self, f.name)
                for f in dataclasses.fields(self)
                if f.name != "lock"
            }

    def inc(self, name: str, amount: int = 1) -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + amount)

    def add(self, name: str, amount: int) -> None:
        with self.lock:
            setattr(self, name, getattr(self, name) + amount)

    def render(self) -> str:
        s = self.snapshot()
        lines = ["Resource statistics:"]
        groups = [
            ("physical I/O", [
                ("head_requests", "HEAD requests"),
                ("probe_requests", "range probes"),
                ("physical_requests", "physical GET requests"),
                ("retries", "retried attempts"),
                ("retry_exhausted", "retry budgets exhausted"),
                ("status_412", "412 precondition failures"),
                ("status_other_errors", "other HTTP errors"),
                ("cancelled_requests", "cancelled physical requests"),
            ]),
            ("bytes", [
                ("range_bytes_requested", "logical range bytes requested"),
                ("wire_bytes", "bytes received over the wire"),
                ("bytes_served_from_mem", "bytes served from memory"),
                ("bytes_served_from_disk", "bytes served from disk"),
                ("coalesced_hits", "in-flight coalesced hits"),
            ]),
            ("cache", [
                ("mem_bytes_cached", "memory bytes currently cached"),
                ("mem_evictions", "memory block evictions"),
                ("mem_evicted_bytes", "memory bytes evicted"),
                ("disk_bytes_cached", "disk bytes currently cached"),
                ("disk_evictions", "disk block evictions"),
                ("disk_evicted_bytes", "disk bytes evicted"),
                ("disk_fragments_rejected", "disk fragments rejected on reopen"),
                ("inflight_budget_waits", "in-flight budget waits"),
            ]),
        ]
        for title, rows in groups:
            lines.append(f"  [{title}]")
            for key, label in rows:
                lines.append(f"    {label:38s} {s[key]:>14d}")
        return "\n".join(lines)
