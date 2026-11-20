"""Dynamic interval-stabbing-max structure over a compressed time axis.

During the per-key knowledge-time sweep, the set of *currently visible*
window episodes changes with every revision.  Each episode stabs a half-open
valid-time interval ``[valid_from, valid_to)`` with a priority; each fact
asks a point query ``event_at`` for:

* the maximum priority among stabbing episodes,
* how many distinct windows attain it (never enumerated), and
* the two smallest ``window_id`` values (UTF-8 byte order) among them -- all
  the provenance the ``ambiguous`` contract is allowed to surface.

This module implements exactly that aggregate with a compressed segment
tree:

* the coordinate list is every distinct ``valid_from`` / ``valid_to`` of the
  key's upsert episodes (collected in a first streaming pass), so the tree
  has ``len(coords) - 1`` elementary leaves and depth ``O(log len(coords))``;
* inserting/removing an interval touches ``O(log n)`` canonical nodes;
* each node keeps a lazy max-heap of ``(-priority, window_id)`` plus live
  membership and per-priority counts, so a node answers
  ``(max_priority, count_at_max, two_smallest_window_ids)`` in amortized
  ``O(log)`` without ever materializing the tied set;
* a point query walks one root-to-leaf path and combines ``O(log n)`` node
  answers.

Per-fact work is therefore ``O(log² n)`` -- never ``O(revisions of the
key)`` -- and the structure's size is ``O(active episodes · log n)``, metered
by the engine against the job's memory budget.
"""

from __future__ import annotations

import bisect
import heapq


class _NodeBag:
    """Live intervals stored at one segment-tree node.

    ``live`` maps ``window_id -> priority`` (a window appears at most once in
    a node: its current episode).  ``counts`` maps priority -> live count.
    ``heap`` is a lazy min-heap of ``(-priority, window_id)``; stale entries
    (from removals/re-prioritizations) are skipped on inspection, so the
    heap's top is always the highest live priority and, among entries at
    that priority, the smallest window_id in UTF-8 (== code-point) order.
    """

    __slots__ = ("live", "counts", "heap")

    def __init__(self) -> None:
        self.live: dict[str, int] = {}
        self.counts: dict[int, int] = {}
        self.heap: list[tuple[int, str]] = []

    def add(self, window_id: str, priority: int) -> None:
        self.live[window_id] = priority
        self.counts[priority] = self.counts.get(priority, 0) + 1
        heapq.heappush(self.heap, (-priority, window_id))

    def discard(self, window_id: str) -> None:
        priority = self.live.pop(window_id)
        self.counts[priority] -= 1
        # The heap entry is left behind and cleaned lazily by top().

    def top(self) -> tuple[int, int, tuple[str, str | None]] | None:
        """(max priority, count, two smallest window_ids) or None if empty."""
        heap = self.heap
        live = self.live
        while heap:
            negp, wid = heap[0]
            if live.get(wid) == -negp:
                break
            heapq.heappop(heap)
        if not heap:
            return None
        priority = -heap[0][0]
        count = self.counts[priority]
        first = heap[0][1]
        second: str | None = None
        if count > 1:
            held = heapq.heappop(heap)
            try:
                while heap:
                    negp, wid = heap[0]
                    if live.get(wid) != -negp:
                        heapq.heappop(heap)  # stale: drop and keep cleaning
                        continue
                    if -negp != priority:
                        break  # next live entry has a lower priority
                    second = wid
                    break
            finally:
                heapq.heappush(heap, held)
        return priority, count, (first, second)


def _combine(
    a: tuple[int, int, tuple[str, str | None]] | None,
    b: tuple[int, int, tuple[str, str | None]] | None,
) -> tuple[int, int, tuple[str, str | None]] | None:
    """Merge two node answers: max priority wins; ties sum counts/min2."""
    if a is None:
        return b
    if b is None:
        return a
    if a[0] > b[0]:
        return a
    if b[0] > a[0]:
        return b
    merged = sorted(w for w in (a[2][0], a[2][1], b[2][0], b[2][1])
                    if w is not None)[:2]
    return a[0], a[1] + b[1], (merged[0], merged[1] if len(merged) > 1 else None)


class StabbingMaxTree:
    """Segment tree over ``len(coords)`` elementary intervals.

    ``coords`` must be the sorted, deduplicated list of every interval
    endpoint the key will ever insert (open-ended intervals use
    ``valid_to=None`` and need no coordinate).  The elementary intervals
    are ``[coords[i], coords[i+1])`` for ``i < len(coords) - 1`` plus one
    overflow leaf ``[coords[-1], +inf)`` so that open-ended intervals and
    points beyond the last coordinate are covered correctly.  Node indices
    follow the classic iterative layout: leaves ``[n, 2n)``, root ``1``.
    """

    __slots__ = ("coords", "n", "bags", "entries", "peak_entries",
                 "inserts", "removes", "queries")

    def __init__(self, coords: list[int]) -> None:
        if len(coords) < 1:
            raise ValueError("need at least one coordinate")
        self.coords = coords
        # One elementary interval per coordinate boundary plus the overflow
        # leaf [coords[-1], +inf): n = len(coords).
        self.n = len(coords)
        self.bags: dict[int, _NodeBag] = {}
        self.entries = 0            # live (node, window) memberships
        self.peak_entries = 0
        self.inserts = 0            # episode insert operations
        self.removes = 0            # episode remove operations
        self.queries = 0            # point queries answered

    # ------------------------------------------------------------------ paths
    def _decompose(self, valid_from: int, valid_to: int | None):
        """Yield canonical node indices covering [valid_from, valid_to)."""
        coords = self.coords
        left = bisect.bisect_left(coords, valid_from)
        right = self.n if valid_to is None else bisect.bisect_left(coords, valid_to)
        n = self.n
        left += n
        right += n
        while left < right:
            if left & 1:
                yield left
                left += 1
            if right & 1:
                right -= 1
                yield right
            left >>= 1
            right >>= 1

    # -------------------------------------------------------------- mutations
    def insert(self, window_id: str, valid_from: int, valid_to: int | None,
               priority: int) -> None:
        for node in self._decompose(valid_from, valid_to):
            bag = self.bags.get(node)
            if bag is None:
                bag = self.bags[node] = _NodeBag()
            bag.add(window_id, priority)
            self.entries += 1
        self.inserts += 1
        if self.entries > self.peak_entries:
            self.peak_entries = self.entries

    def remove(self, window_id: str, valid_from: int, valid_to: int | None) -> None:
        for node in self._decompose(valid_from, valid_to):
            bag = self.bags.get(node)
            if bag is not None:
                bag.discard(window_id)
                self.entries -= 1
        self.removes += 1

    # ----------------------------------------------------------------- query
    def query(self, value: int) -> tuple[int, int, tuple[str, str | None]] | None:
        """Answer the stabbing-max aggregate for point ``value``."""
        self.queries += 1
        i = bisect.bisect_right(self.coords, value) - 1
        if i < 0 or i >= self.n:
            return None
        node = self.n + i
        best: tuple[int, int, tuple[str, str | None]] | None = None
        while node >= 1:
            bag = self.bags.get(node)
            if bag is not None:
                best = _combine(best, bag.top())
            node >>= 1
        return best
