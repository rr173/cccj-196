"""Disk cache with evidence files and crash-fragment rejection.

For every cached block *B* covering [start, end) of one object version *V*
two files exist on disk::

    <root>/sha256(version).b<index>.blob       raw bytes
    <root>/sha256(version).b<index>.evidence   JSON + sha256 of the blob

A block is trusted on a fresh process only when both files exist, the
evidence JSON parses, identity (url/etag/length) matches, the declared range
matches, and the SHA-256 of the blob matches.  Anything else (partial temp
files, checksum mismatch, missing evidence) is quarantined/deleted and
counted as a rejected fragment, never silently served.

Writes are atomic: blob first to ``*.partial`` then ``os.replace``, evidence
second the same way.  A crash can therefore only leave ``.partial`` files or
an evidence-less blob -- both rejected on reopen.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import threading
from pathlib import Path

from .identity import ObjectIdentity
from .stats import Stats


@dataclasses.dataclass(frozen=True, slots=True)
class DiskBlock:
    start: int
    end: int
    path: Path


class DiskCache:
    def __init__(self, root: str | os.PathLike[str], stats: Stats) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.stats = stats
        self._lock = threading.Lock()
        # key (version_id, block_index) -> DiskBlock, guarded by _lock
        self._blocks: dict[tuple[str, int], DiskBlock] = {}
        # pin counters: blocks currently held by readers cannot be evicted
        self._pins: dict[tuple[str, int], int] = {}
        self._recover()

    # ---------------------------------------------------------------- paths
    def _blob_path(self, version_id: str, index: int) -> Path:
        return self.root / f"{version_id}.b{index:012d}.blob"

    def _evidence_path(self, version_id: str, index: int) -> Path:
        return self.root / f"{version_id}.b{index:012d}.evidence"

    def _partial_blob(self, version_id: str, index: int) -> Path:
        return self.root / f"{version_id}.b{index:012d}.blob.partial"

    def _partial_evidence(self, version_id: str, index: int) -> Path:
        return self.root / f"{version_id}.b{index:012d}.evidence.partial"

    # ------------------------------------------------------------- recovery
    def _recover(self) -> None:
        """Scan the cache directory; trust only fully evidenced blocks."""
        seen_ev: set[tuple[str, int]] = set()
        for path in sorted(self.root.iterdir()):
            name = path.name
            if name.endswith(".partial"):
                # Crash leftovers: never trusted.
                self._unlink(path)
                self.stats.inc("disk_fragments_rejected")
                continue
            if name.endswith(".evidence"):
                base = name[: -len(".evidence")]
                parsed = self._parse_base(base)
                if parsed is None:
                    self._unlink(path)
                    self.stats.inc("disk_fragments_rejected")
                    continue
                seen_ev.add(parsed)
                continue
            if name.endswith(".blob"):
                base = name[: -len(".blob")]
                parsed = self._parse_base(base)
                if parsed is None:
                    self._unlink(path)
                    self.stats.inc("disk_fragments_rejected")
                continue
            # Unknown file: leave alone but never serve.
        # Orphan blobs whose evidence never arrived (crash between phases):
        # they cannot be trusted and must be counted and removed.
        for path in list(self.root.glob("*.blob")):
            parsed = self._parse_base(path.name[: -len(".blob")])
            if parsed is not None and parsed not in seen_ev:
                self._unlink(path)
                self.stats.inc("disk_fragments_rejected")
        # Validate every evidenced block.
        for version_id, index in sorted(seen_ev):
            blob = self._blob_path(version_id, index)
            ev_path = self._evidence_path(version_id, index)
            try:
                evidence = json.loads(ev_path.read_text())
                if (
                    evidence.get("version_id") != version_id
                    or evidence.get("block_index") != index
                    or not isinstance(evidence.get("start"), int)
                    or not isinstance(evidence.get("end"), int)
                    or "sha256" not in evidence
                ):
                    raise ValueError("malformed evidence")
                h = hashlib.sha256()
                with blob.open("rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 20), b""):
                        h.update(chunk)
                if h.hexdigest() != evidence["sha256"]:
                    raise ValueError("checksum mismatch")
                if blob.stat().st_size != evidence["end"] - evidence["start"]:
                    raise ValueError("length mismatch")
            except (FileNotFoundError, ValueError, json.JSONDecodeError, OSError):
                self._unlink(blob)
                self._unlink(ev_path)
                self.stats.inc("disk_fragments_rejected")
                continue
            block = DiskBlock(evidence["start"], evidence["end"], blob)
            self._blocks[(version_id, index)] = block
            self.stats.add("disk_bytes_cached", block.end - block.start)

    @staticmethod
    def _parse_base(base: str) -> tuple[str, int] | None:
        # format: "<32 hex>.b<12 digits>"
        try:
            version_id, rest = base.split(".b", 1)
            if len(version_id) != 32:
                return None
            int(version_id, 16)
            index = int(rest)
        except (ValueError, AttributeError):
            return None
        return version_id, index

    @staticmethod
    def _unlink(path: Path) -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    # ------------------------------------------------------------ public API
    def get(self, identity: ObjectIdentity, index: int) -> bytes | None:
        key = (identity.version_id(), index)
        with self._lock:
            block = self._blocks.get(key)
            if block is None:
                return None
        # Read outside the lock; evidence was verified at recovery time.
        try:
            return block.path.read_bytes()
        except FileNotFoundError:  # pragma: no cover - external interference
            with self._lock:
                self._blocks.pop(key, None)
            return None

    def put(self, identity: ObjectIdentity, index: int, start: int, end: int,
            payload: bytes) -> None:
        if len(payload) != end - start:
            raise ValueError("disk cache payload length disagrees with range")
        version_id = identity.version_id()
        key = (version_id, index)
        blob = self._blob_path(version_id, index)
        ev = self._evidence_path(version_id, index)
        checksum = hashlib.sha256(payload).hexdigest()
        evidence = {
            "version_id": version_id,
            "block_index": index,
            "url": identity.url,
            "etag": identity.etag,
            "entity_length": identity.length,
            "start": start,
            "end": end,
            "sha256": checksum,
        }
        # Atomic two-phase write.
        pb = self._partial_blob(version_id, index)
        pe = self._partial_evidence(version_id, index)
        with pb.open("wb") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(pb, blob)
        with pe.open("w") as fh:
            json.dump(evidence, fh)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(pe, ev)
        with self._lock:
            old = self._blocks.get(key)
            self._blocks[key] = DiskBlock(start, end, blob)
        if old is not None:
            self.stats.add("disk_bytes_cached", -(old.end - old.start))
        self.stats.add("disk_bytes_cached", end - start)

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

    def evict_lru(self, need_bytes: int, pinned: set[tuple[str, int]]) -> int:
        """Evict unpinned blocks (oldest evidence mtime first) until budget.

        Returns bytes actually freed.
        """
        freed = 0
        with self._lock:
            candidates = [
                (block.path.stat().st_mtime, key, block)
                for key, block in self._blocks.items()
                if key not in self._pins and key not in pinned
            ]
        candidates.sort()
        for _, key, block in candidates:
            if freed >= need_bytes:
                break
            size = block.end - block.start
            self._unlink(block.path)
            self._unlink(self._evidence_path(key[0], key[1]))
            with self._lock:
                self._blocks.pop(key, None)
            self.stats.inc("disk_evictions")
            self.stats.add("disk_evicted_bytes", size)
            self.stats.add("disk_bytes_cached", -size)
            freed += size
        return freed

    def clear_version(self, identity: ObjectIdentity | None = None) -> int:
        """Remove blocks (optionally for one version). Returns bytes removed."""
        removed = 0
        with self._lock:
            items = list(self._blocks.items())
        for key, block in items:
            if identity is not None and key[0] != identity.version_id():
                continue
            if key in self._pins:  # never delete ranges still read
                continue
            size = block.end - block.start
            self._unlink(block.path)
            self._unlink(self._evidence_path(key[0], key[1]))
            with self._lock:
                self._blocks.pop(key, None)
            removed += size
            self.stats.add("disk_bytes_cached", -size)
        # Also sweep partial files.
        for path in self.root.glob("*.partial"):
            self._unlink(path)
        return removed

    def cached_bytes(self) -> int:
        with self._lock:
            return sum(b.end - b.start for b in self._blocks.values())
