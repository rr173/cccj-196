"""Bounded-memory external sort for temporal-enrich intermediate records.

The temporal engine never assumes fact/revision shards are ordered and never
holds a whole role (or a whole hot key) in memory.  Extraction feeds records
to a :class:`RunBuilder`; once the buffered bytes cross the job's memory
ceiling the buffer is sorted by its binary sort key and spilled as a framed,
checksummed *segment*; a run is then either one in-memory sorted table or a
bounded-fan-in external merge of segments.

Segment layout (a crash can only ever leave a whole unverified file)::

    PQRTMP01\n
    <u64 BE header-json-length><header JSON utf-8>
    <Arrow IPC stream: schema + sorted record batches>
    <u64 BE trailer-json-length><trailer JSON: counts + sha256 of the IPC body>
    PQRTMP01\n

The header records the format version, the owning run id, the record role
(``revision`` / ``fact`` / ``result``), the Arrow schema (digest + text) and
the job's full shard-identity map, so a temporary segment can never be
mistaken for trusted input by a later job: readers re-verify framing, run
ownership, schema digest, record counts and the body SHA-256, and stale
segments of other runs are removed or quarantined at startup.
"""

from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import os
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc

from .exceptions import SegmentIntegrityError

SEGMENT_MAGIC = b"PQRTMP01\n"
SEGMENT_FORMAT = "pqr-temporal-segment"
SEGMENT_FORMAT_VERSION = 1
PKEY_FIELD = "_pkey"

# Rows per IPC message produced by an external merge pass (bounded, so a
# merge pass never materialises the whole merged run into memory at once).
MERGE_WRITE_BATCH = 512
# Maximum number of segment streams merged in one pass; bounds open handles.
MERGE_FAN_IN = 8


def schema_digest(schema: pa.Schema) -> str:
    """Stable digest of an Arrow schema (types + field order + nullability)."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, schema) as _:
        pass
    return hashlib.sha256(sink.getvalue()).hexdigest()


@dataclasses.dataclass(slots=True)
class SegmentMeta:
    path: Path
    role: str
    record_count: int
    run_id: str


class SegmentWriter:
    """Writes one framed, checksummed, sorted segment."""

    def __init__(self) -> None:  # pragma: no cover - use the factory
        raise RuntimeError("use open_segment_writer()")

    def _init(
        self,
        directory: Path,
        role: str,
        schema: pa.Schema,
        *,
        run_id: str,
        shard_identities: dict,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.role = role
        self.schema = schema
        self.run_id = run_id
        self.path = self.directory / f"{run_id}.{role}.{uuid.uuid4().hex}.pqrtseg"
        self._fh = self.path.open("wb")
        self._fh.write(SEGMENT_MAGIC)
        self._digest = hashlib.sha256()
        self.record_count = 0
        self.batch_count = 0
        header = {
            "format": SEGMENT_FORMAT,
            "format_version": SEGMENT_FORMAT_VERSION,
            "role": role,
            "run_id": run_id,
            "schema_digest": schema_digest(schema),
            "schema": str(schema),
            "shard_identities": shard_identities,
            "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        header_bytes = json.dumps(header, sort_keys=True).encode("utf-8")
        self._fh.write(len(header_bytes).to_bytes(8, "big"))
        self._fh.write(header_bytes)
        self._fh.flush()
        self._hashing = pa.PythonFile(_HashingFile(self._fh, self._digest), mode="w")
        self._writer = pa.ipc.new_stream(self._hashing, schema)

    def write_table(self, table: pa.Table) -> None:
        if table.schema != self.schema:
            raise SegmentIntegrityError("segment batch schema disagrees with header")
        self._writer.write_table(table)
        self.record_count += table.num_rows
        self.batch_count += 1

    def close(self) -> SegmentMeta:
        self._writer.close()
        trailer = {
            "record_count": self.record_count,
            "batch_count": self.batch_count,
            "body_sha256": self._digest.hexdigest(),
        }
        trailer_bytes = json.dumps(trailer, sort_keys=True).encode("utf-8")
        self._fh.write(trailer_bytes)
        self._fh.write(len(trailer_bytes).to_bytes(8, "big"))
        self._fh.write(SEGMENT_MAGIC)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        return SegmentMeta(
            path=self.path,
            role=self.role,
            record_count=self.record_count,
            run_id=self.run_id,
        )


class _HashingFile(io.RawIOBase):
    """Native-file-compatible wrapper hashing every byte written through it."""

    def __init__(self, raw, digest) -> None:
        super().__init__()
        self._raw = raw
        self._digest = digest
        self._closed = False

    def writable(self) -> bool:
        return True

    def write(self, data) -> int:
        if isinstance(data, memoryview):
            data = bytes(data)
        self._digest.update(data)
        return self._raw.write(data)

    def flush(self) -> None:
        self._raw.flush()

    def fileno(self) -> int:
        return self._raw.fileno()

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            super().close()

    @property
    def closed(self) -> bool:  # type: ignore[override]
        return self._closed


def open_segment_writer(
    directory: Path,
    role: str,
    schema: pa.Schema,
    *,
    run_id: str,
    shard_identities: dict,
) -> SegmentWriter:
    """Factory wiring the hashing wrapper before Arrow opens the IPC stream."""
    sw = SegmentWriter.__new__(SegmentWriter)
    sw._init(directory, role, schema, run_id=run_id,
             shard_identities=shard_identities)
    return sw


class SegmentReader:
    """Verifies framing/checksum/run ownership, then yields sorted batches."""

    def __init__(self, meta: SegmentMeta, expected_schema: pa.Schema,
                 *, expected_role: str, expected_run_id: str) -> None:
        self.meta = meta
        self.expected_schema = expected_schema
        self.expected_role = expected_role
        self.expected_run_id = expected_run_id
        self._verify()

    def _read_exact(self, fh, n: int, what: str) -> bytes:
        chunk = fh.read(n)
        if len(chunk) != n:
            raise SegmentIntegrityError(
                f"segment {self.meta.path.name}: truncated while reading {what} "
                f"({len(chunk)}/{n} bytes)"
            )
        return chunk

    def _verify(self) -> None:
        path = self.meta.path
        try:
            fh = path.open("rb")
        except OSError as exc:
            raise SegmentIntegrityError(f"cannot read segment {path}: {exc!r}") from exc
        try:
            magic = fh.read(len(SEGMENT_MAGIC))
            if magic != SEGMENT_MAGIC:
                raise SegmentIntegrityError("bad leading magic")
            hlen = int.from_bytes(self._read_exact(fh, 8, "header length"), "big")
            header = json.loads(self._read_exact(fh, hlen, "header"))
            self._body_start = fh.tell()
            file_size = path.stat().st_size
            tail_size = 8 + len(SEGMENT_MAGIC)
            if file_size < self._body_start + tail_size:
                raise SegmentIntegrityError("segment too short for a trailer")
            trailer_len_pos = file_size - tail_size
            fh.seek(trailer_len_pos)
            tlen = int.from_bytes(self._read_exact(fh, 8, "trailer length"), "big")
            trailing = fh.read(len(SEGMENT_MAGIC))
            if trailing != SEGMENT_MAGIC:
                raise SegmentIntegrityError("bad trailing magic")
            if tlen > trailer_len_pos - self._body_start:
                raise SegmentIntegrityError("trailer length overruns segment")
            body_end = trailer_len_pos - tlen
            digest = hashlib.sha256()
            fh.seek(self._body_start)
            remaining = body_end - self._body_start
            while remaining > 0:
                chunk = fh.read(min(1 << 20, remaining))
                if not chunk:
                    raise SegmentIntegrityError("unexpected EOF while hashing body")
                digest.update(chunk)
                remaining -= len(chunk)
            fh.seek(body_end)
            trailer = json.loads(self._read_exact(fh, tlen, "trailer"))
        except SegmentIntegrityError:
            fh.close()
            raise
        except (OSError, json.JSONDecodeError) as exc:
            fh.close()
            raise SegmentIntegrityError(
                f"segment {path.name} unreadable: {exc!r}"
            ) from exc
        if header.get("format") != SEGMENT_FORMAT:
            raise SegmentIntegrityError("unknown segment format")
        if header.get("format_version") != SEGMENT_FORMAT_VERSION:
            raise SegmentIntegrityError(
                f"unsupported segment format version {header.get('format_version')!r}"
            )
        if header.get("role") != self.expected_role:
            raise SegmentIntegrityError(
                f"segment role {header.get('role')!r} != expected {self.expected_role!r}"
            )
        if header.get("run_id") != self.expected_run_id:
            raise SegmentIntegrityError(
                "segment belongs to a different run; temporary files are never "
                "trusted across runs"
            )
        if header.get("schema_digest") != schema_digest(self.expected_schema):
            raise SegmentIntegrityError("segment schema digest differs from run schema")
        if trailer.get("record_count") != self.meta.record_count:
            raise SegmentIntegrityError("trailer record count disagrees with metadata")
        if trailer.get("body_sha256") != digest.hexdigest():
            raise SegmentIntegrityError("segment body checksum mismatch")
        self._trailer = trailer
        fh.seek(self._body_start)
        try:
            peek = pa.ipc.open_stream(fh)
            if peek.schema != self.expected_schema:
                raise SegmentIntegrityError(
                    "segment Arrow schema differs from run schema"
                )
        except SegmentIntegrityError:
            fh.close()
            raise
        except Exception as exc:  # noqa: BLE001 - any Arrow parse failure is fatal
            fh.close()
            raise SegmentIntegrityError(f"segment IPC header invalid: {exc!r}") from exc
        fh.seek(self._body_start)
        self._fh = fh
        self._stream = pa.ipc.open_stream(fh)
        self._emitted = 0

    def close(self) -> None:
        try:
            self._fh.close()
        except (OSError, AttributeError):
            pass

    def iter_batches(self):
        """Lazily decode one batch at a time (bounded resident footprint)."""
        for batch in self._stream:
            self._emitted += batch.num_rows
            yield batch
        if self._emitted != self.meta.record_count or \
                self._emitted != self._trailer["record_count"]:
            raise SegmentIntegrityError(
                f"segment row accounting inconsistent: ipc={self._emitted} "
                f"meta={self.meta.record_count} "
                f"trailer={self._trailer['record_count']}"
            )


def sort_record_table(table: pa.Table) -> pa.Table:
    """Stable sort of a record table by the binary sort-key column."""
    if table.num_rows == 0:
        return table
    indices = pc.sort_indices(table, sort_keys=[(PKEY_FIELD, "ascending")])
    return table.take(indices)


@dataclasses.dataclass(slots=True)
class RunStats:
    records: int = 0
    spill_segments: int = 0
    bytes_in_memory: int = 0
    peak_bytes_in_memory: int = 0
    spilled_bytes: int = 0


class RunBuilder:
    """Accumulates records under a byte ceiling and spills sorted segments."""

    def __init__(
        self,
        role: str,
        schema: pa.Schema,
        memory_limit: int,
        spill_dir: Path,
        *,
        run_id: str,
        shard_identities: dict,
    ) -> None:
        self.role = role
        self.schema = schema
        self.memory_limit = memory_limit
        self.spill_dir = Path(spill_dir)
        self.run_id = run_id
        self.shard_identities = shard_identities
        self._buffers: list[pa.Table] = []
        self._buffered_bytes = 0
        self._segments: list[SegmentMeta] = []
        self.stats = RunStats()

    def add_table(self, table: pa.Table) -> None:
        if table.schema != self.schema:
            raise SegmentIntegrityError(
                f"record table schema mismatch on role {self.role}"
            )
        if table.num_rows == 0:
            return
        for chunk in self._chunk_to_limit(table):
            self._add_one(chunk)

    def _chunk_to_limit(self, table: pa.Table):
        size = _table_byte_size(table)
        if size <= self.memory_limit or table.num_rows <= 1:
            yield table
            return
        per_row = max(1, size // table.num_rows)
        rows_per_chunk = max(1, self.memory_limit // per_row)
        for start in range(0, table.num_rows, rows_per_chunk):
            yield table.slice(start, rows_per_chunk)

    def _add_one(self, table: pa.Table) -> None:
        size = _table_byte_size(table)
        if self._buffered_bytes > 0 and self._buffered_bytes + size > self.memory_limit:
            self.flush()
        self._buffers.append(table)
        self._buffered_bytes += size
        self.stats.records += table.num_rows
        self.stats.bytes_in_memory = self._buffered_bytes
        self.stats.peak_bytes_in_memory = max(
            self.stats.peak_bytes_in_memory, self._buffered_bytes
        )
        if self._buffered_bytes >= self.memory_limit:
            self.flush()

    def flush(self) -> SegmentMeta | None:
        if not self._buffers:
            return None
        combined = (
            self._buffers[0]
            if len(self._buffers) == 1
            else pa.concat_tables(self._buffers)
        )
        ordered = sort_record_table(combined)
        writer = open_segment_writer(
            self.spill_dir,
            self.role,
            self.schema,
            run_id=self.run_id,
            shard_identities=self.shard_identities,
        )
        width = max(1, _row_width(ordered))
        chunk_rows = max(1024, self.memory_limit // width)
        for start in range(0, ordered.num_rows, chunk_rows):
            part = ordered.slice(start, chunk_rows)
            writer.write_table(
                pa.Table.from_batches(part.to_batches(), schema=self.schema)
            )
        meta = writer.close()
        self._segments.append(meta)
        self.stats.spill_segments += 1
        self.stats.spilled_bytes += meta.path.stat().st_size
        self._buffers = []
        self._buffered_bytes = 0
        self.stats.bytes_in_memory = 0
        return meta

    def finish(self) -> "Run":
        if self._segments:
            self.flush()
            return Run(
                role=self.role,
                schema=self.schema,
                segments=list(self._segments),
                memory_table=None,
                stats=self.stats,
                run_id=self.run_id,
                shard_identities=self.shard_identities,
                spill_dir=self.spill_dir,
            )
        table = sort_record_table(
            self._buffers[0]
            if len(self._buffers) == 1
            else pa.concat_tables(self._buffers)
        ) if self._buffers else _empty_table(self.schema)
        return Run(
            role=self.role,
            schema=self.schema,
            segments=[],
            memory_table=table,
            stats=self.stats,
            run_id=self.run_id,
            shard_identities=self.shard_identities,
            spill_dir=self.spill_dir,
        )


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema})


def _row_width(table: pa.Table) -> int:
    if table.num_rows == 0:
        return 1
    return max(1, _table_byte_size(table) // table.num_rows)


def _table_byte_size(table: pa.Table) -> int:
    total = table.nbytes
    try:
        total += table.column(PKEY_FIELD).buffers()[-1].size
    except (IndexError, AttributeError):  # pragma: no cover - defensive
        pass
    return max(total, table.num_rows)


@dataclasses.dataclass(slots=True)
class Run:
    """One sorted record stream: either one in-memory table or N segments."""

    role: str
    schema: pa.Schema
    segments: list[SegmentMeta]
    memory_table: pa.Table | None
    stats: RunStats
    run_id: str
    shard_identities: dict
    spill_dir: Path

    @property
    def record_count(self) -> int:
        return self.stats.records

    def iter_batches(self):
        """Yield record batches in global sort-key order, memory-bounded."""
        if self.memory_table is not None:
            for batch in self.memory_table.to_batches():
                yield batch
            return
        current = list(self.segments)
        if len(current) > MERGE_FAN_IN:
            current = self._merge_passes(current)
        reader_streams = [
            _SegmentBatchStream(meta, self.schema, self.role, self.run_id)
            for meta in current
        ]
        try:
            yield from _heap_merge(reader_streams)
        finally:
            for stream in reader_streams:
                stream.close()

    def _merge_passes(self, metas: list[SegmentMeta]) -> list[SegmentMeta]:
        level = list(metas)
        pass_no = 0
        while len(level) > MERGE_FAN_IN:
            pass_no += 1
            next_level: list[SegmentMeta] = []
            for start in range(0, len(level), MERGE_FAN_IN):
                group = level[start:start + MERGE_FAN_IN]
                streams = [
                    _SegmentBatchStream(meta, self.schema, self.role, self.run_id)
                    for meta in group
                ]
                # Merged segments stay in the same run: the run_id is the
                # identity of the whole run, not of an individual segment.
                writer = open_segment_writer(
                    self.spill_dir,
                    self.role,
                    self.schema,
                    run_id=self.run_id,
                    shard_identities=self.shard_identities,
                )
                buffered: list[pa.RecordBatch] = []
                try:
                    for one_row in _heap_merge(streams):
                        buffered.append(one_row)
                        if len(buffered) >= MERGE_WRITE_BATCH:
                            writer.write_table(
                                pa.Table.from_batches(buffered, schema=self.schema)
                            )
                            buffered.clear()
                    if buffered:
                        writer.write_table(
                            pa.Table.from_batches(buffered, schema=self.schema)
                        )
                finally:
                    for stream in streams:
                        stream.close()
                meta = writer.close()
                next_level.append(meta)
                self.stats.spilled_bytes += meta.path.stat().st_size
            level = next_level
        return level


class _SegmentBatchStream:
    """Lazy record-by-record view over one verified segment reader."""

    def __init__(self, meta: SegmentMeta, schema: pa.Schema, role: str,
                 run_id: str) -> None:
        self._meta = meta
        self._schema = schema
        self._role = role
        self._run_id = run_id
        self._reader: SegmentReader | None = None
        self._batches = None
        self._batch: pa.RecordBatch | None = None
        self._idx = 0
        self._open = True

    def _ensure(self):
        if self._reader is None:
            self._reader = SegmentReader(
                self._meta,
                self._schema,
                expected_role=self._role,
                expected_run_id=self._run_id,
            )
            self._batches = iter(self._reader.iter_batches())

    def head(self):
        if not self._open:
            return None
        self._ensure()
        if self._batch is None or self._idx >= self._batch.num_rows:
            self._batch = next(self._batches, None)
            self._idx = 0
            if self._batch is None:
                self.close()
                return None
        return self._batch, self._idx

    def pop(self):
        head = self.head()
        if head is None:
            return None
        batch, idx = head
        record = batch.slice(idx, 1)
        self._idx += 1
        return record

    def close(self) -> None:
        if self._open:
            self._open = False
            if self._reader is not None:
                self._reader.close()


def _heap_merge(streams: list[_SegmentBatchStream]):
    """Merge sorted streams into one ordered stream of one-row record batches."""
    import heapq

    heap: list[tuple[bytes, int, _SegmentBatchStream]] = []
    for uid, stream in enumerate(streams):
        head = stream.head()
        if head is not None:
            batch, idx = head
            heap.append((batch.column(PKEY_FIELD)[idx].as_py(), uid, stream))
    heapq.heapify(heap)
    while heap:
        _, uid, stream = heapq.heappop(heap)
        yield stream.pop()
        head = stream.head()
        if head is not None:
            batch, idx = head
            heapq.heappush(
                heap,
                (batch.column(PKEY_FIELD)[idx].as_py(), uid, stream),
            )


def cleanup_stale_run_files(spill_root: Path, *, current_run_id: str) -> dict[str, int]:
    """Remove/quarantine temporary files that cannot belong to this run.

    Temporary segments and per-key spill buffers are scoped to one run id;
    anything else left in the spill root (interrupted older runs, foreign
    files) is deleted when well-formed and quarantined when unparseable, so
    no later job ever mistakes run temporaries for trusted input.
    """
    spill_root = Path(spill_root)
    result = {"removed": 0, "quarantined": 0}
    if not spill_root.exists():
        return result
    quarantine = spill_root / "quarantine"
    for path in sorted(spill_root.glob("*.pqrtseg")):
        try:
            with path.open("rb") as fh:
                magic = fh.read(len(SEGMENT_MAGIC))
                if magic != SEGMENT_MAGIC:
                    raise SegmentIntegrityError("bad magic")
                hlen_bytes = fh.read(8)
                if len(hlen_bytes) != 8:
                    raise SegmentIntegrityError("short header length")
                hlen = int.from_bytes(hlen_bytes, "big")
                header = json.loads(fh.read(hlen))
            if current_run_id in str(header.get("run_id", "")):
                continue  # owned by this run (e.g. retry within the process)
            path.unlink()
            result["removed"] += 1
        except (OSError, json.JSONDecodeError, SegmentIntegrityError, KeyError):
            quarantine.mkdir(parents=True, exist_ok=True)
            try:
                os.replace(path, quarantine / path.name)
            except OSError:
                try:
                    path.unlink()
                except OSError:
                    pass
            result["quarantined"] += 1
    # Per-key spill buffers and partial fragments from interrupted runs.
    for pattern in ("*.pqrkeybuf", "*.tmp"):
        for path in sorted(spill_root.glob(pattern)):
            try:
                path.unlink()
                result["removed"] += 1
            except OSError:
                pass
    return result
