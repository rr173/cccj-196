"""Bounded-memory external sort: self-describing, checksummed spill segments.

The diff engine never assumes the two Parquet files are ordered alike and
never loads a whole snapshot into memory.  For each side it:

1. scans declared columns row group by row group (sparse Range GETs),
2. validates/encodes keys one micro-batch at a time into *records*,
3. accumulates records only until the configured memory ceiling, then sorts
   the run by canonical key bytes and spills it as a segment file,
4. merges the per-side runs with a k-way heap merge.

A segment is a single framed file (so a crash can only leave a whole,
unverified file -- never a partially "complete" one)::

    PQRSEG01\\n
    <u64 BE header-json-length><header JSON utf-8>
    <Arrow IPC stream: schema + sorted record batches>
    <u64 BE trailer-json-length><trailer JSON: counts + sha256 of the IPC body>
    PQRSEG01\\n

Every segment records the format version, the Arrow schema (digest + full
schema string), BOTH snapshot identities, the record count and a SHA-256
checksum of the IPC body.  On read the magic framing, counts, schema digest
and checksum are re-verified; any mismatch quarantines the segment and fails
the run instead of merging untrusted facts.
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
from .identity import ObjectIdentity

SEGMENT_MAGIC = b"PQRSEG01\n"
SEGMENT_FORMAT = "pqr-diff-segment"
SEGMENT_FORMAT_VERSION = 1
# Fixed leading record columns; comparison columns follow in declared order.
PKEY_FIELD = "_pkey"
KMETA_FIELD = "_kmeta"   # JSON rendering {type, value} of the business key
RG_FIELD = "_rg"
ROW_FIELD = "_row"
META_FIELDS = (PKEY_FIELD, KMETA_FIELD, RG_FIELD, ROW_FIELD)


def record_schema(compare_fields: list[pa.Field]) -> pa.Schema:
    """Schema of every record table / segment: meta columns + compare values."""
    fields = [
        pa.field(PKEY_FIELD, pa.binary(), nullable=False),
        pa.field(KMETA_FIELD, pa.string(), nullable=False),
        pa.field(RG_FIELD, pa.int32(), nullable=False),
        pa.field(ROW_FIELD, pa.int64(), nullable=False),
    ]
    names = {f.name for f in fields}
    for f in compare_fields:
        if f.name in names:
            raise SegmentIntegrityError(
                f"compare column {f.name!r} clashes with reserved segment metadata"
            )
        fields.append(f)
    return pa.schema(fields)


def schema_digest(schema: pa.Schema) -> str:
    """Stable digest of an Arrow schema (types + field order + nullability)."""
    sink = io.BytesIO()
    with pa.ipc.new_stream(sink, schema) as _:
        pass
    return hashlib.sha256(sink.getvalue()).hexdigest()


def identity_json(identity: ObjectIdentity | None) -> dict | None:
    if identity is None:
        return None
    return {
        "url": identity.url,
        "etag": identity.etag,
        "length": identity.length,
        "version_id": identity.version_id(),
    }


@dataclasses.dataclass(slots=True)
class SegmentMeta:
    path: Path
    side: str
    record_count: int
    schema_digest: str
    schema_text: str
    run_id: str


class SegmentWriter:
    """Writes one framed, checksummed, sorted segment.

    Use :func:`open_segment_writer` so the IPC body is checksummed through the
    hashing wrapper; instances are not intended to be constructed directly.
    """

    def __init__(self) -> None:  # pragma: no cover - use the factory
        raise RuntimeError("use open_segment_writer()")

    def _init(
        self,
        directory: Path,
        side: str,
        schema: pa.Schema,
        *,
        left: ObjectIdentity | None,
        right: ObjectIdentity | None,
        run_id: str,
    ) -> None:
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.side = side
        self.schema = schema
        self.run_id = run_id
        self.path = self.directory / f"{run_id}.{side}.{uuid.uuid4().hex}.pqrseg"
        self._fh = self.path.open("wb")
        self._fh.write(SEGMENT_MAGIC)
        self._digest = hashlib.sha256()
        self.record_count = 0
        self.batch_count = 0
        header = {
            "format": SEGMENT_FORMAT,
            "format_version": SEGMENT_FORMAT_VERSION,
            "side": side,
            "run_id": run_id,
            "schema_digest": schema_digest(schema),
            "schema": str(schema),
            "left_identity": identity_json(left),
            "right_identity": identity_json(right),
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
        # Layout: <IPC body><trailer JSON><u64 BE trailer length><magic>.
        # The fixed 8-byte length sits directly before the trailing magic, so a
        # reader can locate the trailer from the end without an index.
        self._fh.write(trailer_bytes)
        self._fh.write(len(trailer_bytes).to_bytes(8, "big"))
        self._fh.write(SEGMENT_MAGIC)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        return SegmentMeta(
            path=self.path,
            side=self.side,
            record_count=self.record_count,
            schema_digest=schema_digest(self.schema),
            schema_text=str(self.schema),
            run_id=self.run_id,
        )


class _HashingFile(io.RawIOBase):
    """Native-file-compatible wrapper that hashes every byte written through it.

    PyArrow's IPC stream writer requires a writable ``NativeFile``; wrapping a
    raw binary file object with :class:`pyarrow.PythonFile` satisfies that while
    the SHA-256 digest is computed over exactly the bytes Arrow emits.
    """

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
    side: str,
    schema: pa.Schema,
    *,
    left: ObjectIdentity | None,
    right: ObjectIdentity | None,
    run_id: str,
) -> SegmentWriter:
    """Factory wiring the hashing wrapper before Arrow opens the IPC stream."""
    sw = SegmentWriter.__new__(SegmentWriter)
    sw._init(
        directory,
        side,
        schema,
        left=left,
        right=right,
        run_id=run_id,
    )
    return sw


class SegmentReader:
    """Verifies framing/checksum/identity/schema, then yields sorted batches."""

    def __init__(self, meta: SegmentMeta, expected_schema: pa.Schema,
                 *, expected_side: str,
                 left: ObjectIdentity | None, right: ObjectIdentity | None) -> None:
        self.meta = meta
        self.expected_schema = expected_schema
        self.expected_side = expected_side
        self.left = left
        self.right = right
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
        """Stream-verify the whole segment without retaining its payload.

        The checksum must cover the entire IPC body, so we hash in bounded
        chunks while locating the trailer; only metadata (never the decoded
        batches) is retained afterwards.  Batch reading happens lazily through
        :meth:`iter_batches` against the on-disk file.
        """
        path = self.meta.path
        try:
            fh = path.open("rb")
        except OSError as exc:
            raise SegmentIntegrityError(f"cannot read segment {path}: {exc!r}") from exc
        try:
            magic = fh.read(len(SEGMENT_MAGIC))
            if magic != SEGMENT_MAGIC:
                raise SegmentIntegrityError("bad leading magic")
            (hlen,) = _be_uint64(self._read_exact(fh, 8, "header length"))
            header = json.loads(self._read_exact(fh, hlen, "header"))
            self._body_start = fh.tell()
            file_size = path.stat().st_size
            # Fixed-size tail: u64 trailer length + trailing magic.
            tail_size = 8 + len(SEGMENT_MAGIC)
            if file_size < self._body_start + tail_size:
                raise SegmentIntegrityError("segment too short for a trailer")
            trailer_len_pos = file_size - tail_size
            fh.seek(trailer_len_pos)
            (tlen,) = _be_uint64(self._read_exact(fh, 8, "trailer length"))
            trailing = fh.read(len(SEGMENT_MAGIC))
            if trailing != SEGMENT_MAGIC:
                raise SegmentIntegrityError("bad trailing magic")
            if tlen > trailer_len_pos - self._body_start:
                raise SegmentIntegrityError("trailer length overruns segment")
            # Hash the IPC body in bounded chunks and capture its exact span.
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
            trailer_bytes = self._read_exact(fh, tlen, "trailer")
            trailer = json.loads(trailer_bytes)
        except SegmentIntegrityError:
            fh.close()
            raise
        except (OSError, json.JSONDecodeError) as exc:
            fh.close()
            raise SegmentIntegrityError(f"segment {path.name} unreadable: {exc!r}") from exc
        # Header contract.
        if header.get("format") != SEGMENT_FORMAT:
            raise SegmentIntegrityError("unknown segment format")
        if header.get("format_version") != SEGMENT_FORMAT_VERSION:
            raise SegmentIntegrityError(
                f"unsupported segment format version {header.get('format_version')!r}"
            )
        if header.get("side") != self.expected_side:
            raise SegmentIntegrityError(
                f"segment side {header.get('side')!r} != expected {self.expected_side!r}"
            )
        if header.get("schema_digest") != schema_digest(self.expected_schema):
            raise SegmentIntegrityError("segment schema digest differs from run schema")
        self._verify_identity(header.get("left_identity"), self.left, "left")
        self._verify_identity(header.get("right_identity"), self.right, "right")
        if trailer.get("record_count") != self.meta.record_count:
            raise SegmentIntegrityError("trailer record count disagrees with metadata")
        if trailer.get("body_sha256") != digest.hexdigest():
            raise SegmentIntegrityError("segment body checksum mismatch")
        self._trailer = trailer
        self._body_end = body_end
        # Validate the Arrow schema by reading only the stream's header.
        fh.seek(self._body_start)
        try:
            peek = pa.ipc.open_stream(fh)
            if peek.schema != self.expected_schema:
                raise SegmentIntegrityError("segment Arrow schema differs from run schema")
        except SegmentIntegrityError:
            fh.close()
            raise
        except Exception as exc:  # noqa: BLE001 - any Arrow parse failure is fatal
            fh.close()
            raise SegmentIntegrityError(f"segment IPC header invalid: {exc!r}") from exc
        # Keep the handle open for lazy, bounded batch iteration; the IPC stream
        # only decodes one batch at a time on demand.
        fh.seek(self._body_start)
        self._fh = fh
        self._stream = pa.ipc.open_stream(fh)
        self._emitted = 0

    def close(self) -> None:
        try:
            self._fh.close()
        except (OSError, AttributeError):
            pass

    @staticmethod
    def _verify_history(stored: dict | None, expected: ObjectIdentity | None,
                        side: str) -> None:
        if expected is None:
            if stored is not None:
                raise SegmentIntegrityError(f"segment carries unexpected {side} identity")
            return
        if stored is None:
            raise SegmentIntegrityError(f"segment missing required {side} identity")
        if (
            stored.get("url") != expected.url
            or stored.get("etag") != expected.etag
            or stored.get("length") != expected.length
            or stored.get("version_id") != expected.version_id()
        ):
            raise SegmentIntegrityError(
                f"segment {side} identity {stored!r} disagrees with pinned "
                f"{expected.describe()!r}"
            )

    # keep the old name as an alias for clarity at call sites
    _verify_identity = _verify_history

    def iter_batches(self):
        """Lazily decode one batch at a time (bounded resident footprint)."""
        for batch in self._stream:
            self._emitted += batch.num_rows
            yield batch
        if self._emitted != self.meta.record_count or \
                self._emitted != self._trailer["record_count"]:
            raise SegmentIntegrityError(
                f"segment row accounting inconsistent: ipc={self._emitted} "
                f"meta={self.meta.record_count} trailer={self._trailer['record_count']}"
            )


def _be_uint64(raw: bytes) -> tuple[int]:
    return (int.from_bytes(raw, "big"),)


def sort_record_table(table: pa.Table) -> pa.Table:
    """Stable sort of a record table by canonical key bytes (binary order)."""
    if table.num_rows == 0:
        return table
    indices = pc.sort_indices(
        table, sort_keys=[(PKEY_FIELD, "ascending")]
    )
    return table.take(indices)


# ------------------------------------------------------------- run management
@dataclasses.dataclass(slots=True)
class RunStats:
    records: int = 0
    spill_segments: int = 0
    bytes_in_memory: int = 0
    peak_bytes_in_memory: int = 0
    spilled_bytes: int = 0


class RunBuilder:
    """Accumulates records under a byte ceiling and spills sorted segments.

    A single side of a diff owns one builder.  Records are held as Arrow record
    tables; when adding a micro-batch would cross ``memory_limit`` the buffered
    records are sorted and flushed to a framed segment, keeping the resident
    footprint bounded.
    """

    def __init__(
        self,
        side: str,
        schema: pa.Schema,
        memory_limit: int,
        spill_dir: Path,
        *,
        left: ObjectIdentity | None,
        right: ObjectIdentity | None,
        run_id: str,
    ) -> None:
        self.side = side
        self.schema = schema
        self.memory_limit = memory_limit
        self.spill_dir = Path(spill_dir)
        self.left = left
        self.right = right
        self.run_id = run_id
        self._buffers: list[pa.Table] = []
        self._buffered_bytes = 0
        self._segments: list[SegmentMeta] = []
        self.stats = RunStats()

    def add_table(self, table: pa.Table) -> None:
        if table.schema != self.schema:
            raise SegmentIntegrityError(
                f"record table schema mismatch on side {self.side}"
            )
        if table.num_rows == 0:
            return
        # Split an over-sized incoming micro-batch into row slices that each
        # fit the ceiling, so the resident footprint honors ``memory_limit``
        # even when one decoded Parquet micro-batch is itself larger than it.
        for chunk in self._chunk_to_limit(table):
            self._add_one(chunk)

    def _chunk_to_limit(self, table: pa.Table):
        size = _table_byte_size(table)
        if size <= self.memory_limit or table.num_rows <= 1:
            yield table
            return
        # Binary-search-ish linear split on row count using per-row estimates.
        per_row = max(1, size // table.num_rows)
        rows_per_chunk = max(1, self.memory_limit // per_row)
        for start in range(0, table.num_rows, rows_per_chunk):
            part = table.slice(start, rows_per_chunk)
            # A single pathological row may still exceed the limit; in that
            # case it cannot be split further and is accepted as one record.
            yield part

    def _add_one(self, table: pa.Table) -> None:
        size = _table_byte_size(table)
        # Flush first whenever the new chunk would cross the byte ceiling.
        if self._buffered_bytes > 0 and self._buffered_bytes + size > self.memory_limit:
            self.flush()
        self._buffers.append(table)
        self._buffered_bytes += size
        self.stats.records += table.num_rows
        self.stats.bytes_in_memory = self._buffered_bytes
        self.stats.peak_bytes_in_memory = max(
            self.stats.peak_bytes_in_memory, self._buffered_bytes
        )
        # Honor the ceiling regardless of Arrow's minimum buffer accounting, so
        # a small declared limit really produces a spill rather than holding
        # everything resident.
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
            self.side,
            self.schema,
            left=self.left,
            right=self.right,
            run_id=self.run_id,
        )
        # Write in bounded chunks so the IPC serialization footprint stays
        # proportional to one chunk, not the whole run.
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
        # Drop the resident copy; the segment is the durable representation.
        self._buffers = []
        self._buffered_bytes = 0
        self.stats.bytes_in_memory = 0
        return meta

    def finish(self) -> "Run":
        if self._segments:
            # Any residual records become a final segment, so all runs that
            # spilled are uniformly segment-backed.
            self.flush()
            return Run(
                side=self.side,
                schema=self.schema,
                segments=list(self._segments),
                memory_table=None,
                stats=self.stats,
                left=self.left,
                right=self.right,
            )
        table = sort_record_table(
            self._buffers[0]
            if len(self._buffers) == 1
            else pa.concat_tables(self._buffers)
        ) if self._buffers else _empty_table(self.schema)
        return Run(
            side=self.side,
            schema=self.schema,
            segments=[],
            memory_table=table,
            stats=self.stats,
            left=self.left,
            right=self.right,
        )


def _empty_table(schema: pa.Schema) -> pa.Table:
    return pa.table({f.name: pa.array([], type=f.type) for f in schema})


def _row_width(table: pa.Table) -> int:
    if table.num_rows == 0:
        return 1
    return max(1, _table_byte_size(table) // table.num_rows)


def _table_byte_size(table: pa.Table) -> int:
    """Resident-size estimate covering Arrow buffers plus encoded key bytes."""
    total = table.nbytes
    # nbytes excludes some nested/dictionary overhead; count key bytes too.
    try:
        total += table.column(PKEY_FIELD).buffers()[-1].size
    except (IndexError, AttributeError):  # pragma: no cover - defensive
        pass
    return max(total, table.num_rows)


@dataclasses.dataclass(slots=True)
class Run:
    """One side's sorted records: either one in-memory table or N segments."""

    side: str
    schema: pa.Schema
    segments: list[SegmentMeta]
    memory_table: pa.Table | None
    stats: RunStats
    left: ObjectIdentity | None
    right: ObjectIdentity | None

    @property
    def record_count(self) -> int:
        return self.stats.records

    def iter_batches(self):
        """Yield record batches in global canonical-key order, memory-bounded.

        An in-memory run is already sorted.  A spilled run is merged with a
        fixed-fan-in external merge so that at most :data:`MERGE_FAN_IN`
        segment readers are ever open (each holding only its current batch):

        * when the segment count already fits the fan-in, one lazy heap merge
          reads them straight through;
        * otherwise level passes merge every group of ``MERGE_FAN_IN`` segments
          into a new, still-sorted segment in controlled spill, repeating until
          a single segment remains, which is then streamed.

        Resident memory and open handles stay a constant factor over one
        segment regardless of the total segment count.
        """
        if self.memory_table is not None:
            for batch in self.memory_table.to_batches():
                yield batch
            return
        current = list(self.segments)
        if len(current) > MERGE_FAN_IN:
            current = self._merge_passes(current)
        reader_streams = [
            _SegmentBatchStream(meta, self.schema, self.side, self.left,
                                self.right)
            for meta in current
        ]
        try:
            yield from _heap_merge(reader_streams)
        finally:
            for stream in reader_streams:
                stream.close()

    def _merge_passes(self, metas: list[SegmentMeta]) -> list[SegmentMeta]:
        """Repeatedly merge <= MERGE_FAN_IN segments into one until bounded."""
        level = list(metas)
        pass_no = 0
        while len(level) > MERGE_FAN_IN:
            pass_no += 1
            next_level: list[SegmentMeta] = []
            for start in range(0, len(level), MERGE_FAN_IN):
                group = level[start:start + MERGE_FAN_IN]
                streams = [
                    _SegmentBatchStream(meta, self.schema, self.side,
                                        self.left, self.right)
                    for meta in group
                ]
                run_id = f"merge-p{pass_no}-{uuid.uuid4().hex}"
                writer = open_segment_writer(
                    self._merge_dir(),
                    self.side,
                    self.schema,
                    left=self.left,
                    right=self.right,
                    run_id=run_id,
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

    def _merge_dir(self) -> Path:
        # Intermediate merged segments live beside the run's spill segments.
        base = self.segments[0].path.parent if self.segments else Path(".")
        base.mkdir(parents=True, exist_ok=True)
        return base


# Rows per IPC message produced by an external merge pass (bounded, so a merge
# pass never materialises the whole merged run into memory at once).
MERGE_WRITE_BATCH = 512

# Maximum number of segment streams merged in one pass; bounds open file
# handles and the resident footprint of the k-way heap to a small constant.
MERGE_FAN_IN = 8


class _SegmentBatchStream:
    """Lazy record-by-record view over one verified segment reader.

    The underlying :class:`SegmentReader` (and its file handle) is opened only
    on the first read, so merely constructing streams never holds a handle.
    """

    def __init__(self, meta: SegmentMeta, schema: pa.Schema, side: str,
                 left, right) -> None:
        self._meta = meta
        self._schema = schema
        self._side = side
        self._left = left
        self._right = right
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
                expected_side=self._side,
                left=self._left,
                right=self._right,
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


def _heap_merge(streams: list["_SegmentBatchStream"]):
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


def quarantine_segment(meta: SegmentMeta, quarantine_dir: Path) -> Path:
    """Move an untrusted segment into the quarantine directory; return new path."""
    quarantine_dir = Path(quarantine_dir)
    quarantine_dir.mkdir(parents=True, exist_ok=True)
    target = quarantine_dir / meta.path.name
    try:
        os.replace(meta.path, target)
    except OSError:
        try:
            meta.path.unlink()
        except OSError:
            pass
    return target


def cleanup_stale_runs(spill_root: Path, *, current_run_id: str) -> dict[str, int]:
    """On startup, remove/quarantine segments that cannot belong to this run.

    Segments whose framing parses and whose run_id matches are kept (there
    should be none for a fresh id); well-formed leftovers of *other* runs are
    deleted; malformed files are quarantined.  This is the process-termination
    recovery path: no untrusted temporary segment is ever merged.
    """
    spill_root = Path(spill_root)
    result = {"removed": 0, "quarantined": 0}
    if not spill_root.exists():
        return result
    quarantine = spill_root.parent / "quarantine"
    for path in sorted(spill_root.glob("*.pqrseg")):
        try:
            with path.open("rb") as fh:
                magic = fh.read(len(SEGMENT_MAGIC))
                if magic != SEGMENT_MAGIC:
                    raise SegmentIntegrityError("bad magic")
                hlen_bytes = fh.read(8)
                if len(hlen_bytes) != 8:
                    raise SegmentIntegrityError("short header length")
                (hlen,) = _be_uint64(hlen_bytes)
                header = json.loads(fh.read(hlen))
            if header.get("run_id") == current_run_id:
                continue  # owned by this run (e.g. retry within the process)
            # Well-formed foreign segment: its identities cannot be trusted for
            # this run, so delete rather than merge.
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
    # Sweep partial/tmp fragments from interrupted writes.
    for path in sorted(spill_root.glob("*.tmp")):
        try:
            path.unlink()
            result["removed"] += 1
        except OSError:
            pass
    return result
