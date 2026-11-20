"""Temporal-enrich engine: bitemporal revision resolution + window matching.

Pipeline (no whole-object download, no full-table in-memory hold, no external
join engine)::

    manifest pin   -> conditional identity proof of EVERY shard version
                   -> footers only; schema contract across all shards
                      (exact Arrow types; rejected before any row data)
                   -> extract used columns only, validate rows, build
                      sortable records, external-sort revisions by
                      (entity_key, recorded_at, window_id) and facts by
                      (entity_key, known_at, seq) with checksummed spill
                      segments
                   -> merge-join by entity_key; per key a knowledge-time
                      sweep resolves each window's visible episode and a
                      compressed segment tree answers stabbing-max priority
                      per fact (never O(revisions) per fact, never a
                      candidate cartesian product, hot keys spool to disk)
                   -> results external-sorted back to fact-manifest order
                      and written as ONE Parquet wide table, staged in the
                      output directory and atomically swapped into place

Bitemporal semantics per fact ``(entity_key, event_at, known_at)``:

* only revisions of the same ``entity_key`` are considered;
* per ``window_id`` the visible revision is the last one with
  ``recorded_at <= known_at`` -- none or ``withdraw`` means the window does
  not exist for this fact (a moved window never falls back to its old
  range);
* a visible ``upsert`` covers the fact iff
  ``valid_from <= event_at < valid_to`` (no end = unbounded);
* among covering windows only the maximum ``priority`` wins: exactly one
  top window -> ``matched``; several -> ``ambiguous`` (never pick a winner,
  even if their business columns are identical); none -> ``unmatched``.

Every failure path (request/schema/data/identity/protocol/cancellation/
write) removes this run's temporary files and never replaces a previously
published wide table.
"""

from __future__ import annotations

import dataclasses
import json
import os
import resource
import shutil
import time
import uuid
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from .client import RemoteParquetClient
from .exceptions import (
    TemporalCancelled,
    TemporalDataError,
    TemporalPublishError,
    TemporalResourceError,
)
from .temporal_publish import (
    publish_parquet,
    recover_staging,
    staging_path_for,
)
from .temporal_request import (
    ENTITY_KEY,
    EVENT_AT,
    KNOWN_AT,
    OP,
    OP_UPSERT,
    OP_WITHDRAW,
    PRIORITY,
    RECORDED_AT,
    RESERVED_OUTPUT_COLUMNS,
    OUT_DIM,
    OUT_DIM_SOURCES,
    OUT_FACT_SOURCE,
    OUT_STATUS,
    OUT_TOP_COUNT,
    VALID_FROM,
    VALID_TO,
    WINDOW_ID,
    TemporalEnrichRequest,
)
from .temporal_schema import (
    TemporalContract,
    carriage_type,
    check_temporal_contract,
    from_carriage_values,
    to_carriage,
)
from .temporal_segment import (
    PKEY_FIELD,
    Run,
    RunBuilder,
    cleanup_stale_run_files,
)
from .temporal_tree import StabbingMaxTree

REPORT_FORMAT = "pqr-temporal-enrich-report"
OUTPUT_METADATA_KEY = "pqr.temporal_enrich"
STATUS_MATCHED = "matched"
STATUS_AMBIGUOUS = "ambiguous"
STATUS_UNMATCHED = "unmatched"

# Fraction of the job memory budget a single key's spool buffer may hold
# before that key is written to disk and processed file-backed.
_KEY_BUFFER_SHARE = 4
# Rows buffered between two result-builder flushes.
_RESULT_FLUSH_ROWS = 2048
# Rows per output write batch.
_OUTPUT_BATCH_ROWS = 8192


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _frame(raw: bytes) -> bytes:
    return len(raw).to_bytes(8, "big") + raw


def _ordered_signed64(value: int) -> bytes:
    raw = value + (1 << 64) if value < 0 else value
    raw ^= 1 << 63
    return raw.to_bytes(8, "big")


def _revision_pkey(entity_key: str, recorded_at: int, window_id: str) -> bytes:
    return (
        _frame(entity_key.encode("utf-8"))
        + _ordered_signed64(recorded_at)
        + _frame(window_id.encode("utf-8"))
    )


def _fact_pkey(entity_key: str, known_at: int, seq: int) -> bytes:
    return (
        _frame(entity_key.encode("utf-8"))
        + _ordered_signed64(known_at)
        + seq.to_bytes(8, "big")
    )


def _result_pkey(seq: int) -> bytes:
    return seq.to_bytes(8, "big")


def _ekey_of_pkey(pkey: bytes) -> bytes:
    """The framed entity_key prefix of a revision/fact sort key."""
    klen = int.from_bytes(pkey[:8], "big")
    return pkey[8:8 + klen]


class _Peekable:
    __slots__ = ("_it", "head")

    def __init__(self, iterable):
        self._it = iter(iterable)
        self.head = next(self._it, None)

    def advance(self) -> None:
        self.head = next(self._it, None)


def _iter_groups(rows, key_of):
    """Yield ``(key, lazy_row_iterator)`` groups from a sorted row stream."""
    pk = _Peekable(rows)
    while pk.head is not None:
        key = key_of(pk.head)

        def _group(key=key, pk=pk):
            while pk.head is not None and key_of(pk.head) == key:
                row = pk.head
                pk.advance()
                yield row

        yield key, _group()


@dataclasses.dataclass(slots=True)
class _Episode:
    """The currently visible state of one window during the sweep."""

    valid_from: int
    valid_to: int | None
    priority: int
    dims: tuple                       # carriage-form dimension values
    source: tuple                     # (shard_id, row_group, row)


class _KeyBuffer:
    """Per-key spool: memory until the byte share is crossed, then disk.

    A hot key's records are never held resident: once the buffered estimate
    crosses ``limit_bytes`` the buffer (and everything that follows) is
    written to a run-scoped Arrow IPC file and both processing passes read
    it back from disk.
    """

    def __init__(self, spill_dir: Path, run_id: str, ordinal: int,
                 limit_bytes: int, rev_schema: pa.Schema, fact_schema: pa.Schema,
                 rev_names: list[str], fact_names: list[str], meter) -> None:
        self._dir = spill_dir
        self._rev_schema = rev_schema
        self._fact_schema = fact_schema
        self._rev_names = rev_names
        self._fact_names = fact_names
        self._limit = limit_bytes
        self._meter = meter
        self._rev_rows: list[tuple] = []
        self._fact_rows: list[tuple] = []
        self._bytes = 0
        self.spilled = False
        self._rev_pending: list[tuple] = []
        self._fact_pending: list[tuple] = []
        self._rev_writer = None
        self._fact_writer = None
        self._rev_path: Path | None = None
        self._fact_path: Path | None = None
        self._paths: list[Path] = []
        self._sizes_reported = False
        self._base = f"{run_id}.keybuf-{ordinal}"

    @staticmethod
    def _row_bytes(row: tuple) -> int:
        size = 16 * len(row)
        for value in row:
            if isinstance(value, (bytes, str)):
                size += len(value)
            elif isinstance(value, (list, tuple)):
                size += 16 * len(value)
        return size

    def add_rev(self, row: tuple) -> None:
        if self.spilled:
            self._write_spilled(self._rev_writer, self._rev_schema,
                                self._rev_names, row, self._rev_pending)
        else:
            self._rev_rows.append(row)
            self._bytes += self._row_bytes(row)
            self._check_spill()

    def add_fact(self, row: tuple) -> None:
        if self.spilled:
            self._write_spilled(self._fact_writer, self._fact_schema,
                                self._fact_names, row, self._fact_pending)
        else:
            self._fact_rows.append(row)
            self._bytes += self._row_bytes(row)
            self._check_spill()

    # The engine keeps this class small: spilled rows are accumulated in
    # plain lists and flushed in chunks by _flush_pending.

    def _check_spill(self) -> None:
        if not self.spilled and self._bytes > self._limit:
            self._spill()

    def _spill(self) -> None:
        self.spilled = True
        self._rev_path = self._dir / f"{self._base}.rev.pqrkeybuf"
        self._fact_path = self._dir / f"{self._base}.fact.pqrkeybuf"
        self._rev_writer = pa.ipc.new_stream(
            pa.OSFile(str(self._rev_path), "w"), self._rev_schema
        )
        self._fact_writer = pa.ipc.new_stream(
            pa.OSFile(str(self._fact_path), "w"), self._fact_schema
        )
        for row in self._rev_rows:
            self._write_spilled(self._rev_writer, self._rev_schema,
                                self._rev_names, row, self._rev_pending)
        for row in self._fact_rows:
            self._write_spilled(self._fact_writer, self._fact_schema,
                                self._fact_names, row, self._fact_pending)
        self._rev_rows = []
        self._fact_rows = []
        self._paths = [self._rev_path, self._fact_path]
        for path in self._paths:
            self._meter(path, 0)

    def _write_spilled(self, writer, schema, names, row, pending) -> None:
        pending.append(row)
        if len(pending) >= 4096:
            self._flush_pending(writer, schema, names, pending)

    @staticmethod
    def _flush_pending(writer, schema, names, pending) -> None:
        if not pending:
            return
        cols = list(zip(*pending))
        arrays = [
            pa.array(list(values), type=schema.field(name).type)
            for name, values in zip(names, cols)
        ]
        writer.write_batch(pa.RecordBatch.from_arrays(arrays, schema=schema))
        pending.clear()

    def _close_writers(self) -> None:
        if self._rev_writer is not None:
            self._flush_pending(self._rev_writer, self._rev_schema,
                                self._rev_names, self._rev_pending)
            self._rev_writer.close()
            self._rev_writer = None
        if self._fact_writer is not None:
            self._flush_pending(self._fact_writer, self._fact_schema,
                                self._fact_names, self._fact_pending)
            self._fact_writer.close()
            self._fact_writer = None
        if self.spilled and not self._sizes_reported:
            self._sizes_reported = True
            for path in self._paths:
                self._meter(path, path.stat().st_size if path.exists() else 0)

    def iter_revs(self):
        if not self.spilled:
            yield from self._rev_rows
            return
        self._close_writers()
        with pa.OSFile(str(self._rev_path), "r") as fh:
            reader = pa.ipc.open_stream(fh)
            for batch in reader:
                cols = [batch.column(i).to_pylist()
                        for i in range(len(self._rev_names))]
                yield from zip(*cols)

    def iter_facts(self):
        if not self.spilled:
            yield from self._fact_rows
            return
        self._close_writers()
        with pa.OSFile(str(self._fact_path), "r") as fh:
            reader = pa.ipc.open_stream(fh)
            for batch in reader:
                cols = [batch.column(i).to_pylist()
                        for i in range(len(self._fact_names))]
                yield from zip(*cols)

    def cleanup(self) -> None:
        try:
            self._close_writers()
        finally:
            for path in self._paths:
                size = path.stat().st_size if path.exists() else 0
                try:
                    path.unlink()
                except OSError:
                    pass
                if size:
                    self._meter(path, -size)
            self._paths = []


class TemporalEnrichEngine:
    """Runs one temporal-enrich job against pinned shard versions."""

    def __init__(
        self,
        client: RemoteParquetClient,
        request: TemporalEnrichRequest,
        *,
        token=None,
    ) -> None:
        self.client = client
        self.request = request
        self.token = token
        self.run_id = f"temporal-{uuid.uuid4().hex}"
        out_dir = os.path.dirname(request.output_path)
        base = Path(request.spill_dir or os.path.join(
            out_dir, ".pqr-temporal-tmp"
        ))
        self.spill_parent = base
        self.spill_root = base / self.run_id
        self._cleanup_done = False
        self._temp_bytes_current = 0
        self._temp_bytes_peak = 0
        self._key_buffers_spilled = 0
        self._stats: dict[str, int] = {
            "revisions_read": 0,
            "facts_read": 0,
            "keys_total": 0,
            "keys_with_revisions_and_facts": 0,
            "keys_fact_only": 0,
            "keys_revision_only": 0,
            "episodes": 0,
            "tree_removes": 0,
            "stabbing_queries": 0,
            "tree_entries_peak": 0,
            "window_state_entries_peak": 0,
            "result_records": 0,
        }
        self._counts = {
            STATUS_MATCHED: 0,
            STATUS_AMBIGUOUS: 0,
            STATUS_UNMATCHED: 0,
        }

    # ------------------------------------------------------------ public run
    def run(self) -> dict:
        started = _utc_now()
        req = self.request
        out_dir = Path(os.path.dirname(req.output_path))
        try:
            out_dir.mkdir(parents=True, exist_ok=True)
            # Recover/clean interrupted previous attempts before doing
            # anything else; a leftover staging file or stale segment is
            # never trusted.
            recovery_out = recover_staging(out_dir)
            self.spill_root.mkdir(parents=True, exist_ok=True)
            recovery_tmp = cleanup_stale_run_files(
                self.spill_parent, current_run_id=self.run_id
            )
        except OSError as exc:
            raise TemporalPublishError(
                f"cannot create output/staging directory for "
                f"{req.output_path!r}: {exc!r}"
            ) from exc
        try:
            report = self._run_inner(started, recovery_out, recovery_tmp)
            return report
        finally:
            self._cleanup_temp()

    def _run_inner(self, started: str, recovery_out: dict,
                   recovery_tmp: dict) -> dict:
        req = self.request
        # ---------------------------------------------------- pin + inspect
        fact_handles = self._pin_all(req.fact_shards)
        revision_handles = self._pin_all(req.revision_shards)
        contract = check_temporal_contract(
            [(shard.shard_id, info.schema) for shard, _, info in fact_handles],
            [(shard.shard_id, info.schema) for shard, _, info in revision_handles],
            req.fact_columns,
            req.dim_columns,
        )
        self._prepare_layouts(contract)
        shard_identities = {}
        for role, handles in (("fact", fact_handles), ("revision", revision_handles)):
            for shard, obj, _ in handles:
                shard_identities[shard.shard_id] = {
                    "role": role,
                    "url": obj.identity.url,
                    "etag": obj.identity.etag,
                    "length": obj.identity.length,
                    "version_id": obj.identity.version_id(),
                }
        # ------------------------------------------------------- extraction
        rev_run = self._extract_revisions(revision_handles, shard_identities)
        fact_run = self._extract_facts(fact_handles, shard_identities)
        self._sample_temp_peak()
        # ---------------------------------------------------------- matching
        result_run = self._match(rev_run, fact_run, shard_identities)
        self._sample_temp_peak()
        # ------------------------------------------------------------ output
        output_schema = self._output_schema(contract, shard_identities)
        self._write_output(result_run, output_schema)
        finished = _utc_now()
        return self._build_report(
            contract, shard_identities, output_schema,
            started=started, finished=finished,
            recovery={"output_staging": recovery_out, "temp": recovery_tmp},
            runs=(rev_run, fact_run, result_run),
        )

    # -------------------------------------------------------------- pinning
    def _pin_all(self, shards):
        handles = []
        for shard in shards:
            obj = self.client.open_identity(shard.url, shard.etag, shard.length)
            info = obj.inspect(token=self.token)
            handles.append((shard, obj, info))
        return handles

    # ------------------------------------------------------------- layouts
    def _prepare_layouts(self, contract: TemporalContract) -> None:
        self.contract = contract
        dim_names = [f"d__{f.name}" for f in contract.dim_fields]
        fact_names = [f"f__{f.name}" for f in contract.fact_fields]
        self.rev_record_schema = pa.schema([
            pa.field(PKEY_FIELD, pa.binary(), nullable=False),
            pa.field(WINDOW_ID, contract.window_id_type, nullable=False),
            pa.field(RECORDED_AT, pa.int64(), nullable=False),
            pa.field(OP, contract.op_type, nullable=False),
            pa.field(VALID_FROM, pa.int64()),
            pa.field(VALID_TO, pa.int64()),
            pa.field(PRIORITY, pa.int64()),
            *[
                pa.field(f"d__{f.name}", carriage_type(f.type))
                for f in contract.dim_fields
            ],
            pa.field("_shard", pa.string(), nullable=False),
            pa.field("_rg", pa.int32(), nullable=False),
            pa.field("_row", pa.int64(), nullable=False),
        ])
        self.fact_record_schema = pa.schema([
            pa.field(PKEY_FIELD, pa.binary(), nullable=False),
            pa.field("_seq", pa.int64(), nullable=False),
            pa.field(EVENT_AT, pa.int64(), nullable=False),
            pa.field(KNOWN_AT, pa.int64(), nullable=False),
            *[
                pa.field(f"f__{f.name}", carriage_type(f.type))
                for f in contract.fact_fields
            ],
            pa.field("_shard", pa.string(), nullable=False),
            pa.field("_rg", pa.int32(), nullable=False),
            pa.field("_row", pa.int64(), nullable=False),
        ])
        self._source_struct = pa.struct([
            pa.field("shard_id", pa.string()),
            pa.field("row_group", pa.int32()),
            pa.field("row", pa.int64()),
            pa.field("window_id", pa.string()),
        ])
        self._dim_sources_type = pa.list_(pa.field("source", self._source_struct))
        self.result_record_schema = pa.schema([
            pa.field(PKEY_FIELD, pa.binary(), nullable=False),
            pa.field("_seq", pa.int64(), nullable=False),
            pa.field("_status", pa.string(), nullable=False),
            pa.field("_count", pa.int64(), nullable=False),
            *[
                pa.field(f"f__{f.name}", carriage_type(f.type))
                for f in contract.fact_fields
            ],
            *[
                pa.field(f"d__{f.name}", carriage_type(f.type))
                for f in contract.dim_fields
            ],
            pa.field("_fs_shard", pa.string(), nullable=False),
            pa.field("_fs_rg", pa.int32(), nullable=False),
            pa.field("_fs_row", pa.int64(), nullable=False),
            pa.field("_dim_sources", self._dim_sources_type),
        ])
        # Tuple-layout indices (schema order).
        n_dim = len(contract.dim_fields)
        n_fact = len(contract.fact_fields)
        self.R_WID = 1
        self.R_REC = 2
        self.R_OP = 3
        self.R_VF = 4
        self.R_VT = 5
        self.R_PRIO = 6
        self.R_DIM0 = 7
        self.R_SHARD = 7 + n_dim
        self.R_RG = 8 + n_dim
        self.R_ROW = 9 + n_dim
        self.F_SEQ = 1
        self.F_EVENT = 2
        self.F_KNOWN = 3
        self.F_FACT0 = 4
        self.F_SHARD = 4 + n_fact
        self.F_RG = 5 + n_fact
        self.F_ROW = 6 + n_fact
        self.O_STATUS = 2
        self.O_COUNT = 3
        self.O_FACT0 = 4
        self.O_DIM0 = 4 + n_fact
        self.O_FS_SHARD = 4 + n_fact + n_dim
        self.O_FS_RG = 5 + n_fact + n_dim
        self.O_FS_ROW = 6 + n_fact + n_dim
        self.O_SOURCES = 7 + n_fact + n_dim
        # Budget-derived caps for the per-key matching working set.  These
        # are entry-count caps with documented per-entry byte estimates; the
        # sort/reorder buffers themselves always spill when over budget.
        self._key_buffer_limit = max(64 << 10, req_limit := self.request.memory_limit,
                                     ) // _KEY_BUFFER_SHARE
        self._tree_entry_cap = max(1024, self.request.memory_limit // 64)
        self._window_state_cap = max(512, self.request.memory_limit // 256)
        self._coords_cap = max(2048, self.request.memory_limit // 16)

    # ----------------------------------------------------------- extraction
    def _check_cancelled(self) -> None:
        if self.token is not None and getattr(self.token, "cancelled", False):
            raise TemporalCancelled("temporal-enrich run cancelled")

    def _extract_revisions(self, handles, shard_identities) -> Run:
        builder = RunBuilder(
            "revision", self.rev_record_schema, self.request.memory_limit,
            self.spill_root, run_id=self.run_id,
            shard_identities=shard_identities,
        )
        needed = [
            ENTITY_KEY, WINDOW_ID, RECORDED_AT, OP, VALID_FROM, VALID_TO,
            PRIORITY, *self.contract.dim_column_names,
        ]
        for shard, obj, info in handles:
            self._check_cancelled()
            non_empty = [
                rg for rg in range(info.num_row_groups)
                if info.row_group_rows[rg] > 0
            ]
            if not non_empty:
                continue
            base_row = 0
            for rg, batch in self.client.reader.scan_batches(
                obj.identity, needed, non_empty,
                batch_size=self.request.batch_rows, token=self.token,
            ):
                self._check_cancelled()
                self._validate_revision_batch(batch, shard.shard_id, rg, base_row)
                builder.add_table(self._build_revision_records(
                    batch, shard.shard_id, rg, base_row
                ))
                base_row += batch.num_rows
                self._stats["revisions_read"] += batch.num_rows
                del batch
        return builder.finish()

    def _extract_facts(self, handles, shard_identities) -> Run:
        builder = RunBuilder(
            "fact", self.fact_record_schema, self.request.memory_limit,
            self.spill_root, run_id=self.run_id,
            shard_identities=shard_identities,
        )
        needed = [ENTITY_KEY, EVENT_AT, KNOWN_AT, *self.contract.fact_column_names]
        self._fact_seq = 0
        for shard, obj, info in handles:
            self._check_cancelled()
            non_empty = [
                rg for rg in range(info.num_row_groups)
                if info.row_group_rows[rg] > 0
            ]
            if not non_empty:
                continue
            base_row = 0
            for rg, batch in self.client.reader.scan_batches(
                obj.identity, needed, non_empty,
                batch_size=self.request.batch_rows, token=self.token,
            ):
                self._check_cancelled()
                self._validate_fact_batch(batch, shard.shard_id, rg, base_row)
                builder.add_table(self._build_fact_records(
                    batch, shard.shard_id, rg, base_row
                ))
                base_row += batch.num_rows
                self._stats["facts_read"] += batch.num_rows
                del batch
        return builder.finish()

    # ----------------------------------------------------------- validation
    @staticmethod
    def _first_index(mask) -> int:
        return pc.indices_nonzero(mask)[0].as_py()

    def _fail_row(self, shard_id, rg, row, column, problem) -> None:
        raise TemporalDataError(
            f"bad {column} row in shard {shard_id!r} row group {rg} "
            f"row {row}: {problem}",
            diagnostics=[{
                "shard_id": shard_id, "row_group": rg, "row": row,
                "column": column, "problem": problem,
            }],
        )

    def _require_non_null(self, arr, column, shard_id, rg, base_row) -> None:
        if arr.null_count:
            idx = self._first_index(pc.is_null(arr))
            self._fail_row(shard_id, rg, base_row + idx, column,
                           "must not be null")

    def _require_non_empty_utf8(self, arr, column, shard_id, rg, base_row) -> None:
        empty = pc.equal(pc.utf8_length(arr), 0)
        if pc.any(empty).as_py():
            idx = self._first_index(empty)
            self._fail_row(shard_id, rg, base_row + idx, column,
                           "must be a non-empty UTF-8 string")

    def _validate_fact_batch(self, batch, shard_id, rg, base_row) -> None:
        ekey = batch.column(ENTITY_KEY)
        self._require_non_null(ekey, ENTITY_KEY, shard_id, rg, base_row)
        self._require_non_empty_utf8(ekey, ENTITY_KEY, shard_id, rg, base_row)
        for col in (EVENT_AT, KNOWN_AT):
            self._require_non_null(batch.column(col), col, shard_id, rg, base_row)

    def _validate_revision_batch(self, batch, shard_id, rg, base_row) -> None:
        ekey = batch.column(ENTITY_KEY)
        self._require_non_null(ekey, ENTITY_KEY, shard_id, rg, base_row)
        self._require_non_empty_utf8(ekey, ENTITY_KEY, shard_id, rg, base_row)
        wid = batch.column(WINDOW_ID)
        self._require_non_null(wid, WINDOW_ID, shard_id, rg, base_row)
        self._require_non_empty_utf8(wid, WINDOW_ID, shard_id, rg, base_row)
        self._require_non_null(
            batch.column(RECORDED_AT), RECORDED_AT, shard_id, rg, base_row
        )
        op = batch.column(OP)
        self._require_non_null(op, OP, shard_id, rg, base_row)
        known_ops = pc.is_in(
            op, value_set=pa.array([OP_UPSERT, OP_WITHDRAW],
                                   type=self.contract.op_type)
        )
        if pc.any(pc.invert(known_ops)).as_py():
            idx = self._first_index(pc.invert(known_ops))
            self._fail_row(
                shard_id, rg, base_row + idx, OP,
                f"must be {OP_UPSERT!r} or {OP_WITHDRAW!r}",
            )
        vf = batch.column(VALID_FROM)
        vt = batch.column(VALID_TO)
        prio = batch.column(PRIORITY)
        is_upsert = pc.equal(op, OP_UPSERT)
        is_withdraw = pc.equal(op, OP_WITHDRAW)
        # upsert rules: valid_from + priority required; valid_to > valid_from.
        bad = pc.and_(is_upsert, pc.is_null(vf))
        if pc.any(bad).as_py():
            self._fail_row(shard_id, rg, base_row + self._first_index(bad),
                           VALID_FROM, "upsert must carry a valid_from")
        bad = pc.and_(is_upsert, pc.is_null(prio))
        if pc.any(bad).as_py():
            self._fail_row(shard_id, rg, base_row + self._first_index(bad),
                           PRIORITY, "upsert must carry a priority")
        bad = pc.and_(
            is_upsert,
            pc.and_(pc.is_valid(vt), pc.less_equal(vt, vf)),
        )
        if pc.any(bad).as_py():
            self._fail_row(shard_id, rg, base_row + self._first_index(bad),
                           VALID_TO, "must be strictly greater than valid_from")
        # withdraw rules: interval, priority and attached columns all null.
        for column, arr in ((VALID_FROM, vf), (VALID_TO, vt), (PRIORITY, prio)):
            bad = pc.and_(is_withdraw, pc.is_valid(arr))
            if pc.any(bad).as_py():
                self._fail_row(shard_id, rg, base_row + self._first_index(bad),
                               column, "withdraw must carry null here")
        for name in self.contract.dim_column_names:
            arr = batch.column(name)
            bad = pc.and_(is_withdraw, pc.is_valid(arr))
            if pc.any(bad).as_py():
                self._fail_row(shard_id, rg, base_row + self._first_index(bad),
                               name, "withdraw must carry null attached columns")

    # ------------------------------------------------------- record building
    def _build_revision_records(self, batch, shard_id, rg, base_row) -> pa.Table:
        n = batch.num_rows
        ekeys = batch.column(ENTITY_KEY).to_pylist()
        wids = batch.column(WINDOW_ID).to_pylist()
        recs = batch.column(RECORDED_AT).to_pylist()
        pkeys = [
            _revision_pkey(e, r, w) for e, r, w in zip(ekeys, recs, wids)
        ]
        arrays = {
            PKEY_FIELD: pa.array(pkeys, type=pa.binary()),
            WINDOW_ID: batch.column(WINDOW_ID),
            RECORDED_AT: batch.column(RECORDED_AT),
            OP: batch.column(OP),
            VALID_FROM: batch.column(VALID_FROM),
            VALID_TO: batch.column(VALID_TO),
            PRIORITY: batch.column(PRIORITY),
            "_shard": pa.array([shard_id] * n, type=pa.string()),
            "_rg": pa.array([rg] * n, type=pa.int32()),
            "_row": pa.array(range(base_row, base_row + n), type=pa.int64()),
        }
        for f in self.contract.dim_fields:
            arrays[f"d__{f.name}"] = to_carriage(batch.column(f.name), f.type)
        return pa.table(arrays, schema=self.rev_record_schema)

    def _build_fact_records(self, batch, shard_id, rg, base_row) -> pa.Table:
        n = batch.num_rows
        ekeys = batch.column(ENTITY_KEY).to_pylist()
        knowns = batch.column(KNOWN_AT).to_pylist()
        seq0 = self._fact_seq
        pkeys = [
            _fact_pkey(e, k, seq0 + i) for i, (e, k) in enumerate(zip(ekeys, knowns))
        ]
        self._fact_seq += n
        arrays = {
            PKEY_FIELD: pa.array(pkeys, type=pa.binary()),
            "_seq": pa.array(range(seq0, seq0 + n), type=pa.int64()),
            EVENT_AT: batch.column(EVENT_AT),
            KNOWN_AT: batch.column(KNOWN_AT),
            "_shard": pa.array([shard_id] * n, type=pa.string()),
            "_rg": pa.array([rg] * n, type=pa.int32()),
            "_row": pa.array(range(base_row, base_row + n), type=pa.int64()),
        }
        for f in self.contract.fact_fields:
            arrays[f"f__{f.name}"] = to_carriage(batch.column(f.name), f.type)
        return pa.table(arrays, schema=self.fact_record_schema)

    # ------------------------------------------------------------- matching
    def _run_rows(self, run: Run, names: list[str], chunk_rows: int = 8192):
        """Flatten a run into row tuples, converting in bounded chunks."""
        buf: list[pa.RecordBatch] = []
        buffered = 0
        for batch in run.iter_batches():
            buf.append(batch)
            buffered += batch.num_rows
            if buffered >= chunk_rows:
                yield from self._rows_of(buf, names)
                buf = []
                buffered = 0
        if buf:
            yield from self._rows_of(buf, names)

    @staticmethod
    def _rows_of(batches, names):
        table = pa.Table.from_batches(batches)
        cols = [table.column(name).to_pylist() for name in names]
        yield from zip(*cols)

    def _match(self, rev_run: Run, fact_run: Run, shard_identities) -> Run:
        result_builder = RunBuilder(
            "result", self.result_record_schema, self.request.memory_limit,
            self.spill_root, run_id=self.run_id,
            shard_identities=shard_identities,
        )
        self._result_builder = result_builder
        self._pending_results: list[tuple] = []
        rev_names = self.rev_record_schema.names
        fact_names = self.fact_record_schema.names
        rev_groups = _iter_groups(
            self._run_rows(rev_run, rev_names),
            key_of=lambda row: _ekey_of_pkey(row[0]),
        )
        fact_groups = _iter_groups(
            self._run_rows(fact_run, fact_names),
            key_of=lambda row: _ekey_of_pkey(row[0]),
        )
        revs = _Peekable(rev_groups)
        facts = _Peekable(fact_groups)
        while revs.head is not None or facts.head is not None:
            self._check_cancelled()
            if facts.head is None or (
                revs.head is not None and revs.head[0] < facts.head[0]
            ):
                ekey, rows = revs.head
                self._consume_revision_only_key(ekey, rows)
                revs.advance()
            elif revs.head is None or facts.head[0] < revs.head[0]:
                ekey, rows = facts.head
                self._emit_fact_only_key(rows)
                facts.advance()
            else:
                ekey = revs.head[0]
                self._process_key(ekey, revs.head[1], facts.head[1])
                revs.advance()
                facts.advance()
        self._flush_results(force=True)
        return result_builder.finish()

    def _duplicate_revision(self, ekey: bytes, prev: tuple, row) -> None:
        # ``prev`` is (recorded_at, window_id, shard_id, row_group, row).
        occurrences = [
            {"shard_id": prev[2], "row_group": prev[3], "row": prev[4]},
            {
                "shard_id": row[self.R_SHARD],
                "row_group": row[self.R_RG],
                "row": row[self.R_ROW],
            },
        ]
        raise TemporalDataError(
            "duplicate revision for "
            f"(entity_key={ekey.decode('utf-8', 'replace')!r}, "
            f"window_id={prev[1]!r}, recorded_at={prev[0]}): revisions of one "
            "window at one registration time are ambiguous",
            diagnostics=[{
                "entity_key": ekey.decode("utf-8", "replace"),
                "window_id": prev[1],
                "recorded_at": prev[0],
                "occurrences": occurrences,
            }],
        )

    def _consume_revision_only_key(self, ekey: bytes, rows) -> None:
        """Keys without facts still get their revisions fully validated."""
        self._stats["keys_total"] += 1
        self._stats["keys_revision_only"] += 1
        prev = None  # (recorded_at, window_id, shard, rg, row)
        for row in rows:
            cur = (row[self.R_REC], row[self.R_WID], row[self.R_SHARD],
                   row[self.R_RG], row[self.R_ROW])
            if prev is not None and cur[:2] == prev[:2]:
                self._duplicate_revision(ekey, prev, row)
            prev = cur

    def _emit_fact_only_key(self, rows) -> None:
        self._stats["keys_total"] += 1
        self._stats["keys_fact_only"] += 1
        for fact in rows:
            self._emit_result(fact, None, None)

    def _process_key(self, ekey: bytes, rev_rows, fact_rows) -> None:
        self._stats["keys_total"] += 1
        self._stats["keys_with_revisions_and_facts"] += 1
        keybuf = _KeyBuffer(
            self.spill_root, self.run_id, self._stats["keys_total"],
            self._key_buffer_limit, self.rev_record_schema,
            self.fact_record_schema, self.rev_record_schema.names,
            self.fact_record_schema.names, self._temp_meter,
        )
        try:
            coord_set: set[int] = set()
            upserts = 0
            prev = None  # (recorded_at, window_id, shard, rg, row)
            for row in rev_rows:
                cur = (row[self.R_REC], row[self.R_WID], row[self.R_SHARD],
                       row[self.R_RG], row[self.R_ROW])
                if prev is not None and cur[:2] == prev[:2]:
                    self._duplicate_revision(ekey, prev, row)
                prev = cur
                if row[self.R_OP] == OP_UPSERT:
                    coord_set.add(row[self.R_VF])
                    if row[self.R_VT] is not None:
                        coord_set.add(row[self.R_VT])
                    if len(coord_set) > self._coords_cap:
                        raise TemporalResourceError(
                            f"entity key {ekey.decode('utf-8', 'replace')!r} "
                            f"spans more than {self._coords_cap} distinct "
                            "window bounds; the per-key coordinate index "
                            "exceeds the metered budget share"
                        )
                    upserts += 1
                keybuf.add_rev(row)
            for row in fact_rows:
                keybuf.add_fact(row)
            if keybuf.spilled:
                self._key_buffers_spilled += 1
            if upserts == 0:
                for fact in keybuf.iter_facts():
                    self._emit_result(fact, None, None)
                return
            coords = sorted(coord_set)
            tree = StabbingMaxTree(coords)
            window_state: dict[str, _Episode] = {}
            rev_iter = keybuf.iter_revs()
            pending = next(rev_iter, None)
            for fact in keybuf.iter_facts():
                self._check_cancelled()
                known_at = fact[self.F_KNOWN]
                while pending is not None and pending[self.R_REC] <= known_at:
                    self._apply_revision(pending, window_state, tree)
                    pending = next(rev_iter, None)
                answer = tree.query(fact[self.F_EVENT])
                self._stats["stabbing_queries"] += 1
                self._emit_result(fact, answer, window_state)
            self._stats["tree_entries_peak"] = max(
                self._stats["tree_entries_peak"], tree.peak_entries
            )
        finally:
            keybuf.cleanup()

    def _apply_revision(self, rev, window_state, tree) -> None:
        wid = rev[self.R_WID]
        old = window_state.pop(wid, None)
        if old is not None:
            tree.remove(wid, old.valid_from, old.valid_to)
            self._stats["tree_removes"] += 1
        if rev[self.R_OP] == OP_UPSERT:
            episode = _Episode(
                valid_from=rev[self.R_VF],
                valid_to=rev[self.R_VT],
                priority=rev[self.R_PRIO],
                dims=tuple(rev[self.R_DIM0:self.R_DIM0
                               + len(self.contract.dim_fields)]),
                source=(rev[self.R_SHARD], rev[self.R_RG], rev[self.R_ROW]),
            )
            window_state[wid] = episode
            tree.insert(wid, episode.valid_from, episode.valid_to,
                        episode.priority)
            self._stats["episodes"] += 1
            if tree.entries > self._tree_entry_cap:
                raise TemporalResourceError(
                    f"active-window index holds {tree.entries} node "
                    f"memberships (cap {self._tree_entry_cap} from the "
                    "memory budget); the per-key working set exceeds the "
                    "metered budget share"
                )
            if len(window_state) > self._window_state_cap:
                raise TemporalResourceError(
                    f"{len(window_state)} simultaneously active windows "
                    f"(cap {self._window_state_cap} from the memory budget)"
                )
            if len(window_state) > self._stats["window_state_entries_peak"]:
                self._stats["window_state_entries_peak"] = len(window_state)

    # ------------------------------------------------------------ emission
    def _emit_result(self, fact, answer, window_state) -> None:
        seq = fact[self.F_SEQ]
        fact_values = fact[self.F_FACT0:self.F_FACT0
                           + len(self.contract.fact_fields)]
        fact_source = (fact[self.F_SHARD], fact[self.F_RG], fact[self.F_ROW])
        n_dim = len(self.contract.dim_fields)
        if answer is None:
            status, count = STATUS_UNMATCHED, 0
            dims = (None,) * n_dim
            sources = None
        else:
            _, count, (w1, w2) = answer
            if count == 1:
                episode = window_state[w1]
                status = STATUS_MATCHED
                dims = episode.dims
                sources = [self._source_json(episode.source, w1)]
            else:
                status = STATUS_AMBIGUOUS
                dims = (None,) * n_dim
                sources = [
                    self._source_json(window_state[w1].source, w1),
                    self._source_json(window_state[w2].source, w2),
                ]
        self._counts[status] += 1
        self._stats["result_records"] += 1
        self._pending_results.append((
            _result_pkey(seq),
            seq,
            status,
            count,
            *fact_values,
            *dims,
            *fact_source,
            sources,
        ))
        if len(self._pending_results) >= _RESULT_FLUSH_ROWS:
            self._flush_results()

    @staticmethod
    def _source_json(source, window_id) -> dict:
        return {
            "shard_id": source[0],
            "row_group": source[1],
            "row": source[2],
            "window_id": window_id,
        }

    def _flush_results(self, *, force: bool = False) -> None:
        rows = self._pending_results
        if not rows or (not force and len(rows) < _RESULT_FLUSH_ROWS):
            return
        self._pending_results = []
        cols = list(zip(*rows))
        arrays = {
            PKEY_FIELD: pa.array(list(cols[0]), type=pa.binary()),
            "_seq": pa.array(list(cols[1]), type=pa.int64()),
            "_status": pa.array(list(cols[2]), type=pa.string()),
            "_count": pa.array(list(cols[3]), type=pa.int64()),
        }
        pos = 4
        for f in self.contract.fact_fields:
            arrays[f"f__{f.name}"] = pa.array(
                list(cols[pos]), type=carriage_type(f.type)
            )
            pos += 1
        for f in self.contract.dim_fields:
            arrays[f"d__{f.name}"] = pa.array(
                list(cols[pos]), type=carriage_type(f.type)
            )
            pos += 1
        arrays["_fs_shard"] = pa.array(list(cols[pos]), type=pa.string())
        arrays["_fs_rg"] = pa.array(list(cols[pos + 1]), type=pa.int32())
        arrays["_fs_row"] = pa.array(list(cols[pos + 2]), type=pa.int64())
        arrays["_dim_sources"] = pa.array(
            list(cols[pos + 3]), type=self._dim_sources_type
        )
        table = pa.table(arrays, schema=self.result_record_schema)
        self._result_builder.add_table(table)

    # -------------------------------------------------------------- output
    def _output_schema(self, contract, shard_identities) -> pa.Schema:
        fields = [pa.field(f.name, f.type) for f in contract.fact_fields]
        fields.append(pa.field(OUT_STATUS, pa.string(), nullable=False))
        dim_struct = pa.struct([
            pa.field(f.name, f.type) for f in contract.dim_fields
        ])
        fields.append(pa.field(OUT_DIM, dim_struct))
        fields.append(pa.field(OUT_TOP_COUNT, pa.int64(), nullable=False))
        fields.append(pa.field(OUT_FACT_SOURCE, pa.struct([
            pa.field("shard_id", pa.string()),
            pa.field("row_group", pa.int32()),
            pa.field("row", pa.int64()),
        ]), nullable=False))
        fields.append(pa.field(OUT_DIM_SOURCES, self._dim_sources_type))
        metadata = {
            OUTPUT_METADATA_KEY.encode(): json.dumps({
                "format": "pqr.temporal_enrich",
                "format_version": 1,
                "shards": shard_identities,
                "fact_columns": list(self.request.fact_columns),
                "dim_columns": list(self.request.dim_columns),
                "reserved_columns": list(RESERVED_OUTPUT_COLUMNS),
            }, sort_keys=True).encode("utf-8"),
        }
        return pa.schema(fields, metadata=metadata)

    def _write_output(self, result_run: Run, schema: pa.Schema) -> None:
        out_dir = Path(os.path.dirname(self.request.output_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        staging = staging_path_for(out_dir)
        names = self.result_record_schema.names
        writer = None
        try:
            writer = pq.ParquetWriter(staging, schema, compression="snappy")
            for rows in self._batched(
                self._run_rows(result_run, names), _OUTPUT_BATCH_ROWS
            ):
                self._check_cancelled()
                writer.write_batch(self._output_batch(rows, schema))
            writer.close()
            writer = None
        except BaseException:
            if writer is not None:
                try:
                    writer.close()
                except Exception:  # noqa: BLE001 - original error wins
                    pass
            try:
                os.unlink(staging)
            except OSError:
                pass
            raise
        publish_parquet(
            staging_path=staging,
            output_path=self.request.output_path,
            expected_rows=self._stats["facts_read"],
            expected_schema=schema,
        )

    @staticmethod
    def _batched(rows, size: int):
        buf: list[tuple] = []
        for row in rows:
            buf.append(row)
            if len(buf) >= size:
                yield buf
                buf = []
        if buf:
            yield buf

    def _output_batch(self, rows, schema: pa.Schema) -> pa.RecordBatch:
        cols = list(zip(*rows))
        status = list(cols[self.O_STATUS])
        matched = [s == STATUS_MATCHED for s in status]
        arrays: list[pa.Array] = []
        pos = self.O_FACT0
        for f in self.contract.fact_fields:
            arrays.append(from_carriage_values(list(cols[pos]), f.type))
            pos += 1
        arrays.append(pa.array(status, type=pa.string()))
        dim_children = []
        pos = self.O_DIM0
        for f in self.contract.dim_fields:
            dim_children.append(from_carriage_values(list(cols[pos]), f.type))
            pos += 1
        dim_type = schema.field(OUT_DIM).type
        dim_mask = pa.array([not m for m in matched], type=pa.bool_())
        arrays.append(pa.StructArray.from_arrays(
            dim_children, fields=list(dim_type), mask=dim_mask
        ))
        arrays.append(pa.array(list(cols[self.O_COUNT]), type=pa.int64()))
        fs_type = schema.field(OUT_FACT_SOURCE).type
        arrays.append(pa.StructArray.from_arrays(
            [
                pa.array(list(cols[self.O_FS_SHARD]), type=pa.string()),
                pa.array(list(cols[self.O_FS_RG]), type=pa.int32()),
                pa.array(list(cols[self.O_FS_ROW]), type=pa.int64()),
            ],
            fields=list(fs_type),
        ))
        arrays.append(pa.array(
            list(cols[self.O_SOURCES]), type=self._dim_sources_type
        ))
        return pa.RecordBatch.from_arrays(arrays, schema=schema)

    # -------------------------------------------------------------- report
    def _build_report(self, contract, shard_identities, output_schema, *,
                      started, finished, recovery, runs) -> dict:
        rev_run, fact_run, result_run = runs
        ru = resource.getrusage(resource.RUSAGE_SELF)
        output_bytes = (
            os.path.getsize(self.request.output_path)
            if os.path.exists(self.request.output_path) else 0
        )
        spill = {
            "spill_root": str(self.spill_root),
            "segments_revision": rev_run.stats.spill_segments,
            "segments_fact": fact_run.stats.spill_segments,
            "segments_result": result_run.stats.spill_segments,
            "spilled_bytes": (
                rev_run.stats.spilled_bytes
                + fact_run.stats.spilled_bytes
                + result_run.stats.spilled_bytes
            ),
            "key_buffers_spilled": self._key_buffers_spilled,
            "temp_bytes_peak": self._temp_bytes_peak,
            "startup_recovery": recovery,
        }
        return {
            "report_format": REPORT_FORMAT,
            "spec_version": self.request.spec_version,
            "status": "completed",
            "started_at_utc": started,
            "finished_at_utc": finished,
            "run_id": self.run_id,
            "output": {
                "path": self.request.output_path,
                "rows": self._stats["facts_read"],
                "bytes": output_bytes,
            },
            "counts": {
                "facts": self._stats["facts_read"],
                "revisions": self._stats["revisions_read"],
                "matched": self._counts[STATUS_MATCHED],
                "ambiguous": self._counts[STATUS_AMBIGUOUS],
                "unmatched": self._counts[STATUS_UNMATCHED],
            },
            "columns": {
                "fact": list(self.request.fact_columns),
                "dim": list(self.request.dim_columns),
                "reserved": list(RESERVED_OUTPUT_COLUMNS),
            },
            "shards": shard_identities,
            "resources": {
                "network_cache": self.client.stats.snapshot(),
                "peak": {
                    "max_rss_bytes": ru.ru_maxrss * 1024,
                    "resident_record_bytes_peak": max(
                        rev_run.stats.peak_bytes_in_memory,
                        fact_run.stats.peak_bytes_in_memory,
                        result_run.stats.peak_bytes_in_memory,
                    ),
                    "tree_entries_peak": self._stats["tree_entries_peak"],
                    "window_state_entries_peak":
                        self._stats["window_state_entries_peak"],
                },
                "memory_limit_bytes": self.request.memory_limit,
                "spill": spill,
                "matching": {
                    "keys_total": self._stats["keys_total"],
                    "keys_with_revisions_and_facts":
                        self._stats["keys_with_revisions_and_facts"],
                    "keys_fact_only": self._stats["keys_fact_only"],
                    "keys_revision_only": self._stats["keys_revision_only"],
                    "episodes": self._stats["episodes"],
                    "tree_inserts": self._stats["episodes"],
                    "tree_removes": self._stats["tree_removes"],
                    "stabbing_queries": self._stats["stabbing_queries"],
                    "result_records": self._stats["result_records"],
                },
            },
        }

    # -------------------------------------------------------------- temp dir
    def _temp_meter(self, path: Path, delta: int) -> None:
        self._temp_bytes_current += delta
        if delta > 0:
            self._temp_bytes_peak = max(
                self._temp_bytes_peak, self._temp_bytes_current
            )

    def _sample_temp_peak(self) -> None:
        total = 0
        if self.spill_root.exists():
            for dirpath, _, filenames in os.walk(self.spill_root):
                for name in filenames:
                    try:
                        total += (Path(dirpath) / name).stat().st_size
                    except OSError:
                        pass
        self._temp_bytes_current = total
        self._temp_bytes_peak = max(self._temp_bytes_peak, total)

    def _cleanup_temp(self) -> None:
        if self._cleanup_done:
            return
        try:
            shutil.rmtree(self.spill_root, ignore_errors=True)
            base = self.spill_parent
            try:
                if base.exists() and not any(base.iterdir()):
                    base.rmdir()
            except OSError:
                pass
        except OSError:
            pass
        self._cleanup_done = True


def run_temporal_enrich(
    client: RemoteParquetClient, request: TemporalEnrichRequest, *, token=None
) -> dict:
    """Convenience functional entry point matching the class engine."""
    return TemporalEnrichEngine(client, request, token=token).run()
