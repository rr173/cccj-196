"""Version-diff engine: bounded extraction, external merge, deterministic facts.

Pipeline (no whole-object download, no full-table in-memory hold)::

    manifest pin  -> conditional identity proof for BOTH snapshots
                  -> schema contract / drift check (field-level diagnostics)
                  -> per side, row group by row group sparse scan of only the
                     declared key + compare columns
                  -> micro-batch key validation + canonical encoding
                  -> RunBuilder: resident records up to the memory ceiling,
                     then sorted checksummed spill segments
                  -> per-side k-way merge with within-snapshot uniqueness
                     enforcement (duplicate -> provenance-carrying error)
                  -> left/right alignment on canonical keys with Arrow
                     semantics (null vs empty container vs decoded dictionary
                     vs binary stay distinct)
                  -> streamed JSONL facts + audit report, atomically published

Every failure path removes (or quarantines) the run's temporary segments and
never replaces an existing successful result.
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

from .client import RemoteParquetClient
from .diff_drift import check_contract, schema_summary
from .diff_key import encode_scalar
from .diff_manifest import DiffRequest
from .diff_publish import (
    DiffFactWriter,
    KIND_DELETE,
    KIND_INSERT,
    KIND_NOCHANGE,
    KIND_UPDATE,
    build_report,
    publish,
)
from .diff_segment import (
    KMETA_FIELD,
    PKEY_FIELD,
    RG_FIELD,
    ROW_FIELD,
    Run,
    RunBuilder,
    cleanup_stale_runs,
    record_schema,
)
from .diff_values import scalar_to_json
from .exceptions import (
    DiffCancelled,
    DuplicateKeyError,
    NullKeyError,
    QueryCancelled,
)

# Micro-batch rows decoded/validated at once; bounds the per-step footprint
# independently of how large a Parquet row group is.
DEFAULT_MICROBATCH_ROWS = 4096


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclasses.dataclass(slots=True)
class _Record:
    pkey: bytes
    rg: int
    row: int                  # global row index within the snapshot
    values: dict[str, pa.Scalar]
    key_json: dict


class SideStream:
    """K-way merge of one side's sorted runs; enforces key uniqueness.

    Yields :class:`_Record` in canonical key order.  Equal keys across runs
    (or within one) are accumulated and raised as :class:`DuplicateKeyError`
    with every occurrence's identity / row group / row number.
    """

    def __init__(self, run: Run, side: str, identity, compare_columns: tuple[str, ...]):
        self.run = run
        self.side = side
        self.identity = identity
        self.compare_columns = compare_columns
        self._records_gen = None

    def _source_iterators(self):
        # Each run's batches are already sorted by pkey; flatten into record
        # iterators in canonical-key order with a bounded resident footprint.
        for batch in self.run.iter_batches():
            pkeys = batch.column(PKEY_FIELD)
            kmetas = batch.column(KMETA_FIELD)
            rgs = batch.column(RG_FIELD)
            rows = batch.column(ROW_FIELD)
            value_cols = {
                name: batch.column(name) for name in self.compare_columns
            }
            for i in range(batch.num_rows):
                yield (
                    pkeys[i].as_py(),
                    json.loads(kmetas[i].as_py()),
                    rgs[i].as_py(),
                    rows[i].as_py(),
                    {name: col[i] for name, col in value_cols.items()},
                )

    def close(self) -> None:
        """Deterministically release any open segment readers/handles."""
        # Closing the source generator unwinds its finally blocks, which close
        # the underlying lazy segment streams and their file handles.
        gen = self._records_gen
        if gen is not None:
            try:
                gen.close()
            except Exception:  # noqa: BLE001 - best-effort release
                pass
        self._records_gen = None

    def records(self):
        # The run already guarantees globally sorted output (segments sorted +
        # bounded k-way merge); this generator only enforces uniqueness.
        self._records_gen = self._records_impl()
        return self._records_gen

    def _records_impl(self):
        current_key: bytes | None = None
        group: list[_Record] = []
        for pkey, kmeta, rg, row, values in self._source_iterators():
            if current_key is not None and pkey != current_key:
                yield from self._emit(current_key, group)
                group = []
            current_key = pkey
            group.append(
                _Record(pkey=pkey, rg=rg, row=row, values=values, key_json=kmeta)
            )
        if group:
            yield from self._emit(current_key, group)

    def _emit(self, pkey: bytes, group: list[_Record]):
        if len(group) > 1:
            occurrences = [
                {
                    "side": self.side,
                    "identity": {
                        "url": self.identity.url,
                        "etag": self.identity.etag,
                        "length": self.identity.length,
                        "version_id": self.identity.version_id(),
                    },
                    "row_group": rec.rg,
                    "row": rec.row,
                    "key_hex": pkey.hex(),
                }
                for rec in group
            ]
            raise DuplicateKeyError(
                f"duplicate business key {pkey.hex()!r} on {self.side} side: "
                f"{len(group)} occurrences (identity/row-group/row recorded)",
                occurrences=occurrences,
            )
        yield group[0]


def _decode_dictionaries(table: pa.Table) -> pa.Table:
    """Decode every dictionary column to its plain value type.

    Diff semantics compare decoded values; keeping the dictionary encoding
    would make two equal values with different index layouts look different.
    """
    new_cols = []
    for col in table.columns:
        if pa.types.is_dictionary(col.type):
            new_cols.append(col.cast(col.type.value_type))
        else:
            new_cols.append(col)
    return pa.table(dict(zip(table.column_names, new_cols)))


class DiffEngine:
    def __init__(
        self,
        client: RemoteParquetClient,
        request: DiffRequest,
        *,
        spill_dir: str | None = None,
        microbatch_rows: int = DEFAULT_MICROBATCH_ROWS,
        token=None,
    ) -> None:
        self.client = client
        self.request = request
        self.microbatch_rows = microbatch_rows
        self.token = token
        self.run_id = f"diff-{uuid.uuid4().hex}"
        # Temp storage lives next to the output by default, so publication can
        # use same-filesystem atomic renames.
        base = Path(spill_dir or os.path.join(
            os.path.dirname(request.diff_path), ".pqr-diff-tmp"
        ))
        self.spill_root = base / self.run_id
        self.quarantine_dir = base / "quarantine"
        self._cleanup_done = False

    # ------------------------------------------------------------ public run
    def run(self) -> dict:
        started = _utc_now()
        req = self.request
        self.spill_root.mkdir(parents=True, exist_ok=True)
        # Process-termination recovery: never trust segments from other runs.
        self.recovery = cleanup_stale_runs(
            self.spill_root.parent, current_run_id=self.run_id
        )
        try:
            left_obj = self.client.open_identity(
                req.left.url, req.left.etag, req.left.length
            )
            right_obj = self.client.open_identity(
                req.right.url, req.right.etag, req.right.length
            )
            left_info = left_obj.inspect(token=self.token)
            right_info = right_obj.inspect(token=self.token)
            key_field, compare_fields = check_contract(
                left_info.schema,
                right_info.schema,
                req.key_column,
                req.compare_columns,
            )
            columns = [req.key_column, *req.compare_columns]
            left_run, left_meta = self._extract_side(
                "left", left_obj, columns, compare_fields, key_field
            )
            right_run, right_meta = self._extract_side(
                "right", right_obj, columns, compare_fields, key_field
            )
            counts, value_summary, facts_checksum, facts_records, facts_bytes = (
                self._merge_and_write(
                    left_run, right_run, left_obj, right_obj, compare_fields
                )
            )
            finished = _utc_now()
            report = build_report(
                request_json=req.as_json(),
                left_identity=_identity_json(left_obj),
                right_identity=_identity_json(right_obj),
                left_schema=schema_summary(left_info.schema),
                right_schema=schema_summary(right_info.schema),
                key_column=req.key_column,
                compare_columns=req.compare_columns,
                input_rows={
                    "left": left_meta["input_rows"],
                    "right": right_meta["input_rows"],
                },
                unique_keys={
                    "left": left_run.record_count,
                    "right": right_run.record_count,
                },
                counts=counts,
                value_summary=value_summary,
                stats=self.client.stats.snapshot(),
                peak_resources=self._peak_resources(left_run, right_run),
                spill=self._spill_stats(left_run, right_run),
                facts_checksum=facts_checksum,
                facts_records=facts_records,
                facts_bytes=facts_bytes,
                started_at=started,
                finished_at=finished,
            )
            publish(
                diff_path=req.diff_path,
                report_path=req.report_path,
                report=report,
                facts_tmp_path=self._facts_tmp,
                expected_records=facts_records,
                expected_checksum=facts_checksum,
            )
            return report
        except (QueryCancelled,):
            raise
        except BaseException:
            raise
        finally:
            # Every exit path (success, cancellation, protocol/drift/duplicate
            # failure, write error) removes this run's temporary segments; the
            # quarantine directory is retained for forensic inspection.
            self._cleanup_temp()

    # --------------------------------------------------------- extraction
    def _check_cancelled(self) -> None:
        if self.token is not None and getattr(self.token, "cancelled", False):
            raise DiffCancelled("diff run cancelled")

    def _extract_side(self, side, obj, columns, compare_fields, key_field):
        info = obj.inspect(token=self.token)
        n_groups = info.num_row_groups
        schema = record_schema([
            pa.field(f.name, _stored_type(f.type)) for f in compare_fields
        ])
        builder = RunBuilder(
            side=side,
            schema=schema,
            memory_limit=self.request.memory_limit,
            spill_dir=self.spill_root,
            left=self._left_identity(side, obj),
            right=self._right_identity(side, obj),
            run_id=self.run_id,
        )
        input_rows = 0
        global_row = 0
        for rg in range(n_groups):
            self._check_cancelled()
            table = obj.scan(columns=columns, row_groups=[rg], token=self.token)
            table = _decode_dictionaries(table)
            group_rows = table.num_rows
            for start in range(0, group_rows, self.microbatch_rows):
                self._check_cancelled()
                part = table.slice(start, self.microbatch_rows)
                record_table = self._build_record_table(
                    part, rg, global_row + start, schema, compare_fields
                )
                builder.add_table(record_table)
            input_rows += group_rows
            global_row += group_rows
        run = builder.finish()
        return run, {"input_rows": input_rows}

    def _left_identity(self, side, obj):
        return obj.identity if side == "left" else None

    def _right_identity(self, side, obj):
        return obj.identity if side == "right" else None

    def _build_record_table(self, part, rg, base_row, schema, compare_fields):
        n = part.num_rows
        key_array = part.column(self.request.key_column)
        pkeys: list[bytes] = []
        kmetas: list[str] = []
        # Key validation: non-null, encodable. Uniqueness is proven at merge.
        for i in range(n):
            scalar = key_array[i]
            if not scalar.is_valid:
                raise NullKeyError(
                    f"null business key in row group {rg} row {base_row + i} "
                    f"of column {self.request.key_column!r}"
                )
            canonical = encode_scalar(scalar)
            pkeys.append(canonical.encoded)
            kmetas.append(json.dumps(
                {"type": canonical.type_name, "value": canonical.json_value},
                ensure_ascii=False,
            ))
        arrays = {
            PKEY_FIELD: pa.array(pkeys, type=pa.binary()),
            KMETA_FIELD: pa.array(kmetas, type=pa.string()),
            RG_FIELD: pa.array([rg] * n, type=pa.int32()),
            ROW_FIELD: pa.array(list(range(base_row, base_row + n)), type=pa.int64()),
        }
        for f in compare_fields:
            arrays[f.name] = part.column(f.name)
        return pa.table(arrays, schema=schema)

    # ------------------------------------------------------------- merging
    def _merge_and_write(self, left_run, right_run, left_obj, right_obj,
                         compare_fields) -> tuple[dict, dict, str, int, int]:
        req = self.request
        out_dir = Path(os.path.dirname(req.diff_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        writer = DiffFactWriter(out_dir, req.compare_columns)
        self._facts_tmp = writer.tmp_path
        counts = {KIND_INSERT: 0, KIND_DELETE: 0, KIND_UPDATE: 0, KIND_NOCHANGE: 0}
        value_summary = self._empty_value_summary(compare_fields)

        left_stream = SideStream(left_run, "left", left_obj.identity,
                                 req.compare_columns)
        right_stream = SideStream(right_run, "right", right_obj.identity,
                                  req.compare_columns)
        left_iter = iter(left_stream.records())
        right_iter = iter(right_stream.records())
        lrec = next(left_iter, None)
        rrec = next(right_iter, None)
        try:
            while lrec is not None or rrec is not None:
                self._check_cancelled()
                if rrec is None or (lrec is not None and lrec.pkey < rrec.pkey):
                    self._emit(writer, KIND_DELETE, lrec, None, counts, value_summary)
                    lrec = next(left_iter, None)
                elif lrec is None or rrec.pkey < lrec.pkey:
                    self._emit(writer, KIND_INSERT, None, rrec, counts, value_summary)
                    rrec = next(right_iter, None)
                else:
                    assert lrec is not None and rrec is not None
                    if self._records_equal(lrec, rrec, compare_fields):
                        self._emit(writer, KIND_NOCHANGE, lrec, rrec, counts,
                                   value_summary)
                    else:
                        self._emit(writer, KIND_UPDATE, lrec, rrec, counts,
                                   value_summary)
                    lrec = next(left_iter, None)
                    rrec = next(right_iter, None)
        except BaseException:
            writer.discard()
            raise
        finally:
            # Deterministically release segment readers/file handles on every
            # path (success, cancellation, duplicate-key or protocol failure).
            left_stream.close()
            right_stream.close()
        tmp_path, facts_records, checksum = writer.close_and_checksum()
        facts_bytes = os.path.getsize(tmp_path)
        return counts, value_summary, checksum, facts_records, facts_bytes

    def _emit(self, writer, kind, lrec, rrec, counts, value_summary):
        if kind == KIND_INSERT:
            rec = rrec
            before = None
            after = self._row_json(rec)
        elif kind == KIND_DELETE:
            rec = lrec
            before = self._row_json(rec)
            after = None
        else:
            rec = lrec
            before = self._row_json(lrec)
            after = self._row_json(rrec)
        key_json = self._key_json(rec)
        writer.write(kind, key_json, before, after, rec.pkey)
        counts[kind] += 1
        self._accumulate_summary(value_summary, lrec, rrec, kind)

    def _row_json(self, rec: _Record):
        from collections import OrderedDict

        out = OrderedDict()
        for name in self.request.compare_columns:
            out[name] = scalar_to_json(rec.values[name])
        return out

    def _key_json(self, rec: _Record) -> dict:
        return {
            "type": rec.key_json["type"],
            "value": rec.key_json["value"],
            "hex": rec.pkey.hex(),
        }

    def _records_equal(self, lrec, rrec, compare_fields) -> bool:
        for f in compare_fields:
            name = f.name
            ls, rs = lrec.values[name], rrec.values[name]
            # Arrow scalar equality: null == null, empty container != null,
            # binary compared byte-wise, dictionary already decoded.
            if not ls.equals(rs):
                return False
        return True

    def _empty_value_summary(self, compare_fields) -> dict:
        summary = {}
        for f in compare_fields:
            summary[f.name] = {
                "arrow_type": str(_stored_type(f.type)),
                "dictionary_encoded_in_source": pa.types.is_dictionary(f.type),
                "nulls_left": 0,
                "nulls_right": 0,
                "empty_containers_left": 0,
                "empty_containers_right": 0,
            }
        return summary

    def _accumulate_summary(self, summary, lrec, rrec, kind) -> None:
        for name in self.request.compare_columns:
            entry = summary[name]
            if lrec is not None:
                scalar = lrec.values[name]
                if not scalar.is_valid:
                    entry["nulls_left"] += 1
                elif _is_empty_container(scalar):
                    entry["empty_containers_left"] += 1
            if rrec is not None:
                scalar = rrec.values[name]
                if not scalar.is_valid:
                    entry["nulls_right"] += 1
                elif _is_empty_container(scalar):
                    entry["empty_containers_right"] += 1

    # ------------------------------------------------------------- resources
    def _spill_stats(self, left_run, right_run) -> dict:
        return {
            "spill_root": str(self.spill_root),
            "segments_left": left_run.stats.spill_segments,
            "segments_right": right_run.stats.spill_segments,
            "spilled_bytes": (
                left_run.stats.spilled_bytes + right_run.stats.spilled_bytes
            ),
            "resident_record_bytes_peak": max(
                left_run.stats.peak_bytes_in_memory,
                right_run.stats.peak_bytes_in_memory,
            ),
            "startup_recovery": getattr(self, "recovery",
                                        {"removed": 0, "quarantined": 0}),
        }

    def _peak_resources(self, left_run, right_run) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "max_rss_bytes": ru.ru_maxrss * 1024,  # Linux reports KiB
            "max_resident_record_bytes": max(
                left_run.stats.peak_bytes_in_memory,
                right_run.stats.peak_bytes_in_memory,
            ),
            "memory_limit_bytes": self.request.memory_limit,
        }

    def _cleanup_temp(self) -> None:
        if self._cleanup_done:
            return
        # Remove the whole per-run directory; quarantine is kept for forensics.
        try:
            shutil.rmtree(self.spill_root, ignore_errors=True)
            base = self.spill_root.parent
            # Remove the now-empty temp root when this run created the only one.
            try:
                if base.exists() and not any(base.iterdir()):
                    base.rmdir()
            except OSError:
                pass
        except OSError:
            pass
        self._cleanup_done = True


def _stored_type(t: pa.DataType) -> pa.DataType:
    """Type used inside segment records (dictionary decoded to value type)."""
    if pa.types.is_dictionary(t):
        return t.value_type
    return t


def _is_empty_container(scalar: pa.Scalar) -> bool:
    t = scalar.type
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return scalar.is_valid and len(scalar.as_py()) == 0
    if pa.types.is_struct(t):
        return scalar.is_valid and len(scalar.as_py()) == 0
    return False


def _identity_json(obj) -> dict:
    return {
        "url": obj.identity.url,
        "etag": obj.identity.etag,
        "length": obj.identity.length,
        "version_id": obj.identity.version_id(),
    }


def run_diff(client: RemoteParquetClient, request: DiffRequest, *,
             spill_dir: str | None = None, token=None) -> dict:
    """Convenience functional entry point matching the class engine."""
    return DiffEngine(client, request, spill_dir=spill_dir, token=token).run()
