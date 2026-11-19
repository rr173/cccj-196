"""Single-snapshot quality-audit engine.

Pipeline (no whole-object download, no full-table in-memory hold)::

    request pin   -> conditional identity proof of the exact object version
                  -> footer read; contract matched against the top-level schema
                     (exact Arrow types, by name, order-independent)
                  -> per row group: prefetch only the referenced columns'
                     chunks, decode micro-batches, evaluate every rule in
                     contract order, stream violation records
                  -> violations JSONL + JSON report published as one
                     recoverable unit

Every failure path (contract/schema/identity/protocol/cancellation/write)
removes this run's staging files and never touches a previously published
result; a restart recovers or quarantines an interrupted publish via
:func:`parquet_range.quality_publish.recover_publish_unit`.
"""

from __future__ import annotations

import collections
import os
import resource
import time
from pathlib import Path

from .client import RemoteParquetClient
from .exceptions import AuditCancelled, AuditPublishError
from .quality_publish import (
    ViolationWriter,
    build_report,
    publish,
    recover_publish_unit,
)
from .quality_request import QualityAuditRequest
from .quality_rules import build_executor


def _utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class QualityAuditEngine:
    """Runs one quality audit against one pinned object version."""

    def __init__(
        self,
        client: RemoteParquetClient,
        request: QualityAuditRequest,
        *,
        token=None,
    ) -> None:
        self.client = client
        self.request = request
        self.token = token
        self._writer: ViolationWriter | None = None

    # ------------------------------------------------------------ public run
    def run(self) -> dict:
        started = _utc_now()
        req = self.request
        # The publish unit requires both outputs in one directory; check
        # before any I/O so a misconfigured request fails immediately.
        if os.path.dirname(req.violations_path) != os.path.dirname(req.report_path):
            raise AuditPublishError(
                "violations_path and report_path must live in the same "
                f"directory for atomic publication: {req.violations_path!r} "
                f"vs {req.report_path!r}"
            )
        out_dir = Path(os.path.dirname(req.violations_path))
        out_dir.mkdir(parents=True, exist_ok=True)
        # Recover/clean any interrupted previous publish of this unit before
        # doing anything else; never deletes the last valid audit.
        recovery = recover_publish_unit(
            out_dir, req.violations_path, req.report_path
        )
        try:
            obj = self.client.open_identity(
                req.object.url, req.object.etag, req.object.length
            )
            info = obj.inspect(token=self.token)
            # Contract <-> schema exact match, before any row data is read.
            matched = req.contract.match_schema(info.schema)
            columns = req.contract.referenced_fields()
            executors = [
                build_executor(rule, req.contract.fields)
                for rule in req.contract.rules
            ]
            identity_json = {
                "url": obj.identity.url,
                "etag": obj.identity.etag,
                "length": obj.identity.length,
                "version_id": obj.identity.version_id(),
            }
            try:
                writer = ViolationWriter(out_dir)
            except OSError as exc:
                raise AuditPublishError(
                    f"cannot stage violations in {out_dir}: {exc!r}"
                ) from exc
            self._writer = writer
            rows_scanned = 0
            row_offsets = self._row_offsets(info.row_group_rows)
            non_empty = [rg for rg in range(info.num_row_groups)
                         if info.row_group_rows[rg] > 0]
            if columns and non_empty:
                try:
                    batch_iter = self.client.reader.scan_batches(
                        obj.identity, columns, non_empty,
                        batch_size=req.batch_rows, token=self.token,
                    )
                    current_rg = -1
                    base_row = 0
                    for rg, batch in batch_iter:
                        self._check_cancelled()
                        if rg != current_rg:
                            current_rg = rg
                            base_row = row_offsets[rg]
                        self._evaluate_batch(
                            executors, batch, rg, base_row, writer,
                            identity_json,
                        )
                        base_row += batch.num_rows
                        rows_scanned += batch.num_rows
                        del batch
                except OSError as exc:
                    raise AuditPublishError(
                        f"failed writing staged violations: {exc!r}"
                    ) from exc
            else:
                # Empty table (no non-empty row groups): nothing to fetch.
                rows_scanned = info.num_rows
            row_groups_scanned = len(non_empty)
            try:
                tmp_path, records, checksum = writer.close_and_checksum()
            except OSError as exc:
                raise AuditPublishError(
                    f"cannot flush staged violations: {exc!r}"
                ) from exc
            self._writer = None
            violations_bytes = os.path.getsize(tmp_path)
            finished = _utc_now()
            per_rule = collections.OrderedDict(
                (
                    ex.rule_id,
                    collections.OrderedDict([
                        ("violations", writer.per_rule.get(ex.rule_id, 0)),
                        ("null_skipped", ex.null_skipped),
                        ("nan_skipped", ex.nan_skipped),
                    ]),
                )
                for ex in executors
            )
            report = build_report(
                identity=identity_json,
                contract_sha256=req.contract.sha256,
                rule_order=req.contract.rule_order(),
                schema_summary={
                    **req.contract.schema_summary(matched),
                    "file_num_fields": len(info.schema),
                    "file_num_rows": info.num_rows,
                },
                rows_scanned=rows_scanned,
                row_groups_scanned=row_groups_scanned,
                row_groups_total=info.num_row_groups,
                per_rule=per_rule,
                total_violations=records,
                stats=self.client.stats.snapshot(),
                peak_resources=self._peak_resources(),
                recovery=recovery,
                violations_path=req.violations_path,
                violations_records=records,
                violations_bytes=violations_bytes,
                violations_checksum=checksum,
                started_at=started,
                finished_at=finished,
            )
            publish(
                violations_path=req.violations_path,
                report_path=req.report_path,
                report=report,
                violations_tmp_path=tmp_path,
                expected_records=records,
                expected_checksum=checksum,
            )
            return report
        finally:
            # Any exit before a successful publish drops the staging file.
            if self._writer is not None:
                self._writer.discard()
                self._writer = None

    # ----------------------------------------------------------- evaluation
    def _check_cancelled(self) -> None:
        if self.token is not None and getattr(self.token, "cancelled", False):
            raise AuditCancelled("quality audit cancelled")

    @staticmethod
    def _row_offsets(row_group_rows: list[int]) -> list[int]:
        offsets: list[int] = []
        running = 0
        for n in row_group_rows:
            offsets.append(running)
            running += n
        return offsets

    def _evaluate_batch(
        self, executors, batch, rg: int, base_row: int,
        writer: ViolationWriter, identity_json: dict,
    ) -> None:
        columns = {
            name: batch.column(name) for name in batch.schema.names
        }
        checks = [executor.prepare(columns) for executor in executors]
        for row_idx in range(batch.num_rows):
            global_row = base_row + row_idx
            for rule_index, check in enumerate(checks):
                outcome = check(row_idx)
                if outcome is None:
                    continue
                reason, observed = outcome
                executor = executors[rule_index]
                record = collections.OrderedDict()
                record["identity"] = identity_json
                record["rule_id"] = executor.rule_id
                record["rule_kind"] = executor.kind
                record["fields"] = list(executor.fields)
                record["row_group"] = rg
                record["row"] = global_row
                record["observed"] = observed
                record["reason"] = reason
                writer.write(record, (rg, global_row, rule_index))

    # ------------------------------------------------------------- resources
    def _peak_resources(self) -> dict:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return {
            "max_rss_bytes": ru.ru_maxrss * 1024,  # Linux reports KiB
            "batch_rows": self.request.batch_rows,
        }


def run_quality_audit(
    client: RemoteParquetClient, request: QualityAuditRequest, *, token=None
) -> dict:
    """Convenience functional entry point matching the class engine."""
    return QualityAuditEngine(client, request, token=token).run()
