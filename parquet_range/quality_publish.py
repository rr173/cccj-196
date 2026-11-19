"""Atomic publication of quality-audit results and the offline reader.

The violations JSONL and the audit report are one *recoverable publish
unit*:

1. both are written to staging temp files in the destination directory,
   flushed and fsynced;
2. the violations file is re-read from disk and its SHA-256 / record count /
   per-rule counts are verified against the report about to be published;
3. the pair is swapped into place with ``.old`` backups and full rollback,
   so a failure never leaves a half-new pair and never destroys the previous
   successful audit;
4. :func:`recover_publish_unit` runs before every new audit: it removes
   staging leftovers, restores an interrupted swap back to the previous
   valid pair, and quarantines anything inconsistent -- a restart after a
   crash can never mistake an incomplete publish for a successful one, and
   never deletes the last good audit.

The offline reader :func:`read_violations` re-verifies the report, the
violations hash, the deterministic record order and every count *without any
network access*, then serves a paginated view of the violations by rule.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import tempfile
import uuid
from pathlib import Path

from .exceptions import AuditPublishError
from .quality_request import SPEC_VERSION

REPORT_FORMAT = "pqr-quality-report"
# Fixed JSONL record field order (JSON objects preserve insertion order).
RECORD_ORDER = ("identity", "rule_id", "rule_kind", "fields", "row_group",
                "row", "observed", "reason")

_STAGING_PREFIXES = (".quality-violations-", ".quality-report-")


def _atomic_targets(violations_path: str, report_path: str):
    violations_path = os.path.abspath(violations_path)
    report_path = os.path.abspath(report_path)
    if os.path.dirname(violations_path) != os.path.dirname(report_path):
        raise AuditPublishError(
            "violations_path and report_path must live in the same directory "
            f"for atomic publication: {violations_path!r} vs {report_path!r}"
        )
    out_dir = Path(os.path.dirname(violations_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    return violations_path, report_path, out_dir


class ViolationWriter:
    """Streams violation records to a staging file; tracks counts/checksum."""

    def __init__(self, out_dir: Path) -> None:
        fd, self.tmp_path = tempfile.mkstemp(
            prefix=".quality-violations-", suffix=".jsonl.tmp", dir=str(out_dir)
        )
        os.close(fd)
        self._fh = open(self.tmp_path, "w", encoding="utf-8", newline="\n")
        self.checksum = hashlib.sha256()
        self.records_written = 0
        self.per_rule: dict[str, int] = collections.Counter()
        self._last_triple: tuple[int, int, int] | None = None

    def write(self, record: dict, sort_triple: tuple[int, int, int]) -> None:
        # Deterministic-order invariant: (row_group, row, rule order index)
        # must be strictly increasing -- one record per (row, rule) pair.
        if self._last_triple is not None and sort_triple <= self._last_triple:
            raise AuditPublishError(
                "internal: violations emitted out of (row_group, row, rule) order"
            )
        self._last_triple = sort_triple
        if tuple(record.keys()) != RECORD_ORDER:  # fixed field-order contract
            raise AuditPublishError("internal: violation record field order broken")
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        payload = line + "\n"
        self._fh.write(payload)
        self.checksum.update(payload.encode("utf-8"))
        self.per_rule[record["rule_id"]] += 1
        self.records_written += 1

    def close_and_checksum(self) -> tuple[str, int, str]:
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._fh.close()
        return self.tmp_path, self.records_written, self.checksum.hexdigest()

    def discard(self) -> None:
        try:
            if not self._fh.closed:
                self._fh.close()
        finally:
            _silent_unlink(self.tmp_path)


def build_report(
    *,
    identity: dict,
    contract_sha256: str,
    rule_order: list[str],
    schema_summary: dict,
    rows_scanned: int,
    row_groups_scanned: int,
    row_groups_total: int,
    per_rule: dict,
    total_violations: int,
    stats: dict,
    peak_resources: dict,
    recovery: dict,
    violations_path: str,
    violations_records: int,
    violations_bytes: int,
    violations_checksum: str,
    started_at: str,
    finished_at: str,
) -> dict:
    return collections.OrderedDict([
        ("report_format", REPORT_FORMAT),
        ("spec_version", SPEC_VERSION),
        ("status", "completed"),
        ("started_at_utc", started_at),
        ("finished_at_utc", finished_at),
        ("identity", identity),
        ("contract_sha256", contract_sha256),
        ("rules", list(rule_order)),
        ("schema", schema_summary),
        ("rows_scanned", rows_scanned),
        ("row_groups_scanned", row_groups_scanned),
        ("row_groups_total", row_groups_total),
        ("violations_per_rule", per_rule),
        ("total_violations", total_violations),
        ("resources", {
            "network_cache": stats,
            "peak": peak_resources,
            "startup_recovery": recovery,
        }),
        ("violations", {
            "path": violations_path,
            "records": violations_records,
            "bytes": violations_bytes,
            "sha256": violations_checksum,
        }),
    ])


def publish(
    *,
    violations_path: str,
    report_path: str,
    report: dict,
    violations_tmp_path: str,
    expected_records: int,
    expected_checksum: str,
) -> None:
    """Verify the staging files, then atomically replace the target pair."""
    violations_path, report_path, out_dir = _atomic_targets(
        violations_path, report_path
    )
    # Re-verify the flushed violations file from disk before touching targets.
    h = hashlib.sha256()
    line_count = 0
    with open(violations_tmp_path, "rb") as fh:
        for line in fh:
            h.update(line)
            line_count += 1
            json.loads(line.decode("utf-8"))  # every line must parse
    if h.hexdigest() != expected_checksum or line_count != expected_records:
        _silent_unlink(violations_tmp_path)
        raise AuditPublishError(
            "violations verification failed before publication: "
            "checksum/count mismatch"
        )
    if report["violations"]["sha256"] != expected_checksum:
        raise AuditPublishError("report checksum disagrees with violations file")
    if report["violations"]["records"] != expected_records:
        raise AuditPublishError("report record count disagrees with violations file")
    if report["total_violations"] != expected_records:
        raise AuditPublishError("report total does not match violations records")
    counted = sum(entry["violations"] for entry in report["violations_per_rule"].values())
    if counted != expected_records:
        raise AuditPublishError("per-rule counts do not sum to violations records")

    fd, report_tmp = tempfile.mkstemp(
        prefix=".quality-report-", suffix=".json.tmp", dir=str(out_dir)
    )
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    # Two-step replace with rollback: violations first, then report.  A
    # failure anywhere restores the previous pair (if any) completely.
    old_violations = violations_path + ".old"
    old_report = report_path + ".old"
    replaced_violations = False
    replaced_report = False
    try:
        if os.path.exists(violations_path):
            os.replace(violations_path, old_violations)
            replaced_violations = True
        os.replace(violations_tmp_path, violations_path)
        try:
            if os.path.exists(report_path):
                os.replace(report_path, old_report)
                replaced_report = True
            os.replace(report_tmp, report_path)
        except OSError:
            _silent_unlink(report_path)
            if replaced_report and os.path.exists(old_report):
                os.replace(old_report, report_path)
            _silent_unlink(violations_path)
            if replaced_violations and os.path.exists(old_violations):
                os.replace(old_violations, violations_path)
            raise
        _silent_unlink(old_violations)
        _silent_unlink(old_report)
    except OSError as exc:
        _silent_unlink(violations_tmp_path)
        _silent_unlink(report_tmp)
        _silent_unlink(old_violations)
        _silent_unlink(old_report)
        raise AuditPublishError(f"atomic publication failed: {exc!r}") from exc
    _fsync_dir(out_dir)


# ------------------------------------------------------------- crash recovery
def recover_publish_unit(
    out_dir: str | os.PathLike[str],
    violations_path: str,
    report_path: str,
) -> dict:
    """Clean/isolate unfinished publish artifacts; never delete a valid audit.

    Returns a small JSON-able summary for the next report's audit trail.
    """
    out_dir = Path(out_dir)
    summary: dict = {"removed_staging": 0, "restored": [], "quarantined": []}
    if not out_dir.exists():
        return summary
    violations_path = os.path.abspath(violations_path)
    report_path = os.path.abspath(report_path)
    v_old = violations_path + ".old"
    r_old = report_path + ".old"

    # 1. Staging temp files are incomplete by definition: remove them.
    for entry in out_dir.iterdir():
        if entry.name.startswith(_STAGING_PREFIXES) and entry.name.endswith(".tmp"):
            _silent_unlink(entry)
            summary["removed_staging"] += 1

    have_v_old, have_r_old = os.path.exists(v_old), os.path.exists(r_old)
    have_v, have_r = os.path.exists(violations_path), os.path.exists(report_path)

    if have_v_old or have_r_old:
        # A previous publish was interrupted mid-swap.
        if have_v and have_r and _pair_consistent(violations_path, report_path):
            # Both swaps had completed; only the backup cleanup was lost.
            _silent_unlink(v_old)
            _silent_unlink(r_old)
            return summary
        # Restore the previous valid pair; quarantine half-published new files.
        for target, backup in ((violations_path, v_old), (report_path, r_old)):
            if os.path.exists(backup):
                if os.path.exists(target):
                    _quarantine(out_dir, target, summary)
                os.replace(backup, target)
                summary["restored"].append(os.path.basename(target))
        if not (
            os.path.exists(violations_path)
            and os.path.exists(report_path)
            and _pair_consistent(violations_path, report_path)
        ):
            # The pair is unrecoverable: leave nothing that looks successful.
            for target in (violations_path, report_path):
                if os.path.exists(target):
                    _quarantine(out_dir, target, summary)
        return summary

    # No backups: the pair must be complete and consistent, or quarantined.
    if have_v and have_r:
        if not _pair_consistent(violations_path, report_path):
            _quarantine(out_dir, violations_path, summary)
            _quarantine(out_dir, report_path, summary)
    elif have_v or have_r:
        _quarantine(out_dir, violations_path if have_v else report_path, summary)
    return summary


def _pair_consistent(violations_path: str, report_path: str) -> bool:
    """A report+violations pair is consistent iff the report verifies."""
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
        if report.get("status") != "completed":
            return False
        expected = report["violations"]
        h = hashlib.sha256()
        records = 0
        with open(violations_path, "rb") as fh:
            for line in fh:
                h.update(line)
                records += 1
        return (
            h.hexdigest() == expected["sha256"]
            and records == expected["records"]
            and report.get("total_violations") == records
        )
    except Exception:  # noqa: BLE001 - any inconsistency means untrustworthy
        return False


def _quarantine(out_dir: Path, path: str, summary: dict) -> None:
    """Move an untrustworthy artifact aside; never silently delete data."""
    quarantine_dir = out_dir / "quarantine"
    quarantine_dir.mkdir(exist_ok=True)
    dest = quarantine_dir / f"{os.path.basename(path)}.{uuid.uuid4().hex[:12]}"
    try:
        os.replace(path, dest)
        summary["quarantined"].append(str(dest))
    except OSError:
        pass


# ------------------------------------------------------------- offline reader
def read_violations(
    report_path: str,
    violations_path: str | None = None,
    *,
    rule_id: str | None = None,
    offset: int = 0,
    limit: int | None = 100,
) -> dict:
    """Verify a published audit offline and page through its violations.

    Performs *no* network I/O.  Recomputes the violations SHA-256, record
    count, per-rule counts and the deterministic (row_group, row, rule)
    ordering, and compares all of them against the report before returning
    the requested page.  Raises :class:`AuditPublishError` on any mismatch.
    """
    report_path = os.path.abspath(report_path)
    try:
        with open(report_path, "r", encoding="utf-8") as fh:
            report = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise AuditPublishError(
            f"cannot read audit report {report_path!r}: {exc!r}"
        ) from exc
    if report.get("report_format") != REPORT_FORMAT:
        raise AuditPublishError(
            f"not a quality-audit report: {report_path!r}"
        )
    if report.get("status") != "completed":
        raise AuditPublishError(
            f"report status is {report.get('status')!r}, not a completed audit"
        )
    if offset < 0:
        raise AuditPublishError("offset must be >= 0")
    if limit is not None and limit < 0:
        raise AuditPublishError("limit must be >= 0")
    try:
        vpath = os.path.abspath(violations_path or report["violations"]["path"])
        rule_order = report["rules"]
        per_rule_report = report["violations_per_rule"]
        expected_sha = report["violations"]["sha256"]
        expected_records = report["violations"]["records"]
        expected_total = report["total_violations"]
    except (KeyError, TypeError) as exc:
        raise AuditPublishError(
            f"report {report_path!r} is missing required fields: {exc!r}"
        ) from exc
    rule_index = {rid: i for i, rid in enumerate(rule_order)}

    h = hashlib.sha256()
    records = 0
    per_rule: dict[str, int] = collections.Counter()
    last_triple: tuple[int, int, int] | None = None
    order_stable = True
    page: list[dict] = []
    matched = 0
    try:
        vfh = open(vpath, "rb")
    except OSError as exc:
        raise AuditPublishError(
            f"cannot open violations file {vpath!r}: {exc!r}"
        ) from exc
    with vfh as fh:
        for raw in fh:
            h.update(raw)
            records += 1
            try:
                obj = json.loads(raw.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                raise AuditPublishError(
                    f"violations file {vpath!r} is not valid JSONL: {exc!r}"
                ) from exc
            if list(obj.keys()) != list(RECORD_ORDER):
                order_stable = False
            rid = obj["rule_id"]
            per_rule[rid] += 1
            triple = (obj["row_group"], obj["row"], rule_index.get(rid, -1))
            if last_triple is not None and triple <= last_triple:
                order_stable = False
            last_triple = triple
            if rule_id is None or rid == rule_id:
                if matched >= offset and (limit is None or len(page) < limit):
                    page.append(obj)
                matched += 1
    if h.hexdigest() != expected_sha:
        raise AuditPublishError("violations checksum does not match the report")
    if records != expected_records:
        raise AuditPublishError("violations record count does not match the report")
    if records != expected_total:
        raise AuditPublishError("report total_violations disagrees with the file")
    for rid in rule_order:
        try:
            expected = per_rule_report[rid]["violations"]
        except (KeyError, TypeError) as exc:
            raise AuditPublishError(
                f"report is missing the per-rule count for {rid!r}"
            ) from exc
        if per_rule.get(rid, 0) != expected:
            raise AuditPublishError(
                f"per-rule count for {rid!r} does not match the report"
            )
    if not order_stable:
        raise AuditPublishError("violations are not in deterministic rule/row order")
    return {
        "report": report,
        "violations_path": vpath,
        "verified": True,
        "order_stable": True,
        "page_info": {
            "rule_id": rule_id,
            "offset": offset,
            "limit": limit,
            "matched": matched,
            "returned": len(page),
        },
        "page": page,
    }


def explain_quality(report_path: str, violations_path: str | None = None) -> dict:
    """Offline verification summary of a published audit (no network I/O)."""
    result = read_violations(report_path, violations_path, limit=0)
    report = result["report"]
    return {
        "report": report,
        "violations_path": result["violations_path"],
        "verified": True,
        "order_stable": True,
        "total_violations": report["total_violations"],
        "violations_per_rule": report["violations_per_rule"],
    }


def _silent_unlink(path: str | os.PathLike[str]) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
