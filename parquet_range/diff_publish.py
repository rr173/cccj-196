"""Atomic publication of diff facts and offline report explanation.

Outputs are always written to sibling temporary files in the destination
directory and only renamed into place after the *entire* JSONL is flushed and
fsynced and the report's counts and checksum agree with it.  A failure at any
point removes the temporary files and never touches a previously published
result.

JSONL record field order is fixed by declaration (``kind``, ``key``, then
``before``/``after``) and records are stably ordered by canonical key bytes.
The JSON audit report is a single document carrying both identities, schema
summaries, all four kind counts, comparison-value summaries, real
network/cache/spill statistics, peak resources and the completion state.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import tempfile
from pathlib import Path

from .diff_manifest import SPEC_VERSION
from .exceptions import DiffPublishError

# Fixed record field order (JSON objects preserve insertion order).
RECORD_ORDER = ("kind", "key", "before", "after")
KIND_INSERT = "insert"
KIND_DELETE = "delete"
KIND_UPDATE = "update"
KIND_NOCHANGE = "no-change"


def _atomic_targets(diff_path: str, report_path: str):
    diff_path = os.path.abspath(diff_path)
    report_path = os.path.abspath(report_path)
    if os.path.dirname(diff_path) != os.path.dirname(report_path):
        # os.replace is only guaranteed atomic within one filesystem; require
        # both outputs to share a directory so publication stays atomic-ish
        # and recoverable in one place.
        raise DiffPublishError(
            "diff_path and report_path must live in the same directory for "
            f"atomic publication: {diff_path!r} vs {report_path!r}"
        )
    out_dir = Path(os.path.dirname(diff_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    return diff_path, report_path, out_dir


class DiffFactWriter:
    """Streams JSONL facts to a temp file and tracks counts/checksum."""

    def __init__(self, out_dir: Path, compare_columns: tuple[str, ...]) -> None:
        fd, self.tmp_path = tempfile.mkstemp(
            prefix=".diff-facts-", suffix=".jsonl.tmp", dir=str(out_dir)
        )
        os.close(fd)
        self._fh = open(self.tmp_path, "w", encoding="utf-8", newline="\n")
        self.compare_columns = compare_columns
        self.counts = collections.Counter()
        self.checksum = hashlib.sha256()
        self.records_written = 0
        self._last_key: bytes | None = None

    def write(self, kind: str, key_json: dict, before: dict | None,
              after: dict | None, sort_key: bytes) -> None:
        # Stable ordering invariant: facts must arrive in canonical key order.
        if self._last_key is not None and sort_key < self._last_key:
            raise DiffPublishError(
                "internal: diff facts emitted out of canonical key order"
            )
        self._last_key = sort_key
        record = collections.OrderedDict()
        record["kind"] = kind
        record["key"] = key_json
        if before is not None:
            record["before"] = before
        if after is not None:
            record["after"] = after
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        payload = line + "\n"
        encoded = payload.encode("utf-8")
        self._fh.write(payload)
        self.checksum.update(encoded)
        self.counts[kind] += 1
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
    request_json: dict,
    left_identity: dict,
    right_identity: dict,
    left_schema: dict,
    right_schema: dict,
    key_column: str,
    compare_columns: tuple[str, ...],
    input_rows: dict,
    unique_keys: dict,
    counts: dict,
    value_summary: dict,
    stats: dict,
    peak_resources: dict,
    spill: dict,
    facts_checksum: str,
    facts_records: int,
    facts_bytes: int,
    started_at: str,
    finished_at: str,
) -> dict:
    return collections.OrderedDict([
        ("report_format", "pqr-diff-report"),
        ("spec_version", SPEC_VERSION),
        ("status", "completed"),
        ("started_at_utc", started_at),
        ("finished_at_utc", finished_at),
        ("key_column", key_column),
        ("compare_columns", list(compare_columns)),
        ("left_identity", left_identity),
        ("right_identity", right_identity),
        ("left_schema", left_schema),
        ("right_schema", right_schema),
        ("input_rows", input_rows),
        ("unique_keys", unique_keys),
        ("counts", {
            KIND_INSERT: counts.get(KIND_INSERT, 0),
            KIND_DELETE: counts.get(KIND_DELETE, 0),
            KIND_UPDATE: counts.get(KIND_UPDATE, 0),
            KIND_NOCHANGE: counts.get(KIND_NOCHANGE, 0),
        }),
        ("comparison_value_summary", value_summary),
        ("resources", {
            "network_cache": stats,
            "spill": spill,
            "peak": peak_resources,
        }),
        ("facts", {
            "path": request_json["diff_path"],
            "records": facts_records,
            "bytes": facts_bytes,
            "sha256": facts_checksum,
        }),
    ])


def publish(
    *,
    diff_path: str,
    report_path: str,
    report: dict,
    facts_tmp_path: str,
    expected_records: int,
    expected_checksum: str,
) -> None:
    """Validate then atomically replace both target paths.

    Pre-existing successful outputs are replaced only after both new files are
    fully verified; on any error the targets stay untouched.
    """
    diff_path, report_path, out_dir = _atomic_targets(diff_path, report_path)
    # Re-verify the flushed facts file from disk before touching any target.
    h = hashlib.sha256()
    line_count = 0
    with open(facts_tmp_path, "rb") as fh:
        for line in fh:
            h.update(line)
            line_count += 1
            json.loads(line.decode("utf-8"))  # every line must parse
    if h.hexdigest() != expected_checksum or line_count != expected_records:
        _silent_unlink(facts_tmp_path)
        raise DiffPublishError(
            "facts verification failed before publication: checksum/count mismatch"
        )
    if report["facts"]["sha256"] != expected_checksum:
        raise DiffPublishError("report checksum disagrees with facts")
    if report["facts"]["records"] != expected_records:
        raise DiffPublishError("report record count disagrees with facts")
    counted = report["counts"]
    if sum(counted.values()) != expected_records:
        raise DiffPublishError("report kind counts do not sum to facts records")

    fd, report_tmp = tempfile.mkstemp(
        prefix=".diff-report-", suffix=".json.tmp", dir=str(out_dir)
    )
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
        fh.write("\n")
        fh.flush()
        os.fsync(fh.fileno())

    # Two-step replace: facts first, then report.  If the second replace fails
    # after the first succeeded, roll BOTH targets back so no target pair is
    # ever half-new/half-old or left missing.
    old_facts = diff_path + ".old"
    old_report = report_path + ".old"
    replaced_facts = False
    replaced_report = False
    try:
        if os.path.exists(diff_path):
            os.replace(diff_path, old_facts)
            replaced_facts = True
        os.replace(facts_tmp_path, diff_path)
        try:
            if os.path.exists(report_path):
                os.replace(report_path, old_report)
                replaced_report = True
            os.replace(report_tmp, report_path)
        except OSError:
            # Restore the previous report (if any), then roll facts back too.
            _silent_unlink(report_path)
            if replaced_report and os.path.exists(old_report):
                os.replace(old_report, report_path)
            _silent_unlink(diff_path)
            if replaced_facts and os.path.exists(old_facts):
                os.replace(old_facts, diff_path)
            raise
        _silent_unlink(old_facts)
        _silent_unlink(old_report)
    except OSError as exc:
        _silent_unlink(facts_tmp_path)
        _silent_unlink(report_tmp)
        _silent_unlink(old_facts)
        _silent_unlink(old_report)
        raise DiffPublishError(f"atomic publication failed: {exc!r}") from exc
    _fsync_dir(out_dir)


def explain(report_path: str, diff_path: str | None = None) -> dict:
    """Read a generated report (and facts) offline; never performs network I/O.

    Re-verifies that the facts file on disk still matches the report's record
    count and SHA-256 and returns a JSON-compatible explanation.  Raises if the
    artifacts are missing, incomplete or tampered with.
    """
    report_path = os.path.abspath(report_path)
    with open(report_path, "r", encoding="utf-8") as fh:
        report = json.load(fh)
    if report.get("status") != "completed":
        raise DiffPublishError(
            f"report status is {report.get('status')!r}, not a completed result"
        )
    facts_path = os.path.abspath(diff_path or report["facts"]["path"])
    h = hashlib.sha256()
    records = 0
    kind_counts: dict = {}
    ordered_keys: list[str] = []
    last_key: bytes | None = None
    ordered = True
    with open(facts_path, "rb") as fh:
        for raw in fh:
            h.update(raw)
            records += 1
            obj = json.loads(raw.decode("utf-8"))
            kind = obj["kind"]
            kind_counts[kind] = kind_counts.get(kind, 0) + 1
            key_hex = obj["key"]["hex"]
            ordered_keys.append(key_hex)
            key_bytes = bytes.fromhex(key_hex)
            if last_key is not None and key_bytes < last_key:
                ordered = False
            last_key = key_bytes
            # Fixed field-order contract.
            if list(obj.keys())[0:2] != ["kind", "key"]:
                ordered = False
    if h.hexdigest() != report["facts"]["sha256"]:
        raise DiffPublishError("facts checksum does not match the report")
    if records != report["facts"]["records"]:
        raise DiffPublishError("facts record count does not match the report")
    # Tolerate zero-valued kinds that are present in the report's fixed four-key
    # count object but have no facts lines.
    normalized = {
        kind: kind_counts.get(kind, 0) for kind in report["counts"]
    }
    if normalized != report["counts"]:
        raise DiffPublishError("facts kind counts do not match the report")
    return {
        "report": report,
        "facts_path": facts_path,
        "facts_verified": True,
        "canonical_order_stable": ordered,
        "first_keys": ordered_keys[:5],
        "last_keys": ordered_keys[-5:],
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
