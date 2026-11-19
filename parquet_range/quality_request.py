"""Quality-audit requests: pinned object identity + contract + output paths.

A request JSON document looks like::

    {
      "spec_version": 1,
      "object":    {"url": "http://h/f.parquet", "etag": "\"ab…\"", "length": 123},
      "contract":  { ...inline contract... },        // or:
      "contract_path": "contract.json",              // exactly one of the two
      "violations_path": "/out/violations.jsonl",
      "report_path":     "/out/report.json",
      "batch_rows": 4096
    }

The object identity is the same pinned triple the diff feature uses
(canonical URL + strong ETag + entity length); it is proven against the live
service before any data byte is read.  Nothing here performs I/O beyond
reading the request/contract documents themselves.
"""

from __future__ import annotations

import dataclasses
import os

from .diff_manifest import SnapshotManifest
from .exceptions import AuditRequestError, ContractError
from .quality_contract import QualityContract, load_json_strict

SPEC_VERSION = 1
DEFAULT_BATCH_ROWS = 4096
#: Decoded micro-batches are capped to keep the resident footprint bounded.
MIN_BATCH_ROWS = 1
MAX_BATCH_ROWS = 1 << 20


@dataclasses.dataclass(frozen=True, slots=True)
class QualityAuditRequest:
    """Everything needed to run one single-snapshot quality audit."""

    object: SnapshotManifest              # pinned input identity
    contract: QualityContract             # validated quality contract
    violations_path: str                  # final JSONL violations destination
    report_path: str                      # final JSON audit report destination
    batch_rows: int = DEFAULT_BATCH_ROWS  # decode/evaluate micro-batch size
    spec_version: int = SPEC_VERSION

    @classmethod
    def from_dict(cls, raw: object, *, base_dir: str | None = None) -> "QualityAuditRequest":
        if not isinstance(raw, dict):
            raise AuditRequestError("quality-audit request must be a JSON object")
        unknown = sorted(set(raw) - {
            "spec_version", "object", "contract", "contract_path",
            "violations_path", "report_path", "batch_rows",
        })
        if unknown:
            raise AuditRequestError(
                f"quality-audit request: unknown keys {unknown}"
            )
        version = raw.get("spec_version", SPEC_VERSION)
        if version != SPEC_VERSION:
            raise AuditRequestError(
                f"unsupported request spec_version {version!r} "
                f"(expected {SPEC_VERSION})"
            )
        try:
            obj = SnapshotManifest.from_dict(raw.get("object"), side="object")
        except Exception as exc:
            if isinstance(exc, AuditRequestError):
                raise
            raise AuditRequestError(f"object identity is invalid: {exc}") from exc

        has_inline = "contract" in raw
        has_path = "contract_path" in raw
        if has_inline == has_path:
            raise AuditRequestError(
                "exactly one of 'contract' (inline object) and 'contract_path' "
                "must be given"
            )
        try:
            if has_inline:
                contract = QualityContract.from_dict(raw["contract"])
            else:
                contract_path = raw["contract_path"]
                if not isinstance(contract_path, str) or not contract_path:
                    raise AuditRequestError("contract_path must be a non-empty string")
                if base_dir and not os.path.isabs(contract_path):
                    contract_path = os.path.join(base_dir, contract_path)
                contract = QualityContract.from_json_file(contract_path)
        except ContractError:
            raise
        except AuditRequestError:
            raise

        violations_path = raw.get("violations_path")
        report_path = raw.get("report_path")
        for name, value in (("violations_path", violations_path),
                            ("report_path", report_path)):
            if not isinstance(value, str) or not value:
                raise AuditRequestError(f"{name} must be a non-empty string")
        batch_rows = raw.get("batch_rows", DEFAULT_BATCH_ROWS)
        if (not isinstance(batch_rows, int) or isinstance(batch_rows, bool)
                or not (MIN_BATCH_ROWS <= batch_rows <= MAX_BATCH_ROWS)):
            raise AuditRequestError(
                f"batch_rows must be an integer in "
                f"[{MIN_BATCH_ROWS}, {MAX_BATCH_ROWS}], got {batch_rows!r}"
            )
        return cls(
            object=obj,
            contract=contract,
            violations_path=os.path.abspath(violations_path),
            report_path=os.path.abspath(report_path),
            batch_rows=batch_rows,
            spec_version=SPEC_VERSION,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "QualityAuditRequest":
        try:
            raw = load_json_strict(path, what="quality-audit request file")
        except ContractError as exc:
            raise AuditRequestError(str(exc)) from exc
        return cls.from_dict(raw, base_dir=os.path.dirname(os.path.abspath(path)))

    def as_json(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "object": self.object.as_json(),
            "contract": self.contract.raw,
            "violations_path": self.violations_path,
            "report_path": self.report_path,
            "batch_rows": self.batch_rows,
        }
