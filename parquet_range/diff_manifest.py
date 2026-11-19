"""Snapshot manifests and diff request descriptions.

A diff is driven by two *snapshot manifests*, each pinning one concrete object
version the publisher has previously observed::

    {
      "url":    "http://host/path.parquet",   # canonicalized at load time
      "etag":   "\"sha256-...\"",             # REQUIRED strong ETag
      "length": 1234567                        # REQUIRED entity length
    }

The request additionally names the unique business-key column, the comparison
columns, the two result paths and a memory ceiling that bounds in-memory
records and therefore forces real external-sort spills on large inputs.

Nothing here performs I/O; :class:`~parquet_range.client.RemoteParquetClient`
proves every manifest against the live service before data extraction begins.
"""

from __future__ import annotations

import dataclasses
import json
import os

from .exceptions import ManifestError
from .identity import canonical_url, require_strong_etag

SPEC_VERSION = 1


@dataclasses.dataclass(frozen=True, slots=True)
class SnapshotManifest:
    """One pinned snapshot: canonical URL + expected strong ETag + length."""

    url: str
    etag: str
    length: int

    @classmethod
    def from_dict(cls, raw: object, *, side: str) -> "SnapshotManifest":
        if not isinstance(raw, dict):
            raise ManifestError(f"{side} manifest must be a JSON object")
        missing = [k for k in ("url", "etag", "length") if k not in raw]
        if missing:
            raise ManifestError(f"{side} manifest missing fields: {missing}")
        url = raw["url"]
        etag = raw["etag"]
        length = raw["length"]
        if not isinstance(url, str) or not url.strip():
            raise ManifestError(f"{side} manifest url must be a non-empty string")
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise ManifestError(
                f"{side} manifest length must be a positive integer, got {length!r}"
            )
        try:
            canon = canonical_url(url)
            strong = require_strong_etag(etag)
        except ManifestError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize every malformed input
            raise ManifestError(f"{side} manifest is invalid: {exc!r}") from exc
        return cls(url=canon, etag=strong, length=length)

    def as_json(self) -> dict:
        return {"url": self.url, "etag": self.etag, "length": self.length}


@dataclasses.dataclass(frozen=True, slots=True)
class DiffRequest:
    """Everything needed to run one version diff."""

    left: SnapshotManifest                 # baseline ("before")
    right: SnapshotManifest                # target ("after")
    key_column: str
    compare_columns: tuple[str, ...]
    diff_path: str                         # final JSONL facts destination
    report_path: str                       # final JSON audit report destination
    memory_limit: int                      # max in-memory record bytes per side run
    spec_version: int = SPEC_VERSION

    @classmethod
    def from_dict(cls, raw: object) -> "DiffRequest":
        if not isinstance(raw, dict):
            raise ManifestError("diff request must be a JSON object")
        left = SnapshotManifest.from_dict(raw.get("left"), side="left")
        right = SnapshotManifest.from_dict(raw.get("right"), side="right")
        key_column = raw.get("key_column")
        if not isinstance(key_column, str) or not key_column:
            raise ManifestError("key_column must be a non-empty string")
        cols = raw.get("compare_columns")
        if not isinstance(cols, list) or not cols:
            raise ManifestError("compare_columns must be a non-empty list")
        if not all(isinstance(c, str) and c for c in cols):
            raise ManifestError("compare_columns entries must be non-empty strings")
        if key_column in cols:
            raise ManifestError(
                f"key column {key_column!r} must not also be a compare column"
            )
        if len(set(cols)) != len(cols):
            raise ManifestError("compare_columns contains duplicates")
        diff_path = raw.get("diff_path")
        report_path = raw.get("report_path")
        for name, value in (("diff_path", diff_path), ("report_path", report_path)):
            if not isinstance(value, str) or not value:
                raise ManifestError(f"{name} must be a non-empty string")
        memory_limit = raw.get("memory_limit", 64 << 20)
        if (
            not isinstance(memory_limit, int)
            or isinstance(memory_limit, bool)
            or memory_limit < 512
        ):
            raise ManifestError(
                "memory_limit must be an integer >= 512 bytes"
            )
        return cls(
            left=left,
            right=right,
            key_column=key_column,
            compare_columns=tuple(cols),
            diff_path=os.path.abspath(diff_path),
            report_path=os.path.abspath(report_path),
            memory_limit=memory_limit,
            spec_version=int(raw.get("spec_version", SPEC_VERSION)),
        )

    @classmethod
    def from_json_file(cls, path: str) -> "DiffRequest":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"cannot read request file {path!r}: {exc!r}") from exc
        return cls.from_dict(raw)

    def as_json(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "left": self.left.as_json(),
            "right": self.right.as_json(),
            "key_column": self.key_column,
            "compare_columns": list(self.compare_columns),
            "diff_path": self.diff_path,
            "report_path": self.report_path,
            "memory_limit": self.memory_limit,
        }
