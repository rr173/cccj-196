"""Temporal-enrich job manifests: pinned shard identities + output contract.

One run of ``temporal-enrich`` is driven by a single JSON job manifest::

    {
      "spec_version": 1,
      "fact_shards": [
        {"shard_id": "f0", "url": "http://h/f0.parquet",
         "etag": "\"ab…\"", "length": 12345}
      ],
      "revision_shards": [
        {"shard_id": "r0", "url": "http://h/r0.parquet",
         "etag": "\"cd…\"", "length": 23456}
      ],
      "fact_columns": ["event_id", "payload"],
      "dim_columns": ["service_area", "owner"],
      "output_path": "/out/wide.parquet",
      "memory_limit": 8388608,
      "spill_dir": "/out/.pqr-temporal-tmp",   // optional
      "batch_rows": 4096                        // optional
    }

Both shard arrays are *ordered*: facts are emitted in manifest shard order
and, inside a shard, in physical row order.  Every entry pins one concrete
object version (canonical URL + strong ETag + entity length) exactly like the
diff snapshot manifests; ``shard_id`` is unique across the whole job and is
the handle every provenance coordinate in the output refers to.

The column names used by the temporal contract are fixed (``entity_key``,
``event_at``, ``known_at``, ``window_id``, ``recorded_at``, ``op``,
``valid_from``, ``valid_to``, ``priority``); the manifest only chooses which
*additional* fact columns are carried into the wide table and which dimension
columns are attached from the winning revision.  Nothing in this module
performs I/O beyond reading the manifest document itself.
"""

from __future__ import annotations

import dataclasses
import json
import os

from .exceptions import TemporalRequestError
from .identity import canonical_url, require_strong_etag

SPEC_VERSION = 1
DEFAULT_BATCH_ROWS = 4096
MIN_BATCH_ROWS = 1
MAX_BATCH_ROWS = 1 << 20
#: Smallest accepted working-memory budget (bytes).
MIN_MEMORY_LIMIT = 4096

# Fixed contract column names (not configurable).
ENTITY_KEY = "entity_key"
EVENT_AT = "event_at"
KNOWN_AT = "known_at"
WINDOW_ID = "window_id"
RECORDED_AT = "recorded_at"
OP = "op"
VALID_FROM = "valid_from"
VALID_TO = "valid_to"
PRIORITY = "priority"
OP_UPSERT = "upsert"
OP_WITHDRAW = "withdraw"

# Reserved output columns; manifests may not occupy these names.
OUT_STATUS = "_match_status"
OUT_DIM = "_dim"
OUT_TOP_COUNT = "_top_window_count"
OUT_FACT_SOURCE = "_fact_source"
OUT_DIM_SOURCES = "_dim_sources"
RESERVED_OUTPUT_COLUMNS = (
    OUT_STATUS,
    OUT_DIM,
    OUT_TOP_COUNT,
    OUT_FACT_SOURCE,
    OUT_DIM_SOURCES,
)


@dataclasses.dataclass(frozen=True, slots=True)
class ShardManifest:
    """One pinned shard: job-unique id + canonical URL + strong ETag + length."""

    shard_id: str
    url: str
    etag: str
    length: int

    @classmethod
    def from_dict(cls, raw: object, *, role: str, index: int) -> "ShardManifest":
        where = f"{role}[{index}]"
        if not isinstance(raw, dict):
            raise TemporalRequestError(f"{where}: shard entry must be a JSON object")
        unknown = sorted(set(raw) - {"shard_id", "url", "etag", "length"})
        if unknown:
            raise TemporalRequestError(f"{where}: unknown keys {unknown}")
        missing = [k for k in ("shard_id", "url", "etag", "length") if k not in raw]
        if missing:
            raise TemporalRequestError(f"{where}: shard entry missing {missing}")
        shard_id = raw["shard_id"]
        if not isinstance(shard_id, str) or not shard_id:
            raise TemporalRequestError(
                f"{where}: shard_id must be a non-empty string"
            )
        url = raw["url"]
        if not isinstance(url, str) or not url.strip():
            raise TemporalRequestError(f"{where}: url must be a non-empty string")
        length = raw["length"]
        if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
            raise TemporalRequestError(
                f"{where}: length must be a positive integer, got {length!r}"
            )
        try:
            canon = canonical_url(url)
            strong = require_strong_etag(raw["etag"])
        except TemporalRequestError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize malformed identity
            raise TemporalRequestError(f"{where}: invalid shard identity: {exc}") from exc
        return cls(shard_id=shard_id, url=canon, etag=strong, length=length)

    def as_json(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "url": self.url,
            "etag": self.etag,
            "length": self.length,
        }


def _parse_columns(raw: object, *, name: str, allow_empty: bool) -> tuple[str, ...]:
    if not isinstance(raw, list) or (not raw and not allow_empty):
        raise TemporalRequestError(
            f"{name} must be a {'possibly empty ' if allow_empty else 'non-empty'} "
            "list of column names"
        )
    if not all(isinstance(c, str) and c for c in raw):
        raise TemporalRequestError(f"{name} entries must be non-empty strings")
    if len(set(raw)) != len(raw):
        raise TemporalRequestError(f"{name} contains duplicates")
    reserved = [c for c in raw if c in RESERVED_OUTPUT_COLUMNS]
    if reserved:
        raise TemporalRequestError(
            f"{name} may not use reserved output column names {reserved} "
            f"(reserved: {list(RESERVED_OUTPUT_COLUMNS)})"
        )
    return tuple(raw)


@dataclasses.dataclass(frozen=True, slots=True)
class TemporalEnrichRequest:
    """Everything needed to run one temporal-enrich job."""

    fact_shards: tuple[ShardManifest, ...]
    revision_shards: tuple[ShardManifest, ...]
    fact_columns: tuple[str, ...]         # fact columns carried into the output
    dim_columns: tuple[str, ...]          # dimension columns attached (>= 1)
    output_path: str                      # final Parquet wide-table destination
    memory_limit: int                     # working-memory budget in bytes
    spill_dir: str | None = None          # controlled temp directory (optional)
    batch_rows: int = DEFAULT_BATCH_ROWS  # decode/validate micro-batch size
    spec_version: int = SPEC_VERSION

    @classmethod
    def from_dict(cls, raw: object) -> "TemporalEnrichRequest":
        if not isinstance(raw, dict):
            raise TemporalRequestError("temporal-enrich request must be a JSON object")
        unknown = sorted(set(raw) - {
            "spec_version", "fact_shards", "revision_shards", "fact_columns",
            "dim_columns", "output_path", "memory_limit", "spill_dir",
            "batch_rows",
        })
        if unknown:
            raise TemporalRequestError(f"temporal-enrich request: unknown keys {unknown}")
        version = raw.get("spec_version", SPEC_VERSION)
        if version != SPEC_VERSION:
            raise TemporalRequestError(
                f"unsupported request spec_version {version!r} "
                f"(expected {SPEC_VERSION})"
            )
        fact_shards_raw = raw.get("fact_shards")
        revision_shards_raw = raw.get("revision_shards")
        if not isinstance(fact_shards_raw, list):
            raise TemporalRequestError("fact_shards must be a (possibly empty) array")
        if not isinstance(revision_shards_raw, list):
            raise TemporalRequestError("revision_shards must be a (possibly empty) array")
        fact_shards = tuple(
            ShardManifest.from_dict(entry, role="fact_shards", index=i)
            for i, entry in enumerate(fact_shards_raw)
        )
        revision_shards = tuple(
            ShardManifest.from_dict(entry, role="revision_shards", index=i)
            for i, entry in enumerate(revision_shards_raw)
        )
        seen: set[str] = set()
        for shard in (*fact_shards, *revision_shards):
            if shard.shard_id in seen:
                raise TemporalRequestError(
                    f"duplicate shard_id {shard.shard_id!r} in job manifest"
                )
            seen.add(shard.shard_id)

        fact_columns = _parse_columns(
            raw.get("fact_columns"), name="fact_columns", allow_empty=True
        )
        dim_columns = _parse_columns(
            raw.get("dim_columns"), name="dim_columns", allow_empty=False
        )

        output_path = raw.get("output_path")
        if not isinstance(output_path, str) or not output_path:
            raise TemporalRequestError("output_path must be a non-empty string")
        memory_limit = raw.get("memory_limit", 64 << 20)
        if (
            not isinstance(memory_limit, int)
            or isinstance(memory_limit, bool)
            or memory_limit < MIN_MEMORY_LIMIT
        ):
            raise TemporalRequestError(
                f"memory_limit must be an integer >= {MIN_MEMORY_LIMIT} bytes"
            )
        spill_dir = raw.get("spill_dir")
        if spill_dir is not None and (not isinstance(spill_dir, str) or not spill_dir):
            raise TemporalRequestError("spill_dir must be a non-empty string when given")
        batch_rows = raw.get("batch_rows", DEFAULT_BATCH_ROWS)
        if (not isinstance(batch_rows, int) or isinstance(batch_rows, bool)
                or not (MIN_BATCH_ROWS <= batch_rows <= MAX_BATCH_ROWS)):
            raise TemporalRequestError(
                f"batch_rows must be an integer in "
                f"[{MIN_BATCH_ROWS}, {MAX_BATCH_ROWS}], got {batch_rows!r}"
            )
        return cls(
            fact_shards=fact_shards,
            revision_shards=revision_shards,
            fact_columns=fact_columns,
            dim_columns=dim_columns,
            output_path=os.path.abspath(output_path),
            memory_limit=memory_limit,
            spill_dir=os.path.abspath(spill_dir) if spill_dir else None,
            batch_rows=batch_rows,
            spec_version=SPEC_VERSION,
        )

    @classmethod
    def from_json_file(cls, path: str) -> "TemporalEnrichRequest":
        try:
            with open(path, "r", encoding="utf-8") as fh:
                raw = json.load(
                    fh, object_pairs_hook=_reject_duplicate_object_pairs
                )
        except TemporalRequestError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise TemporalRequestError(
                f"cannot read temporal-enrich request file {path!r}: {exc!r}"
            ) from exc
        return cls.from_dict(raw)

    def as_json(self) -> dict:
        return {
            "spec_version": self.spec_version,
            "fact_shards": [s.as_json() for s in self.fact_shards],
            "revision_shards": [s.as_json() for s in self.revision_shards],
            "fact_columns": list(self.fact_columns),
            "dim_columns": list(self.dim_columns),
            "output_path": self.output_path,
            "memory_limit": self.memory_limit,
            "spill_dir": self.spill_dir,
            "batch_rows": self.batch_rows,
        }


def _reject_duplicate_object_pairs(pairs: list[tuple[str, object]]) -> dict:
    """JSON object hook refusing silent duplicate-key overrides."""
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise TemporalRequestError(
                f"duplicate key {key!r} in temporal-enrich request JSON"
            )
        out[key] = value
    return out
