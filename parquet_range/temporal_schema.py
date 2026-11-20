"""Temporal-enrich schema contract: exact-type checks across every shard.

After the footers are read (but *before* a single row of data is fetched)
every shard's top-level schema is checked against the temporal contract:

* required fact columns: ``entity_key`` (UTF-8 string), ``event_at`` and
  ``known_at`` (both ``int64``);
* required revision columns: ``entity_key`` and ``window_id`` (UTF-8
  strings), ``recorded_at`` (``int64``), ``op`` (UTF-8 string),
  ``valid_from`` / ``valid_to`` / ``priority`` (``int64``);
* the join key type is *exactly* the same Arrow type in every shard of both
  roles, and every required column has *exactly* the same type in every
  shard of its role -- nothing is silently cast or reconciled;
* selected business columns (fact carry columns and attached dimension
  columns) exist in every shard of their role with exactly the same type,
  and the type is one of the supported top-level scalar types (UTF-8
  string, integers, boolean, binary, decimal128/256, Arrow timestamp --
  plus date/time and floating point as a convenience superset).

All problems are collected as field-level diagnostics naming the shard and
column, and the job fails with :class:`TemporalSchemaError` before any row
data is read.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from .exceptions import TemporalSchemaError
from .temporal_request import (
    ENTITY_KEY,
    EVENT_AT,
    KNOWN_AT,
    OP,
    PRIORITY,
    RECORDED_AT,
    VALID_FROM,
    VALID_TO,
    WINDOW_ID,
)


def _is_utf8(t: pa.DataType) -> bool:
    return pa.types.is_string(t) or pa.types.is_large_string(t)


def business_type_supported(t: pa.DataType) -> bool:
    """Whether a top-level Arrow type may be a selected business column."""
    if _is_utf8(t) or pa.types.is_integer(t) or pa.types.is_boolean(t):
        return True
    if (
        pa.types.is_binary(t)
        or pa.types.is_large_binary(t)
        or pa.types.is_fixed_size_binary(t)
    ):
        return True
    if pa.types.is_decimal(t) or pa.types.is_timestamp(t):
        return True
    # Convenience superset beyond the required minimum.
    if pa.types.is_date(t) or pa.types.is_time(t) or pa.types.is_floating(t):
        return True
    return False


def carriage_type(t: pa.DataType) -> pa.DataType:
    """Lossless transport type used inside spill records.

    Temporal logical types travel as their raw stored integer so no unit or
    timezone information can be lost by Python datetime round-trips; every
    other supported type travels as itself.
    """
    if pa.types.is_timestamp(t) or pa.types.is_date64(t) or pa.types.is_time64(t):
        return pa.int64()
    if pa.types.is_date32(t) or pa.types.is_time32(t):
        return pa.int32()
    return t


def to_carriage(column: pa.ChunkedArray | pa.Array, t: pa.DataType):
    """Convert a decoded column of declared type ``t`` to carriage form."""
    ct = carriage_type(t)
    if ct == t:
        return column
    return column.cast(ct)


def from_carriage_values(values: list, t: pa.DataType) -> pa.Array:
    """Rebuild a declared-type array from carriage-form Python values."""
    ct = carriage_type(t)
    arr = pa.array(values, type=ct)
    if ct == t:
        return arr
    return arr.cast(t)


@dataclasses.dataclass(frozen=True, slots=True)
class TemporalContract:
    """Resolved, validated schema contract for one temporal-enrich job."""

    key_type: pa.DataType                  # entity_key type (both roles)
    window_id_type: pa.DataType            # window_id type (revision role)
    op_type: pa.DataType                   # op type (revision role)
    fact_fields: tuple[pa.Field, ...]      # selected fact columns (declared types)
    dim_fields: tuple[pa.Field, ...]       # attached dimension columns

    @property
    def fact_column_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.fact_fields)

    @property
    def dim_column_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.dim_fields)


def _check_required(
    diagnostics: list[dict],
    *,
    role: str,
    shard_id: str,
    fields: dict[str, pa.Field],
    name: str,
    predicate,
    expectation: str,
) -> pa.DataType | None:
    field = fields.get(name)
    if field is None:
        diagnostics.append({
            "role": role, "shard_id": shard_id, "column": name,
            "problem": "required column is missing",
            "expected": expectation, "observed": None,
        })
        return None
    if not predicate(field.type):
        diagnostics.append({
            "role": role, "shard_id": shard_id, "column": name,
            "problem": f"required column has wrong type (need {expectation})",
            "expected": expectation, "observed": str(field.type),
        })
        return None
    return field.type


def check_temporal_contract(
    fact_schemas: list[tuple[str, pa.Schema]],
    revision_schemas: list[tuple[str, pa.Schema]],
    fact_columns: tuple[str, ...],
    dim_columns: tuple[str, ...],
) -> TemporalContract:
    """Validate every shard schema; return the resolved contract.

    ``fact_schemas`` / ``revision_schemas`` are ``(shard_id, schema)`` pairs
    in manifest order.  Raises :class:`TemporalSchemaError` with the full
    diagnostic list when anything is off.
    """
    diagnostics: list[dict] = []

    key_types: list[tuple[str, str, pa.DataType]] = []   # (role, shard, type)
    wid_types: list[pa.DataType] = []
    op_types: list[pa.DataType] = []
    fact_col_types: dict[str, pa.DataType] = {}
    dim_col_types: dict[str, pa.DataType] = {}

    # ------------------------------------------------------------ fact role
    for shard_id, schema in fact_schemas:
        fields = {f.name: f for f in schema}
        kt = _check_required(
            diagnostics, role="fact", shard_id=shard_id, fields=fields,
            name=ENTITY_KEY, predicate=_is_utf8,
            expectation="UTF-8 string (string/large_string)",
        )
        if kt is not None:
            key_types.append(("fact", shard_id, kt))
        for ts_col in (EVENT_AT, KNOWN_AT):
            _check_required(
                diagnostics, role="fact", shard_id=shard_id, fields=fields,
                name=ts_col, predicate=pa.types.is_int64,
                expectation="int64 (signed 64-bit, UTC microseconds)",
            )
        for name in fact_columns:
            field = fields.get(name)
            if field is None:
                diagnostics.append({
                    "role": "fact", "shard_id": shard_id, "column": name,
                    "problem": "selected fact column is missing",
                    "expected": "present with a supported scalar type",
                    "observed": None,
                })
                continue
            if not business_type_supported(field.type):
                diagnostics.append({
                    "role": "fact", "shard_id": shard_id, "column": name,
                    "problem": f"unsupported business column type {field.type!s}",
                    "expected": "top-level UTF-8 string / integer / boolean / "
                                "binary / decimal / timestamp (date, time and "
                                "floating point also accepted)",
                    "observed": str(field.type),
                })
                continue
            prev = fact_col_types.setdefault(name, field.type)
            if prev != field.type:
                diagnostics.append({
                    "role": "fact", "shard_id": shard_id, "column": name,
                    "problem": "selected fact column type differs across fact shards",
                    "expected": str(prev), "observed": str(field.type),
                })

    # -------------------------------------------------------- revision role
    for shard_id, schema in revision_schemas:
        fields = {f.name: f for f in schema}
        kt = _check_required(
            diagnostics, role="revision", shard_id=shard_id, fields=fields,
            name=ENTITY_KEY, predicate=_is_utf8,
            expectation="UTF-8 string (string/large_string)",
        )
        if kt is not None:
            key_types.append(("revision", shard_id, kt))
        wt = _check_required(
            diagnostics, role="revision", shard_id=shard_id, fields=fields,
            name=WINDOW_ID, predicate=_is_utf8,
            expectation="UTF-8 string (string/large_string)",
        )
        if wt is not None:
            wid_types.append(wt)
        ot = _check_required(
            diagnostics, role="revision", shard_id=shard_id, fields=fields,
            name=OP, predicate=_is_utf8,
            expectation="UTF-8 string (string/large_string)",
        )
        if ot is not None:
            op_types.append(ot)
        for ts_col in (RECORDED_AT, VALID_FROM, VALID_TO, PRIORITY):
            _check_required(
                diagnostics, role="revision", shard_id=shard_id, fields=fields,
                name=ts_col, predicate=pa.types.is_int64,
                expectation="int64 (signed 64-bit, UTC microseconds)",
            )
        for name in dim_columns:
            field = fields.get(name)
            if field is None:
                diagnostics.append({
                    "role": "revision", "shard_id": shard_id, "column": name,
                    "problem": "selected dimension column is missing",
                    "expected": "present with a supported scalar type",
                    "observed": None,
                })
                continue
            if not business_type_supported(field.type):
                diagnostics.append({
                    "role": "revision", "shard_id": shard_id, "column": name,
                    "problem": f"unsupported dimension column type {field.type!s}",
                    "expected": "top-level UTF-8 string / integer / boolean / "
                                "binary / decimal / timestamp (date, time and "
                                "floating point also accepted)",
                    "observed": str(field.type),
                })
                continue
            prev = dim_col_types.setdefault(name, field.type)
            if prev != field.type:
                diagnostics.append({
                    "role": "revision", "shard_id": shard_id, "column": name,
                    "problem": "dimension column type differs across revision shards",
                    "expected": str(prev), "observed": str(field.type),
                })

    # ------------------------------------------------------- cross-shard unity
    distinct_key_types = {t for _, _, t in key_types}
    if len(distinct_key_types) > 1:
        diagnostics.append({
            "role": "both", "shard_id": None, "column": ENTITY_KEY,
            "problem": "entity_key type differs across shards/roles",
            "expected": "one exact UTF-8 string type everywhere",
            "observed": sorted(str(t) for t in distinct_key_types),
        })
    if len(set(wid_types)) > 1:
        diagnostics.append({
            "role": "revision", "shard_id": None, "column": WINDOW_ID,
            "problem": "window_id type differs across revision shards",
            "expected": "one exact UTF-8 string type",
            "observed": sorted(str(t) for t in set(wid_types)),
        })
    if len(set(op_types)) > 1:
        diagnostics.append({
            "role": "revision", "shard_id": None, "column": OP,
            "problem": "op type differs across revision shards",
            "expected": "one exact UTF-8 string type",
            "observed": sorted(str(t) for t in set(op_types)),
        })

    if diagnostics:
        raise TemporalSchemaError(
            "temporal-enrich schema contract violated; job refused before any "
            "row data was read: "
            + "; ".join(
                f"[{d['role']}/{d.get('shard_id') or '*'}] {d['column']}: "
                f"{d['problem']}"
                for d in diagnostics[:8]
            )
            + (" ..." if len(diagnostics) > 8 else ""),
            diagnostics=diagnostics,
        )

    key_type = key_types[0][2] if key_types else pa.string()
    window_id_type = wid_types[0] if wid_types else pa.string()
    op_type = op_types[0] if op_types else pa.string()
    fact_fields = tuple(
        pa.field(name, fact_col_types[name]) for name in fact_columns
        if name in fact_col_types
    )
    # When a role has no shards at all its selected columns cannot be
    # type-checked; use a placeholder string type so the output schema is
    # well-formed (a degenerate but complete empty wide table where every
    # fact is unmatched and the dimension struct is always null).
    if not revision_schemas and dim_columns:
        dim_col_types = {name: pa.string() for name in dim_columns}
    if not fact_schemas and fact_columns:
        fact_col_types = {name: pa.string() for name in fact_columns}
        fact_fields = tuple(
            pa.field(name, fact_col_types[name]) for name in fact_columns
        )
    dim_fields = tuple(
        pa.field(name, dim_col_types[name]) for name in dim_columns
        if name in dim_col_types
    )
    return TemporalContract(
        key_type=key_type,
        window_id_type=window_id_type,
        op_type=op_type,
        fact_fields=fact_fields,
        dim_fields=dim_fields,
    )
