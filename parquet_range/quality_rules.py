"""Row-rule executors: exact, per-value evaluation over record-batch columns.

One executor instance exists per contract rule for the whole audit.  For each
micro-batch the engine first calls :meth:`RuleExecutor.prepare` (which
extracts that batch's column values exactly once, vectorized), then calls the
returned ``check(row_in_batch)`` closure for every row *in contract rule
order* -- so violations stream out in deterministic (row_group, row, rule)
order.  Nothing proportional to the table is retained: executors keep only
small counters between batches.

Semantics are deliberately explicit -- no Python truthiness on Arrow values,
no unsafe signed/unsigned conversions:

* values are extracted with ``to_pylist()`` (nulls become ``None``; temporal
  columns are first cast to their exact stored integer, so timezone-aware
  timestamps compare as instants and ``uint64`` compares as Python ints);
* null is a violation only for ``not_null`` rules or when the rule sets
  ``null_is_violation``; otherwise it is *not applicable* and counted;
* NaN (float fields only) likewise defaults to *not applicable* unless
  ``nan_is_violation`` is set;
* comparisons are ordinary Python ``int``/``float``/``str`` comparisons on
  same-typed values, which are exact for every supported type.
"""

from __future__ import annotations

import math
from typing import Callable

import pyarrow as pa

from .quality_contract import (
    RULE_ENUM,
    RULE_NOT_NULL,
    RULE_ORDER,
    RULE_PATTERN,
    RULE_RANGE,
    Rule,
    temporal_int_type,
    type_kind,
)
from .quality_regex import MAX_SUBJECT_LEN
from .quality_values import typed_value_json

# Stable machine-readable reason codes embedded in violation records.
REASON_NULL = "null"
REASON_NAN = "nan"
REASON_BELOW_MIN = "below_min"
REASON_ABOVE_MAX = "above_max"
REASON_NOT_IN_ENUM = "not_in_enum"
REASON_PATTERN_MISMATCH = "pattern_mismatch"
REASON_PATTERN_TOO_LONG = "pattern_input_too_long"
REASON_ORDER = "order_violated"

# check(row_in_batch) -> (reason, observed) or None
RowCheck = Callable[[int], tuple[str, dict] | None]


def _compare_values(column: pa.Array, ftype: pa.DataType) -> list:
    """Extract one batch of values in the rule's exact comparison domain."""
    if type_kind(ftype) in ("timestamp", "date32", "date64"):
        return column.cast(temporal_int_type(ftype)).to_pylist()
    return column.to_pylist()


class RuleExecutor:
    """Base class: shared counters and the rule's static metadata."""

    def __init__(self, rule: Rule, fields: dict[str, pa.DataType]) -> None:
        self.rule = rule
        self.field_types = {name: fields[name] for name in rule.fields}
        self.null_skipped = 0
        self.nan_skipped = 0

    @property
    def rule_id(self) -> str:
        return self.rule.rule_id

    @property
    def kind(self) -> str:
        return self.rule.kind

    @property
    def fields(self) -> tuple[str, ...]:
        return self.rule.fields

    def prepare(self, columns: dict[str, pa.Array]) -> RowCheck:
        raise NotImplementedError  # pragma: no cover

    def skipped_stats(self) -> dict:
        return {"null_skipped": self.null_skipped,
                "nan_skipped": self.nan_skipped}


class NotNullExecutor(RuleExecutor):
    """Reports nulls; NaN is a value and is *not* reported."""

    def prepare(self, columns: dict[str, pa.Array]) -> RowCheck:
        (field,) = self.fields
        ftype = self.field_types[field]
        values = columns[field].to_pylist()
        observed_null = {field: typed_value_json(ftype, None)}

        def check(i: int):
            if values[i] is None:
                return REASON_NULL, observed_null
            return None

        return check


class RangeExecutor(RuleExecutor):
    def prepare(self, columns: dict[str, pa.Array]) -> RowCheck:
        (field,) = self.fields
        ftype = self.field_types[field]
        kind = type_kind(ftype)
        values = _compare_values(columns[field], ftype)
        rule = self.rule
        minb, maxb = rule.min_bound, rule.max_bound

        def check(i: int):
            value = values[i]
            if value is None:
                if rule.null_is_violation:
                    return REASON_NULL, {field: typed_value_json(ftype, None)}
                self.null_skipped += 1
                return None
            if kind == "float" and math.isnan(value):
                if rule.nan_is_violation:
                    return REASON_NAN, {field: typed_value_json(ftype, value)}
                self.nan_skipped += 1
                return None
            if minb is not None and (
                value < minb or (value == minb and not rule.min_inclusive)
            ):
                return REASON_BELOW_MIN, {field: typed_value_json(ftype, value)}
            if maxb is not None and (
                value > maxb or (value == maxb and not rule.max_inclusive)
            ):
                return REASON_ABOVE_MAX, {field: typed_value_json(ftype, value)}
            return None

        return check


class EnumExecutor(RuleExecutor):
    def prepare(self, columns: dict[str, pa.Array]) -> RowCheck:
        (field,) = self.fields
        ftype = self.field_types[field]
        kind = type_kind(ftype)
        values = _compare_values(columns[field], ftype)
        members = self.rule.members
        assert members is not None

        def check(i: int):
            value = values[i]
            if value is None:
                if self.rule.null_is_violation:
                    return REASON_NULL, {field: typed_value_json(ftype, None)}
                self.null_skipped += 1
                return None
            if kind == "float" and math.isnan(value):
                if self.rule.nan_is_violation:
                    return REASON_NAN, {field: typed_value_json(ftype, value)}
                self.nan_skipped += 1
                return None
            if value not in members:
                return REASON_NOT_IN_ENUM, {field: typed_value_json(ftype, value)}
            return None

        return check


class PatternExecutor(RuleExecutor):
    def prepare(self, columns: dict[str, pa.Array]) -> RowCheck:
        (field,) = self.fields
        ftype = self.field_types[field]
        values = columns[field].to_pylist()
        compiled = self.rule.compiled

        def check(i: int):
            value = values[i]
            if value is None:
                if self.rule.null_is_violation:
                    return REASON_NULL, {field: typed_value_json(ftype, None)}
                self.null_skipped += 1
                return None
            if len(value) > MAX_SUBJECT_LEN:
                # Bounded execution: oversized subjects are flagged with a
                # dedicated reason instead of being matched (or crashing).
                return REASON_PATTERN_TOO_LONG, {
                    field: typed_value_json(ftype, value)
                }
            if not compiled.search(value):
                return REASON_PATTERN_MISMATCH, {
                    field: typed_value_json(ftype, value)
                }
            return None

        return check


class OrderExecutor(RuleExecutor):
    def prepare(self, columns: dict[str, pa.Array]) -> RowCheck:
        left, right = self.fields
        ltype = self.field_types[left]
        kind = type_kind(ltype)
        left_values = _compare_values(columns[left], ltype)
        right_values = _compare_values(columns[right], self.field_types[right])
        rule = self.rule

        def check(i: int):
            lval, rval = left_values[i], right_values[i]
            if lval is None or rval is None:
                if rule.null_is_violation:
                    return REASON_NULL, {
                        left: typed_value_json(ltype, lval),
                        right: typed_value_json(ltype, rval),
                    }
                self.null_skipped += 1
                return None
            if kind == "float" and (math.isnan(lval) or math.isnan(rval)):
                if rule.nan_is_violation:
                    return REASON_NAN, {
                        left: typed_value_json(ltype, lval),
                        right: typed_value_json(ltype, rval),
                    }
                self.nan_skipped += 1
                return None
            if lval < rval or (rule.allow_equal and lval == rval):
                return None
            return REASON_ORDER, {
                left: typed_value_json(ltype, lval),
                right: typed_value_json(ltype, rval),
            }

        return check


_EXECUTORS = {
    RULE_NOT_NULL: NotNullExecutor,
    RULE_RANGE: RangeExecutor,
    RULE_ENUM: EnumExecutor,
    RULE_PATTERN: PatternExecutor,
    RULE_ORDER: OrderExecutor,
}


def build_executor(rule: Rule, fields: dict[str, pa.DataType]) -> RuleExecutor:
    """Instantiate the executor for one validated contract rule."""
    return _EXECUTORS[rule.kind](rule, fields)
