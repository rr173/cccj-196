"""Quality-audit contracts: declaration, validation and schema matching.

A *contract* is a JSON document owned by the data team::

    {
      "spec_version": 1,
      "fields": {
        "id":    "int64",
        "price": "float64",
        "ts":    "timestamp[us, tz=UTC]",
        "name":  "string"
      },
      "rules": [
        {"id": "id-non-null",  "kind": "not_null", "field": "id"},
        {"id": "price-range",  "kind": "range", "field": "price",
         "min": 0, "max": 1000000, "min_inclusive": true,
         "max_inclusive": false, "nan_is_violation": true},
        {"id": "ccy-enum",     "kind": "enum", "field": "ccy",
         "values": ["USD", "EUR"]},
        {"id": "name-pattern", "kind": "pattern", "field": "name",
         "pattern": "^[A-Z][a-z]+$"},
        {"id": "window-order", "kind": "order",
         "left": "start_ts", "right": "end_ts", "allow_equal": false}
      ]
    }

Everything that can be known without reading row data is validated at load
time and raises :class:`~parquet_range.exceptions.ContractError` with a
precise contract location *before* a single row is fetched:

* unknown rule kinds, unknown/duplicate keys, duplicate rule ids;
* rules referencing undeclared fields;
* parameter types incompatible with the declared Arrow field type
  (range bounds outside an integer width, string bounds for numeric fields,
  enum members of the wrong type, pattern rules on non-string fields, ...);
* malformed or unsupported string patterns (compiled at load by the bounded
  NFA engine in :mod:`parquet_range.quality_regex`).

After the footer is read, :meth:`QualityContract.match_schema` requires every
declared field to exist in the Parquet file's *top-level* schema with exactly
the declared Arrow type (field order is irrelevant; extra file fields are
ignored).  Nothing is silently converted, skipped or defaulted.

Null/NaN semantics (uniform across rule kinds):

* ``not_null`` reports nulls (NaN is a *value*, not a null);
* every other rule treats null (and, for float fields, NaN) as
  *not applicable* -- skipped, counted, never a violation -- unless the rule
  explicitly sets ``null_is_violation`` / ``nan_is_violation`` to true.
"""

from __future__ import annotations

import dataclasses
import datetime
import hashlib
import json
import math
import re

import pyarrow as pa

from .exceptions import ContractError, UnsupportedPatternError
from .quality_regex import PatternSyntaxError, compile_pattern

SPEC_VERSION = 1

RULE_NOT_NULL = "not_null"
RULE_RANGE = "range"
RULE_ENUM = "enum"
RULE_PATTERN = "pattern"
RULE_ORDER = "order"
RULE_KINDS = (RULE_NOT_NULL, RULE_RANGE, RULE_ENUM, RULE_PATTERN, RULE_ORDER)

_MAX_RULE_ID_LEN = 128

_SIMPLE_TYPES: dict[str, pa.DataType] = {
    "bool": pa.bool_(),
    "int8": pa.int8(), "int16": pa.int16(),
    "int32": pa.int32(), "int64": pa.int64(),
    "uint8": pa.uint8(), "uint16": pa.uint16(),
    "uint32": pa.uint32(), "uint64": pa.uint64(),
    "float32": pa.float32(), "float64": pa.float64(),
    "string": pa.string(), "large_string": pa.large_string(),
    "binary": pa.binary(), "large_binary": pa.large_binary(),
    "date32": pa.date32(), "date64": pa.date64(),
}

_TIMESTAMP_RE = re.compile(r"^timestamp\[(s|ms|us|ns)(?:\s*,\s*tz=([^\]]+))?\]$")


# ------------------------------------------------------------- type parsing
def parse_logical_type(text: object, location: str) -> pa.DataType:
    """Parse a contract logical-type string into an exact Arrow type."""
    if not isinstance(text, str) or not text:
        raise ContractError(
            f"{location}: logical type must be a non-empty string, got {text!r}",
            [{"location": location, "problem": "logical type must be a string"}],
        )
    simple = _SIMPLE_TYPES.get(text)
    if simple is not None:
        return simple
    m = _TIMESTAMP_RE.match(text)
    if m:
        unit, tz = m.group(1), m.group(2)
        if tz is not None:
            tz = tz.strip()
            if not tz:
                raise ContractError(
                    f"{location}: empty timezone in {text!r}",
                    [{"location": location, "problem": "empty timezone name"}],
                )
            return pa.timestamp(unit, tz=tz)
        return pa.timestamp(unit)
    raise ContractError(
        f"{location}: unknown logical type {text!r}; supported: "
        f"{sorted(_SIMPLE_TYPES)} and timestamp[<unit>[, tz=<name>]]",
        [{"location": location, "problem": f"unknown logical type {text!r}"}],
    )


def type_kind(t: pa.DataType) -> str:
    """Coarse classification used for parameter-compatibility checks."""
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return "binary"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_date32(t):
        return "date32"
    if pa.types.is_date64(t):
        return "date64"
    return "other"


def temporal_int_type(t: pa.DataType) -> pa.DataType:
    """Integer type holding the exact stored value of a temporal type."""
    if pa.types.is_date32(t):
        return pa.int32()
    return pa.int64()  # timestamp (any unit) and date64


# ------------------------------------------------------------- rule model
@dataclasses.dataclass(frozen=True, slots=True)
class Rule:
    """One validated row rule; kind-specific attributes are pre-converted."""

    rule_id: str
    kind: str
    fields: tuple[str, ...]
    null_is_violation: bool = False
    nan_is_violation: bool = False
    # range
    min_bound: object = None
    max_bound: object = None
    min_inclusive: bool = True
    max_inclusive: bool = True
    # enum
    members: frozenset | None = None
    # pattern
    compiled: object = None
    # order
    allow_equal: bool = False

    def as_json(self) -> dict:
        out: dict = {"id": self.rule_id, "kind": self.kind,
                     "fields": list(self.fields)}
        out["null_is_violation"] = self.null_is_violation
        out["nan_is_violation"] = self.nan_is_violation
        return out


# ------------------------------------------------------------- JSON loading
def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    obj: dict = {}
    for key, value in pairs:
        if key in obj:
            raise ContractError(f"duplicate key {key!r} in JSON object")
        obj[key] = value
    return obj


def load_json_strict(path: str, *, what: str) -> object:
    """Load JSON rejecting duplicate object keys (a silent-merge hazard)."""
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh, object_pairs_hook=_unique_object)
    except ContractError:
        raise
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"cannot read {what} {path!r}: {exc!r}") from exc


def canonical_sha256(raw: object) -> str:
    """SHA-256 over the canonical serialization (sorted keys, tight separators)."""
    blob = json.dumps(raw, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# ------------------------------------------------------------- value parsing
def _int_width_range(t: pa.DataType) -> tuple[int, int]:
    width = t.bit_width
    if pa.types.is_unsigned_integer(t):
        return 0, (1 << width) - 1
    return -(1 << (width - 1)), (1 << (width - 1)) - 1


def _require_json_int(raw: object, location: str) -> int:
    if isinstance(raw, bool) or not isinstance(raw, int):
        raise ContractError(
            f"{location}: expected a JSON integer, got {raw!r}",
            [{"location": location,
              "problem": f"expected integer, got {type(raw).__name__}"}],
        )
    return raw


def _require_json_number(raw: object, location: str) -> float:
    if isinstance(raw, bool) or not isinstance(raw, (int, float)):
        raise ContractError(
            f"{location}: expected a JSON number, got {raw!r}",
            [{"location": location,
              "problem": f"expected number, got {type(raw).__name__}"}],
        )
    value = float(raw)
    if math.isnan(value):
        raise ContractError(
            f"{location}: NaN is not a valid bound",
            [{"location": location, "problem": "NaN bound"}],
        )
    return value


def _parse_iso_datetime(raw: object, location: str) -> datetime.datetime:
    if not isinstance(raw, str):
        raise ContractError(
            f"{location}: timestamp bound must be an ISO-8601 string, got {raw!r}",
            [{"location": location, "problem": "timestamp bound must be a string"}],
        )
    try:
        dt = datetime.datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ContractError(
            f"{location}: malformed ISO-8601 timestamp {raw!r}: {exc}",
            [{"location": location, "problem": f"malformed ISO-8601: {exc}"}],
        ) from exc
    return dt


def _is_aware(dt: datetime.datetime) -> bool:
    return dt.tzinfo is not None and dt.tzinfo.utcoffset(dt) is not None


def temporal_value_to_int(t: pa.DataType, value: object) -> int:
    """Exact conversion of a datetime/date into the field's stored integer."""
    arr = pa.array([value], type=t)
    return int(arr.cast(temporal_int_type(t))[0].as_py())


def _convert_temporal_bound(t: pa.DataType, raw: object, location: str) -> int:
    kind = type_kind(t)
    if kind == "timestamp":
        dt = _parse_iso_datetime(raw, location)
        field_aware = t.tz is not None
        if field_aware and not _is_aware(dt):
            raise ContractError(
                f"{location}: field {t!s} is timezone-aware but bound {raw!r} "
                "carries no UTC offset",
                [{"location": location,
                  "problem": "naive bound for timezone-aware field"}],
            )
        if not field_aware and _is_aware(dt):
            raise ContractError(
                f"{location}: field {t!s} is timezone-naive but bound {raw!r} "
                "carries a UTC offset",
                [{"location": location,
                  "problem": "timezone-aware bound for naive field"}],
            )
        return temporal_value_to_int(t, dt)
    # date32 / date64
    if not isinstance(raw, str):
        raise ContractError(
            f"{location}: date bound must be an ISO-8601 string, got {raw!r}",
            [{"location": location, "problem": "date bound must be a string"}],
        )
    try:
        day = datetime.date.fromisoformat(raw)
    except ValueError as exc:
        raise ContractError(
            f"{location}: malformed ISO-8601 date {raw!r}: {exc}",
            [{"location": location, "problem": f"malformed ISO-8601 date: {exc}"}],
        ) from exc
    return temporal_value_to_int(t, day)


def convert_bound(t: pa.DataType, raw: object, location: str) -> object:
    """Convert a JSON range bound into the rule's exact comparison domain."""
    kind = type_kind(t)
    if kind == "int":
        value = _require_json_int(raw, location)
        lo, hi = _int_width_range(t)
        if not (lo <= value <= hi):
            raise ContractError(
                f"{location}: bound {value} outside the range of {t!s} "
                f"[{lo}, {hi}]",
                [{"location": location,
                  "problem": f"bound {value} outside [{lo}, {hi}] for {t!s}"}],
            )
        return value
    if kind == "float":
        return _require_json_number(raw, location)
    if kind in ("timestamp", "date32", "date64"):
        return _convert_temporal_bound(t, raw, location)
    raise ContractError(
        f"{location}: range bounds are not supported for field type {t!s}",
        [{"location": location,
          "problem": f"range not supported for type {t!s}"}],
    )


def convert_enum_member(t: pa.DataType, raw: object, location: str) -> object:
    """Convert one enum member into the exact comparison domain."""
    kind = type_kind(t)
    if kind == "string":
        if not isinstance(raw, str):
            raise ContractError(
                f"{location}: enum member {raw!r} is not a string",
                [{"location": location, "problem": "enum member must be a string"}],
            )
        return raw
    if kind == "bool":
        if not isinstance(raw, bool):
            raise ContractError(
                f"{location}: enum member {raw!r} is not a boolean",
                [{"location": location, "problem": "enum member must be a boolean"}],
            )
        return raw
    if kind == "int":
        value = _require_json_int(raw, location)
        lo, hi = _int_width_range(t)
        if not (lo <= value <= hi):
            raise ContractError(
                f"{location}: enum member {value} outside the range of {t!s}",
                [{"location": location,
                  "problem": f"enum member {value} outside [{lo}, {hi}]"}],
            )
        return value
    if kind == "float":
        return _require_json_number(raw, location)
    if kind in ("timestamp", "date32", "date64"):
        return _convert_temporal_bound(t, raw, location)
    raise ContractError(
        f"{location}: enum rules are not supported for field type {t!s}",
        [{"location": location,
          "problem": f"enum not supported for type {t!s}"}],
    )


# ------------------------------------------------------------- rule builders
def _check_no_unknown_keys(rule: dict, allowed: set[str], location: str) -> None:
    unknown = sorted(set(rule) - allowed)
    if unknown:
        raise ContractError(
            f"{location}: unknown rule keys {unknown}",
            [{"location": location, "problem": f"unknown keys {unknown}"}],
        )


def _require_field_ref(fields: dict[str, pa.DataType], name: object,
                       location: str) -> str:
    if not isinstance(name, str) or not name:
        raise ContractError(
            f"{location}: field reference must be a non-empty string, got {name!r}",
            [{"location": location, "problem": "field reference must be a string"}],
        )
    if name not in fields:
        raise ContractError(
            f"{location}: rule references undeclared field {name!r}",
            [{"location": location,
              "problem": f"undeclared field {name!r}"}],
        )
    return name


def _bool_param(rule: dict, key: str, default: bool, location: str) -> bool:
    raw = rule.get(key, default)
    if not isinstance(raw, bool):
        raise ContractError(
            f"{location}.{key}: expected a boolean, got {raw!r}",
            [{"location": f"{location}.{key}", "problem": "expected boolean"}],
        )
    return raw


def _applicability_flags(rule: dict, location: str, *,
                         float_involved: bool) -> tuple[bool, bool]:
    null_flag = _bool_param(rule, "null_is_violation", False, location)
    nan_flag = _bool_param(rule, "nan_is_violation", False, location)
    if nan_flag and not float_involved:
        raise ContractError(
            f"{location}.nan_is_violation: nan_is_violation requires a "
            "float-typed field",
            [{"location": f"{location}.nan_is_violation",
              "problem": "nan_is_violation requires a float-typed field"}],
        )
    return null_flag, nan_flag


def _build_not_null(rule: dict, fields: dict[str, pa.DataType],
                    location: str) -> Rule:
    _check_no_unknown_keys(rule, {"id", "kind", "field"}, location)
    field = _require_field_ref(fields, rule.get("field"), f"{location}.field")
    return Rule(rule_id=rule["id"], kind=RULE_NOT_NULL, fields=(field,))


def _build_range(rule: dict, fields: dict[str, pa.DataType],
                 location: str) -> Rule:
    _check_no_unknown_keys(
        rule,
        {"id", "kind", "field", "min", "max", "min_inclusive", "max_inclusive",
         "null_is_violation", "nan_is_violation"},
        location,
    )
    field = _require_field_ref(fields, rule.get("field"), f"{location}.field")
    ftype = fields[field]
    kind = type_kind(ftype)
    if kind not in ("int", "float", "timestamp", "date32", "date64"):
        raise ContractError(
            f"{location}.field: range rules require a numeric or temporal "
            f"field, but {field!r} is {ftype!s}",
            [{"location": f"{location}.field",
              "problem": f"range not supported for type {ftype!s}"}],
        )
    if "min" not in rule and "max" not in rule:
        raise ContractError(
            f"{location}: a range rule needs at least one of 'min' / 'max'",
            [{"location": location, "problem": "range without min and max"}],
        )
    min_bound = (convert_bound(ftype, rule["min"], f"{location}.min")
                 if "min" in rule else None)
    max_bound = (convert_bound(ftype, rule["max"], f"{location}.max")
                 if "max" in rule else None)
    if min_bound is not None and max_bound is not None and min_bound > max_bound:
        raise ContractError(
            f"{location}: min bound {rule['min']!r} exceeds max bound "
            f"{rule['max']!r}",
            [{"location": location, "problem": "min bound exceeds max bound"}],
        )
    null_flag, nan_flag = _applicability_flags(
        rule, location, float_involved=(kind == "float")
    )
    return Rule(
        rule_id=rule["id"], kind=RULE_RANGE, fields=(field,),
        null_is_violation=null_flag, nan_is_violation=nan_flag,
        min_bound=min_bound, max_bound=max_bound,
        min_inclusive=_bool_param(rule, "min_inclusive", True, location),
        max_inclusive=_bool_param(rule, "max_inclusive", True, location),
    )


def _build_enum(rule: dict, fields: dict[str, pa.DataType],
                location: str) -> Rule:
    _check_no_unknown_keys(
        rule,
        {"id", "kind", "field", "values", "null_is_violation",
         "nan_is_violation"},
        location,
    )
    field = _require_field_ref(fields, rule.get("field"), f"{location}.field")
    ftype = fields[field]
    kind = type_kind(ftype)
    if kind in ("binary", "other"):
        raise ContractError(
            f"{location}.field: enum rules are not supported for type {ftype!s}",
            [{"location": f"{location}.field",
              "problem": f"enum not supported for type {ftype!s}"}],
        )
    raw_values = rule.get("values")
    if not isinstance(raw_values, list) or not raw_values:
        raise ContractError(
            f"{location}.values: enum requires a non-empty list of members",
            [{"location": f"{location}.values",
              "problem": "enum values must be a non-empty list"}],
        )
    members: list = []
    seen: set = set()
    for i, raw in enumerate(raw_values):
        member = convert_enum_member(ftype, raw, f"{location}.values[{i}]")
        if member in seen:
            raise ContractError(
                f"{location}.values[{i}]: duplicate enum member {raw!r}",
                [{"location": f"{location}.values[{i}]",
                  "problem": f"duplicate enum member {raw!r}"}],
            )
        seen.add(member)
        members.append(member)
    null_flag, nan_flag = _applicability_flags(
        rule, location, float_involved=(kind == "float")
    )
    return Rule(
        rule_id=rule["id"], kind=RULE_ENUM, fields=(field,),
        null_is_violation=null_flag, nan_is_violation=nan_flag,
        members=frozenset(members),
    )


def _build_pattern(rule: dict, fields: dict[str, pa.DataType],
                   location: str) -> Rule:
    _check_no_unknown_keys(
        rule, {"id", "kind", "field", "pattern", "null_is_violation"}, location
    )
    field = _require_field_ref(fields, rule.get("field"), f"{location}.field")
    ftype = fields[field]
    if type_kind(ftype) != "string":
        raise ContractError(
            f"{location}.field: pattern rules require a string field, but "
            f"{field!r} is {ftype!s}",
            [{"location": f"{location}.field",
              "problem": f"pattern not supported for type {ftype!s}"}],
        )
    raw_pattern = rule.get("pattern")
    if not isinstance(raw_pattern, str) or not raw_pattern:
        raise ContractError(
            f"{location}.pattern: pattern must be a non-empty string",
            [{"location": f"{location}.pattern",
              "problem": "pattern must be a non-empty string"}],
        )
    try:
        compiled = compile_pattern(raw_pattern)
    except PatternSyntaxError as exc:
        raise UnsupportedPatternError(
            f"{location}.pattern: unsupported pattern {raw_pattern!r}: {exc}",
            [{"location": f"{location}.pattern",
              "problem": f"unsupported pattern at offset {exc.position}: "
                         f"{exc.problem}"}],
        ) from exc
    null_flag, _ = _applicability_flags(rule, location, float_involved=False)
    return Rule(
        rule_id=rule["id"], kind=RULE_PATTERN, fields=(field,),
        null_is_violation=null_flag, compiled=compiled,
    )


def _build_order(rule: dict, fields: dict[str, pa.DataType],
                 location: str) -> Rule:
    _check_no_unknown_keys(
        rule,
        {"id", "kind", "left", "right", "allow_equal", "null_is_violation",
         "nan_is_violation"},
        location,
    )
    left = _require_field_ref(fields, rule.get("left"), f"{location}.left")
    right = _require_field_ref(fields, rule.get("right"), f"{location}.right")
    if left == right:
        raise ContractError(
            f"{location}: order rule needs two distinct fields, got {left!r} "
            "twice",
            [{"location": location, "problem": "left and right are identical"}],
        )
    ltype, rtype = fields[left], fields[right]
    if ltype != rtype:
        raise ContractError(
            f"{location}: order rule requires identical field types, got "
            f"{left}={ltype!s} vs {right}={rtype!s} (no implicit conversion "
            "is ever applied)",
            [{"location": location,
              "problem": f"type mismatch {ltype!s} vs {rtype!s}"}],
        )
    kind = type_kind(ltype)
    if kind not in ("int", "float", "timestamp", "date32", "date64", "string"):
        raise ContractError(
            f"{location}: order rules require numeric, temporal or string "
            f"fields, got {ltype!s}",
            [{"location": location,
              "problem": f"order not supported for type {ltype!s}"}],
        )
    null_flag, nan_flag = _applicability_flags(
        rule, location, float_involved=(kind == "float")
    )
    return Rule(
        rule_id=rule["id"], kind=RULE_ORDER, fields=(left, right),
        null_is_violation=null_flag, nan_is_violation=nan_flag,
        allow_equal=_bool_param(rule, "allow_equal", False, location),
    )


_BUILDERS = {
    RULE_NOT_NULL: _build_not_null,
    RULE_RANGE: _build_range,
    RULE_ENUM: _build_enum,
    RULE_PATTERN: _build_pattern,
    RULE_ORDER: _build_order,
}


# ------------------------------------------------------------- the contract
class QualityContract:
    """A fully validated quality contract; immutable after loading."""

    def __init__(self, raw: dict, fields: dict[str, pa.DataType],
                 rules: list[Rule]) -> None:
        self.raw = raw
        self.fields = fields
        self.rules = rules
        self.spec_version = raw["spec_version"]
        self.sha256 = canonical_sha256(raw)

    # ------------------------------------------------------------ loading
    @classmethod
    def from_dict(cls, raw: object) -> "QualityContract":
        if not isinstance(raw, dict):
            raise ContractError("contract must be a JSON object")
        unknown = sorted(set(raw) - {"spec_version", "fields", "rules"})
        if unknown:
            raise ContractError(
                f"contract: unknown top-level keys {unknown}",
                [{"location": "$", "problem": f"unknown keys {unknown}"}],
            )
        version = raw.get("spec_version")
        if version != SPEC_VERSION:
            raise ContractError(
                f"contract.spec_version: unsupported version {version!r} "
                f"(expected {SPEC_VERSION})",
                [{"location": "spec_version",
                  "problem": f"unsupported spec_version {version!r}"}],
            )
        fields = cls._parse_fields(raw.get("fields"))
        rules = cls._parse_rules(raw.get("rules"), fields)
        return cls(raw, fields, rules)

    @classmethod
    def from_json_file(cls, path: str) -> "QualityContract":
        return cls.from_dict(load_json_strict(path, what="contract file"))

    @staticmethod
    def _parse_fields(raw: object) -> dict[str, pa.DataType]:
        if not isinstance(raw, dict) or not raw:
            raise ContractError(
                "contract.fields: must be a non-empty object mapping field "
                "names to logical types",
                [{"location": "fields", "problem": "fields must be a non-empty object"}],
            )
        fields: dict[str, pa.DataType] = {}
        for name, type_text in raw.items():
            if not isinstance(name, str) or not name:
                raise ContractError(
                    f"contract.fields: field names must be non-empty strings, "
                    f"got {name!r}",
                    [{"location": "fields", "problem": "empty field name"}],
                )
            fields[name] = parse_logical_type(type_text, f"fields.{name}")
        return fields

    @staticmethod
    def _parse_rules(raw: object, fields: dict[str, pa.DataType]) -> list[Rule]:
        if not isinstance(raw, list) or not raw:
            raise ContractError(
                "contract.rules: must be a non-empty list of rule objects",
                [{"location": "rules", "problem": "rules must be a non-empty list"}],
            )
        rules: list[Rule] = []
        seen_ids: dict[str, int] = {}
        for i, item in enumerate(raw):
            location = f"rules[{i}]"
            if not isinstance(item, dict):
                raise ContractError(
                    f"contract.{location}: each rule must be a JSON object",
                    [{"location": location, "problem": "rule must be an object"}],
                )
            rule_id = item.get("id")
            if (not isinstance(rule_id, str) or not rule_id
                    or len(rule_id) > _MAX_RULE_ID_LEN):
                raise ContractError(
                    f"contract.{location}.id: rule id must be a non-empty "
                    f"string of at most {_MAX_RULE_ID_LEN} chars, got {rule_id!r}",
                    [{"location": f"{location}.id",
                      "problem": "missing or oversized rule id"}],
                )
            if rule_id in seen_ids:
                raise ContractError(
                    f"contract.{location}.id: duplicate rule id {rule_id!r} "
                    f"(first used at rules[{seen_ids[rule_id]}])",
                    [{"location": f"{location}.id",
                      "problem": f"duplicate rule id {rule_id!r}"}],
                )
            seen_ids[rule_id] = i
            kind = item.get("kind")
            if kind not in _BUILDERS:
                raise ContractError(
                    f"contract.{location}.kind: unknown rule kind {kind!r}; "
                    f"supported: {list(RULE_KINDS)}",
                    [{"location": f"{location}.kind",
                      "problem": f"unknown rule kind {kind!r}"}],
                )
            rules.append(_BUILDERS[kind](item, fields, f"contract.{location}"))
        return rules

    # ------------------------------------------------------------ queries
    def referenced_fields(self) -> list[str]:
        """Fields any rule reads, in contract declaration order (deterministic)."""
        used: set[str] = set()
        for rule in self.rules:
            used.update(rule.fields)
        return [name for name in self.fields if name in used]

    def rule_order(self) -> list[str]:
        return [rule.rule_id for rule in self.rules]

    def match_schema(self, schema: pa.Schema) -> list[pa.Field]:
        """Match declared fields against the file's top-level schema.

        Every declared field must exist with *exactly* the declared Arrow
        type.  Field order is irrelevant; undeclared file fields (nested or
        not) are simply never projected.  Raises
        :class:`~parquet_range.exceptions.AuditSchemaError` with field-level
        diagnostics before any row data is read.
        """
        from .exceptions import AuditSchemaError

        by_name = {f.name: f for f in schema}
        diagnostics: list[dict] = []
        matched: list[pa.Field] = []
        for name, contract_type in self.fields.items():
            field = by_name.get(name)
            if field is None:
                diagnostics.append({
                    "field": name,
                    "problem": "declared field is missing from the Parquet "
                               "top-level schema",
                    "contract_type": str(contract_type),
                    "parquet_type": None,
                })
                continue
            if field.type != contract_type:
                diagnostics.append({
                    "field": name,
                    "problem": "type mismatch: contract requires an exact "
                               "match (no conversion is ever applied)",
                    "contract_type": str(contract_type),
                    "parquet_type": str(field.type),
                })
                continue
            matched.append(field)
        if diagnostics:
            raise AuditSchemaError(
                "contract does not match the Parquet schema: "
                + "; ".join(f"{d['field']}: {d['problem']}" for d in diagnostics),
                diagnostics=diagnostics,
            )
        return matched

    def schema_summary(self, matched: list[pa.Field]) -> dict:
        return {
            "matched_fields": [
                {"name": f.name, "type": str(f.type)} for f in matched
            ],
            "declared_field_count": len(self.fields),
        }
