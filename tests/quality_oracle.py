"""Independent quality-audit oracle: expected violations from first principles.

This module never imports anything from ``parquet_range`` -- it decodes the
Parquet bytes with local PyArrow, parses the contract document itself and
re-implements every rule's semantics row by row (using Python's ``re`` for
patterns).  Tests compare the product's violations JSONL record-for-record
against this ground truth.

Shared *specification* constants (reason codes, the pattern subject cap, the
identity digest formula, the typed-value rendering) are re-derived here
independently; they are part of the feature's documented output contract.
"""

from __future__ import annotations

import datetime
import hashlib
import io
import math
import re

import pyarrow as pa
import pyarrow.parquet as pq

# --- specification constants (documented in docs/quality_design.md) --------
REASON_NULL = "null"
REASON_NAN = "nan"
REASON_BELOW_MIN = "below_min"
REASON_ABOVE_MAX = "above_max"
REASON_NOT_IN_ENUM = "not_in_enum"
REASON_PATTERN_MISMATCH = "pattern_mismatch"
REASON_PATTERN_TOO_LONG = "pattern_input_too_long"
REASON_ORDER = "order_violated"
MAX_SUBJECT_LEN = 65536

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
_EPOCH_DATE = datetime.date(1970, 1, 1)
_UNIT_SCALE = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
_UNIT_DIGITS = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def version_id_of(url: str, etag: str, length: int) -> str:
    digest = hashlib.sha256(f"{url}\n{etag}\n{length}".encode()).digest()
    return digest.hex()[:32]


# ------------------------------------------------------------- type parsing
def parse_type(text: str) -> pa.DataType:
    simple = {
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
    if text in simple:
        return simple[text]
    m = re.fullmatch(r"timestamp\[(s|ms|us|ns)(?:\s*,\s*tz=([^\]]+))?\]", text)
    if m:
        return pa.timestamp(m.group(1), tz=m.group(2)) if m.group(2) \
            else pa.timestamp(m.group(1))
    raise ValueError(f"unknown logical type {text!r}")


def _kind_of(t: pa.DataType) -> str:
    if pa.types.is_boolean(t):
        return "bool"
    if pa.types.is_integer(t):
        return "int"
    if pa.types.is_floating(t):
        return "float"
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return "string"
    if pa.types.is_timestamp(t):
        return "timestamp"
    if pa.types.is_date32(t):
        return "date32"
    if pa.types.is_date64(t):
        return "date64"
    return "other"


def _temporal_int_type(t: pa.DataType) -> pa.DataType:
    return pa.int32() if pa.types.is_date32(t) else pa.int64()


# --------------------------------------------------------- value rendering
def _fmt_epoch_utc(stored: int, unit: str) -> str:
    scale = _UNIT_SCALE[unit]
    secs, rem = divmod(stored, scale)
    try:
        dt = _EPOCH + datetime.timedelta(seconds=secs)
    except OverflowError:
        return f"{stored} [{unit} since 1970-01-01T00:00:00Z]"
    text = dt.strftime("%Y-%m-%dT%H:%M:%S")
    digits = _UNIT_DIGITS[unit]
    if digits:
        frac = f"{rem:0{digits}d}".rstrip("0")
        if frac:
            text += "." + frac
    return text + "+00:00"


def _fmt_date_days(days: int) -> str:
    try:
        return (_EPOCH_DATE + datetime.timedelta(days=days)).isoformat()
    except OverflowError:
        return f"{days} [days since 1970-01-01]"


def _fmt_date_ms(ms: int) -> str:
    days, rem = divmod(ms, 86_400_000)
    base = _fmt_date_days(days)
    return f"{base} (+{rem} ms)" if rem else base


def typed_value(t: pa.DataType, value) -> dict:
    """Independent re-implementation of the typed observed-value rendering."""
    type_text = str(t)
    if value is None:
        return {"type": type_text, "value": None}
    if pa.types.is_timestamp(t):
        return {"type": type_text, "value": _fmt_epoch_utc(int(value), t.unit)}
    if pa.types.is_date32(t):
        return {"type": type_text, "value": _fmt_date_days(int(value))}
    if pa.types.is_date64(t):
        return {"type": type_text, "value": _fmt_date_ms(int(value))}
    if pa.types.is_floating(t):
        fv = float(value)
        if math.isnan(fv):
            return {"type": type_text, "value": "NaN"}
        if math.isinf(fv):
            return {"type": type_text,
                    "value": "Infinity" if fv > 0 else "-Infinity"}
        return {"type": type_text, "value": fv}
    if pa.types.is_integer(t):
        return {"type": type_text, "value": int(value)}
    if pa.types.is_boolean(t):
        return {"type": type_text, "value": bool(value)}
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return {"type": type_text, "value": str(value)}
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        return {"type": type_text, "value": bytes(value).hex()}
    return {"type": type_text, "value": str(value)}


# ------------------------------------------------------- contract semantics
def _bound_to_domain(t: pa.DataType, raw):
    """Convert a JSON bound/member to the comparison domain (ints for temporal)."""
    kind = _kind_of(t)
    if kind == "int":
        return int(raw)
    if kind == "float":
        return float(raw)
    if kind == "bool":
        return bool(raw)
    if kind == "string":
        return str(raw)
    if kind == "timestamp":
        dt = datetime.datetime.fromisoformat(raw)
        return int(pa.array([dt], type=t).cast(pa.int64())[0].as_py())
    if kind in ("date32", "date64"):
        day = datetime.date.fromisoformat(raw)
        return int(pa.array([day], type=t).cast(_temporal_int_type(t))[0].as_py())
    raise ValueError(f"no bound domain for {t}")


def _apply_rule(rule: dict, ftypes: dict, cols: dict, i: int):
    """Return (reason, observed) if row i violates the rule, else None."""
    kind = rule["kind"]
    null_is_v = rule.get("null_is_violation", False)
    nan_is_v = rule.get("nan_is_violation", False)

    def observed(name, value):
        return typed_value(ftypes[name], value)

    if kind == "not_null":
        field = rule["field"]
        if cols[field][i] is None:
            return REASON_NULL, {field: observed(field, None)}
        return None

    if kind in ("range", "enum", "pattern"):
        field = rule["field"]
        t = ftypes[field]
        v = cols[field][i]
        if v is None:
            return (REASON_NULL, {field: observed(field, None)}) if null_is_v else None
        if _kind_of(t) == "float" and math.isnan(v):
            return (REASON_NAN, {field: observed(field, v)}) if nan_is_v else None
        if kind == "range":
            if "min" in rule:
                minb = _bound_to_domain(t, rule["min"])
                if v < minb or (v == minb and not rule.get("min_inclusive", True)):
                    return REASON_BELOW_MIN, {field: observed(field, v)}
            if "max" in rule:
                maxb = _bound_to_domain(t, rule["max"])
                if v > maxb or (v == maxb and not rule.get("max_inclusive", True)):
                    return REASON_ABOVE_MAX, {field: observed(field, v)}
            return None
        if kind == "enum":
            members = {_bound_to_domain(t, raw) for raw in rule["values"]}
            if v not in members:
                return REASON_NOT_IN_ENUM, {field: observed(field, v)}
            return None
        # pattern
        if len(v) > MAX_SUBJECT_LEN:
            return REASON_PATTERN_TOO_LONG, {field: observed(field, v)}
        if re.search(rule["pattern"], v, flags=re.ASCII) is None:
            return REASON_PATTERN_MISMATCH, {field: observed(field, v)}
        return None

    if kind == "order":
        left, right = rule["left"], rule["right"]
        t = ftypes[left]
        lv, rv = cols[left][i], cols[right][i]
        pair = {left: observed(left, lv), right: observed(right, rv)}
        if lv is None or rv is None:
            return (REASON_NULL, pair) if null_is_v else None
        if _kind_of(t) == "float" and (math.isnan(lv) or math.isnan(rv)):
            return (REASON_NAN, pair) if nan_is_v else None
        ok = lv < rv or (rule.get("allow_equal", False) and lv == rv)
        return None if ok else (REASON_ORDER, pair)

    raise ValueError(f"unknown rule kind {kind!r}")


def oracle_expected_violations(data: bytes, contract: dict,
                               identity: dict) -> list[dict]:
    """Compute the full expected violations JSONL content (as dicts)."""
    pf = pq.ParquetFile(io.BytesIO(data))
    fields = contract["fields"]
    ftypes = {name: parse_type(tstr) for name, tstr in fields.items()}
    rules = contract["rules"]
    records: list[dict] = []
    global_row = 0
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=list(fields))
        cols: dict[str, list] = {}
        for name, t in ftypes.items():
            col = table.column(name)
            if _kind_of(t) in ("timestamp", "date32", "date64"):
                col = col.cast(_temporal_int_type(t))
            cols[name] = col.to_pylist()
        for i in range(table.num_rows):
            for rule in rules:
                outcome = _apply_rule(rule, ftypes, cols, i)
                if outcome is None:
                    continue
                reason, observed = outcome
                involved = ([rule["field"]] if "field" in rule
                            else [rule["left"], rule["right"]])
                records.append({
                    "identity": identity,
                    "rule_id": rule["id"],
                    "rule_kind": rule["kind"],
                    "fields": involved,
                    "row_group": rg,
                    "row": global_row + i,
                    "observed": observed,
                    "reason": reason,
                })
        global_row += table.num_rows
    return records
