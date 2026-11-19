"""Typed JSON rendering of observed values in quality-violation records.

Every observed value is emitted as ``{"type": <arrow type>, "value": ...}``
so a consumer never has to guess types from JSON tokens:

* null           -> ``{"type": t, "value": null}``
* NaN / +-Inf    -> ``{"type": t, "value": "NaN" / "Infinity" / "-Infinity"}``
* timestamps     -> ISO-8601 of the *instant* rendered in UTC with the full
                    precision of the field's unit (the field's own timezone,
                    if any, stays visible in the ``type`` string); values
                    outside the datetime range fall back to the raw integer
* date32/date64  -> ISO-8601 calendar date
* binary         -> lowercase hex

Temporal renderings are computed from the *stored integer* (the exact value
the rules compared), never from a lossy float or a truncated datetime.
"""

from __future__ import annotations

import datetime
import math

import pyarrow as pa

_EPOCH = datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)
_EPOCH_DATE = datetime.date(1970, 1, 1)
_UNIT_SCALE = {"s": 1, "ms": 1_000, "us": 1_000_000, "ns": 1_000_000_000}
_UNIT_DIGITS = {"s": 0, "ms": 3, "us": 6, "ns": 9}


def format_epoch_utc(stored: int, unit: str) -> str:
    """Render a stored timestamp integer as an exact UTC ISO-8601 string."""
    scale = _UNIT_SCALE[unit]
    secs, rem = divmod(stored, scale)  # floor division: exact for negatives
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


def format_date_days(days: int) -> str:
    """Render a date32 day count as an ISO calendar date."""
    try:
        return (_EPOCH_DATE + datetime.timedelta(days=days)).isoformat()
    except OverflowError:
        return f"{days} [days since 1970-01-01]"


def format_date_ms(ms: int) -> str:
    """Render a date64 millisecond count as an ISO calendar date."""
    days, rem = divmod(ms, 86_400_000)
    base = format_date_days(days)
    if rem:  # date64 values are expected to be whole days; stay exact
        return f"{base} (+{rem} ms)"
    return base


def typed_value_json(arrow_type: pa.DataType, value: object) -> dict:
    """Render one observed value with its Arrow type attached.

    ``value`` is the *comparison-domain* Python object the rule engine used:
    a Python scalar for plain types, or the stored integer for temporal
    types.  ``None`` renders a typed null.
    """
    type_text = str(arrow_type)
    if value is None:
        return {"type": type_text, "value": None}
    if pa.types.is_timestamp(arrow_type):
        return {"type": type_text,
                "value": format_epoch_utc(int(value), arrow_type.unit)}
    if pa.types.is_date32(arrow_type):
        return {"type": type_text, "value": format_date_days(int(value))}
    if pa.types.is_date64(arrow_type):
        return {"type": type_text, "value": format_date_ms(int(value))}
    if pa.types.is_floating(arrow_type):
        fv = float(value)
        if math.isnan(fv):
            return {"type": type_text, "value": "NaN"}
        if math.isinf(fv):
            return {"type": type_text,
                    "value": "Infinity" if fv > 0 else "-Infinity"}
        return {"type": type_text, "value": fv}
    if pa.types.is_integer(arrow_type):
        return {"type": type_text, "value": int(value)}
    if pa.types.is_boolean(arrow_type):
        return {"type": type_text, "value": bool(value)}
    if pa.types.is_string(arrow_type) or pa.types.is_large_string(arrow_type):
        return {"type": type_text, "value": str(value)}
    if (pa.types.is_binary(arrow_type) or pa.types.is_large_binary(arrow_type)
            or pa.types.is_fixed_size_binary(arrow_type)):
        return {"type": type_text, "value": bytes(value).hex()}
    # Defensive fallback: keep the type explicit rather than guessing.
    return {"type": type_text, "value": str(value)}
