"""Arrow-to-JSON value rendering for diff facts.

Comparison values keep full Arrow semantics until the merge step: nulls,
empty containers, dictionary-decoded values and binary are all represented by
real Arrow scalars and compared with Arrow equality.  Only the final JSONL
facts are rendered here, with explicit type markers so nothing ambiguous is
left to a consumer's JSON parser:

* ``null`` is JSON ``null``;
* an *empty* list/struct is ``{"type": "...", "value": []/{}}`` -- structurally
  distinct from null;
* binary is base16 text plus an explicit ``"binary"`` marker;
* date/time/timestamp are ISO-8601 strings with their Arrow type named;
* dictionary columns are decoded before reaching this module, so their JSON
  value is simply the decoded value;
* NaN/Inf (which plain JSON cannot express) are emitted as tagged strings.
"""

from __future__ import annotations

import datetime
import math

import pyarrow as pa

_SENTINEL = object()


def scalar_to_json(scalar: pa.Scalar):
    """Render one Arrow scalar as a JSON-compatible Python object."""
    if scalar is None:
        return None
    if not scalar.is_valid:
        return None
    t = scalar.type
    value = scalar.as_py()
    return _value_to_json(value, t)


def _value_to_json(value, t: pa.DataType):
    if value is None:
        return None
    if pa.types.is_dictionary(t):
        # Decoded dictionaries arrive here already decoded; guard anyway.
        return _value_to_json(value, t.value_type)
    if pa.types.is_binary(t) or pa.types.is_large_binary(t) or pa.types.is_fixed_size_binary(t):
        return {"type": "binary", "value": bytes(value).hex()}
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        return value
    if pa.types.is_boolean(t):
        return bool(value)
    if pa.types.is_integer(t):
        return int(value)
    if pa.types.is_floating(t):
        fv = float(value)
        if math.isnan(fv):
            return {"type": "float", "value": "NaN"}
        if math.isinf(fv):
            return {"type": "float", "value": "Infinity" if fv > 0 else "-Infinity"}
        return fv
    if pa.types.is_decimal(t):
        return {"type": str(t), "value": str(value)}
    if pa.types.is_date(t):
        return {"type": str(t), "value": _iso_date(value)}
    if pa.types.is_timestamp(t):
        return {"type": str(t), "value": _iso_datetime(value)}
    if pa.types.is_time(t):
        return {"type": str(t), "value": str(value)}
    if pa.types.is_list(t) or pa.types.is_large_list(t) or pa.types.is_fixed_size_list(t):
        return {
            "type": str(t),
            "value": [_value_to_json(v, t.value_type) for v in value],
        }
    if pa.types.is_struct(t):
        return {
            "type": str(t),
            "value": {
                t.field(i).name: _value_to_json(
                    value[t.field(i).name], t.field(i).type
                )
                for i in range(t.num_fields)
            },
        }
    if pa.types.is_map(t):
        return {
            "type": str(t),
            "value": [
                {
                    "key": _value_to_json(k, t.key_type),
                    "value": _value_to_json(v, t.item_type),
                }
                for k, v in value
            ],
        }
    if pa.types.is_null(t):
        return None
    # Fallback: keep the type explicit rather than silently coercing.
    return {"type": str(t), "value": str(value)}


def _iso_date(value) -> str:
    if isinstance(value, datetime.date):
        return value.isoformat()
    return str(value)


def _iso_datetime(value) -> str:
    if isinstance(value, datetime.datetime):
        return value.isoformat()
    return str(value)
