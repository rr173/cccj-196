"""Canonical, sortable, collision-free encoding of business keys.

A business key must be a **non-null top-level Arrow scalar** of one of the
allowed scalar types (string / integers / bool / date / timestamp / binary /
decimal ...).  Floating point is deliberately rejected: NaN and signed-zero
semantics make it unsuitable as an identity.

The encoding is a *self-describing byte string*, not a concatenation of the
raw value and not a bare hash:

* the first byte is a fixed **type tag**, so values of different types can
  never collide and have one deterministic global ordering;
* every variable-length payload is **length-framed** (8-byte big-endian byte
  length followed by exactly that many bytes), so boundary ambiguity between
  adjacent values is impossible -- two distinct keys can never produce the
  same bytes;
* numeric payloads are big-endian with the sign bit flipped (``ordered``
  transform), giving the natural signed numeric ordering under raw byte
  comparison;
* timestamps include their unit and whether they carry a timezone, and their
  timezone name is length-framed into the payload, so two instants stored in
  different zones do not collide;
* null keys are rejected up front; a dedicated tag exists only so the encoder
  can name the violation precisely.

Because the encoding is byte-wise totally ordered, external sort runs can be
merged by comparing these bytes directly, and the published JSONL is stably
ordered by them.
"""

from __future__ import annotations

import dataclasses

import pyarrow as pa

from .exceptions import NullKeyError, UnsupportedKeyTypeError

# Type tags.  The numeric values are part of the on-disk/on-wire format and
# must never be renumbered once released.
TAG_NULL = 0x00
TAG_BOOL = 0x01
TAG_INT8 = 0x10
TAG_INT16 = 0x11
TAG_INT32 = 0x12
TAG_INT64 = 0x13
TAG_UINT8 = 0x20
TAG_UINT16 = 0x21
TAG_UINT32 = 0x22
TAG_UINT64 = 0x23
TAG_DATE32 = 0x30
TAG_DATE64 = 0x31
TAG_TIME32 = 0x32
TAG_TIME64 = 0x33
TAG_TIMESTAMP = 0x34
TAG_STRING = 0x40
TAG_BINARY = 0x41
TAG_LARGE_STRING = 0x42
TAG_LARGE_BINARY = 0x43
TAG_DECIMAL128 = 0x50
TAG_DECIMAL256 = 0x51

# Canonical type names surfaced in JSONL/reports (stable strings).
_TYPE_NAMES: dict[int, str] = {
    TAG_NULL: "null",
    TAG_BOOL: "bool",
    TAG_INT8: "int8",
    TAG_INT16: "int16",
    TAG_INT32: "int32",
    TAG_INT64: "int64",
    TAG_UINT8: "uint8",
    TAG_UINT16: "uint16",
    TAG_UINT32: "uint32",
    TAG_UINT64: "uint64",
    TAG_DATE32: "date32[day]",
    TAG_DATE64: "date64[ms]",
    TAG_TIME32: "time32",
    TAG_TIME64: "time64",
    TAG_TIMESTAMP: "timestamp",
    TAG_STRING: "string",
    TAG_BINARY: "binary",
    TAG_LARGE_STRING: "large_string",
    TAG_LARGE_BINARY: "large_binary",
    TAG_DECIMAL128: "decimal128",
    TAG_DECIMAL256: "decimal256",
}


@dataclasses.dataclass(frozen=True, slots=True)
class CanonicalKey:
    """One encoded key plus enough provenance to render it for humans."""

    encoded: bytes          # total, sortable canonical bytes (tag + payload)
    type_tag: int
    type_name: str
    json_value: object      # JSON-native rendering of the scalar value

    def hex(self) -> str:
        return self.encoded.hex()

    def as_json(self) -> dict:
        return {
            "type": self.type_name,
            "hex": self.hex(),
            "value": self.json_value,
        }


def _frame(payload: bytes) -> bytes:
    """Length-frame a variable-length payload (8-byte BE length + payload)."""
    return len(payload).to_bytes(8, "big") + payload


def _ordered_signed(value: int, width: int) -> bytes:
    """Big-endian signed integer transformed so byte order == numeric order."""
    bits = width * 8
    if value < 0:
        # Two's complement of the negative number, then flip the sign bit.
        raw = value + (1 << bits)
    else:
        raw = value
    raw ^= 1 << (bits - 1)
    return raw.to_bytes(width, "big")


def _ordered_unsigned(value: int, width: int) -> bytes:
    return value.to_bytes(width, "big")


def _check_integer(value: int) -> None:
    if isinstance(value, bool):  # bool is a subclass of int -- keep them apart
        raise UnsupportedKeyTypeError("boolean key must use the bool column type")


def encode_scalar(scalar: pa.Scalar) -> CanonicalKey:
    """Encode a non-null Arrow scalar into its canonical key form."""
    if scalar is None or not scalar.is_valid:
        raise NullKeyError("business key is null; keys must be non-null")

    t = scalar.type
    value = scalar.as_py()

    if pa.types.is_null(t):
        raise NullKeyError("business key column is of null type; no valid key exists")
    if pa.types.is_boolean(t):
        return CanonicalKey(
            bytes([TAG_BOOL, 1 if value else 0]),
            TAG_BOOL,
            _TYPE_NAMES[TAG_BOOL],
            bool(value),
        )
    if pa.types.is_int8(t):
        _check_integer(value)
        return _int_key(TAG_INT8, int(value), 1, int(value))
    if pa.types.is_int16(t):
        _check_integer(value)
        return _int_key(TAG_INT16, int(value), 2, int(value))
    if pa.types.is_int32(t):
        _check_integer(value)
        return _int_key(TAG_INT32, int(value), 4, int(value))
    if pa.types.is_int64(t):
        _check_integer(value)
        return _int_key(TAG_INT64, int(value), 8, int(value))
    if pa.types.is_uint8(t):
        return _int_key(TAG_UINT8, int(value), 1, int(value), unsigned=True)
    if pa.types.is_uint16(t):
        return _int_key(TAG_UINT16, int(value), 2, int(value), unsigned=True)
    if pa.types.is_uint32(t):
        return _int_key(TAG_UINT32, int(value), 4, int(value), unsigned=True)
    if pa.types.is_uint64(t):
        return _int_key(TAG_UINT64, int(value), 8, int(value), unsigned=True)
    if pa.types.is_date(t):
        if pa.types.is_date32(t):
            iv = int(scalar.cast(pa.int32()).as_py())
            return _int_key(TAG_DATE32, iv, 4, iv)
        iv = int(scalar.cast(pa.int64()).as_py())
        return _int_key(TAG_DATE64, iv, 8, iv)
    if pa.types.is_time(t):
        if pa.types.is_time32(t):
            iv = int(scalar.cast(pa.int32()).as_py())
            return _int_key(TAG_TIME32, iv, 4, iv)
        iv = int(scalar.cast(pa.int64()).as_py())
        return _int_key(TAG_TIME64, iv, 8, iv)
    if pa.types.is_timestamp(t):
        return _timestamp_key(scalar, t, value)
    if pa.types.is_string(t) or pa.types.is_large_string(t):
        tag = TAG_LARGE_STRING if pa.types.is_large_string(t) else TAG_STRING
        raw = value.encode("utf-8")
        return CanonicalKey(
            bytes([tag]) + _frame(raw),
            tag,
            _TYPE_NAMES[tag],
            value,
        )
    if pa.types.is_binary(t) or pa.types.is_large_binary(t):
        tag = TAG_LARGE_BINARY if pa.types.is_large_binary(t) else TAG_BINARY
        return CanonicalKey(
            bytes([tag]) + _frame(bytes(value)),
            tag,
            _TYPE_NAMES[tag],
            bytes(value).hex(),
        )
    if pa.types.is_decimal(t):
        return _decimal_key(scalar, t)

    raise UnsupportedKeyTypeError(
        f"key type {t!s} is not an allowed top-level scalar key type "
        "(float/half-float and nested/list/struct types cannot be keys)"
    )


def _int_key(tag: int, value: int, width: int, json_value: int,
             *, unsigned: bool = False) -> CanonicalKey:
    payload = (
        _ordered_unsigned(value, width) if unsigned
        else _ordered_signed(value, width)
    )
    return CanonicalKey(
        bytes([tag]) + payload,
        tag,
        _TYPE_NAMES[tag],
        json_value,
    )


def _timestamp_key(scalar: pa.Scalar, t: pa.TimestampType, value) -> CanonicalKey:
    unit_code = {"s": 1, "ms": 2, "us": 3, "ns": 4}[t.unit]
    tz = t.tz or ""
    # Render as an ISO-8601 string when a timezone is attached (PyArrow gives a
    # tz-aware datetime); naive timestamps keep their unit-scaled integer.
    if tz:
        json_value = value.isoformat()
    else:
        json_value = int(scalar.cast(pa.int64()).as_py())
    tz_bytes = tz.encode("utf-8")
    payload = bytes([unit_code]) + _frame(tz_bytes) + _ordered_signed(
        int(scalar.cast(pa.int64()).as_py()), 8
    )
    type_name = f"timestamp[{t.unit}{', ' + tz if tz else ''}]"
    return CanonicalKey(
        bytes([TAG_TIMESTAMP]) + payload,
        TAG_TIMESTAMP,
        type_name,
        json_value,
    )


def _decimal_key(scalar: pa.Scalar, t: pa.Decimal128Type | pa.Decimal256Type) -> CanonicalKey:
    width = 16 if pa.types.is_decimal128(t) else 32
    tag = TAG_DECIMAL128 if width == 16 else TAG_DECIMAL256
    # Canonical storage form: the unscaled integer (every digit preserved,
    # including the fractional part) in ordered two's-complement bytes.  The
    # precision/scale and sign are encoded explicitly and the payload is
    # length-framed, so distinct values and decimal parameters never collide.
    raw = scalar.as_py()  # decimal.Decimal
    unscaled = int(raw.scaleb(t.scale))  # e.g. Decimal('12.34'), scale 2 -> 1234
    # The ordered transform already encodes sign (negatives sort first), so no
    # separate sign byte is needed; precision/scale plus a length frame make
    # distinct decimal parameters and values collision-free.
    payload_bytes = _ordered_signed(unscaled, width)
    payload = bytes([t.precision, t.scale]) + _frame(payload_bytes)
    type_name = f"decimal({t.precision},{t.scale})"
    return CanonicalKey(
        bytes([tag]) + payload,
        tag,
        type_name,
        str(raw),
    )


def key_type_allowed(t: pa.DataType) -> bool:
    """Return whether a schema field type is legal as a key column."""
    if pa.types.is_null(t) or pa.types.is_floating(t):
        return False
    if (
        pa.types.is_list(t) or pa.types.is_large_list(t)
        or pa.types.is_fixed_size_list(t) or pa.types.is_struct(t)
        or pa.types.is_map(t)
    ):
        return False
    return True
