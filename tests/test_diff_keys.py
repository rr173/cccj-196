"""Canonical key encoding: ordering, collision-freedom, type boundaries.

The encoding is the load-bearing guarantee behind external-sort ordering and
the stable JSONL order, so it is tested directly across every allowed type and
end-to-end with several key types in real Parquet snapshots.
"""

from __future__ import annotations

import hashlib

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_range import DiffEngine, DiffRequest, SnapshotManifest
from parquet_range.diff_key import encode_scalar, key_type_allowed
from parquet_range.exceptions import NullKeyError, UnsupportedKeyTypeError


def test_signed_integer_ordering_and_no_collision():
    vals = [-(2 ** 31), -100, -1, 0, 1, 100, 2 ** 31 - 1]
    enc = [encode_scalar(pa.scalar(v, pa.int32())).encoded for v in vals]
    assert enc == sorted(enc)
    assert len(set(enc)) == len(enc)
    # Sign bit transform: every negative sorts before every non-negative.
    assert enc[0] < encode_scalar(pa.scalar(0, pa.int32())).encoded


def test_unsigned_and_cross_width_ordering():
    widths = [pa.uint8(), pa.uint16(), pa.uint32(), pa.uint64()]
    for t in widths:
        vals = [0, 1, 255] if pa.types.is_uint8(t) else [0, 1, 2 ** 16 - 1]
        enc = [encode_scalar(pa.scalar(v, t)).encoded for v in vals]
        assert enc == sorted(enc)
    # Distinct tags mean int8 value 1 never collides with int64 value 1.
    assert (
        encode_scalar(pa.scalar(1, pa.int8())).encoded
        != encode_scalar(pa.scalar(1, pa.int64())).encoded
    )


def test_length_framing_prevents_boundary_collision():
    # Without an explicit length frame, these could be concatenated to equal
    # bytes; the framing guarantees distinct encodings.
    a = encode_scalar(pa.scalar("a\x00b", pa.string())).encoded
    b = encode_scalar(pa.scalar("a", pa.string())).encoded
    c = encode_scalar(pa.scalar("a\x00bX", pa.string())).encoded
    assert len({a, b, c}) == 3
    # String ordering is bytewise and prefixes sort first.
    assert b < a < c


def test_type_tags_separate_string_binary_bool_int():
    cases = [
        pa.scalar("1", pa.string()),
        pa.scalar(b"1", pa.binary()),
        pa.scalar(True, pa.bool_()),
        pa.scalar(1, pa.int32()),
    ]
    enc = {encode_scalar(s).encoded for s in cases}
    assert len(enc) == len(cases)


def test_timestamp_timezone_and_unit_are_part_of_identity():
    naive = encode_scalar(pa.scalar(1700000000, pa.timestamp("us"))).encoded
    utc = encode_scalar(
        pa.scalar(1700000000, pa.timestamp("us", tz="UTC"))
    ).encoded
    ms = encode_scalar(pa.scalar(1700000000, pa.timestamp("ms"))).encoded
    assert len({naive, utc, ms}) == 3


def test_date_and_decimal_round_trip_types():
    import datetime
    import decimal

    d = encode_scalar(pa.scalar(datetime.date(2024, 1, 2), pa.date32()))
    # date32 canonical value is days since the Unix epoch (Arrow's int storage).
    assert d.json_value == 19724
    assert d.type_name.startswith("date32")
    dec = encode_scalar(pa.scalar(decimal.Decimal("12.34"),
                                  pa.decimal128(10, 2)))
    assert dec.json_value == "12.34"
    dec2 = encode_scalar(pa.scalar(decimal.Decimal("-12.34"),
                                   pa.decimal128(10, 2)))
    assert dec2.encoded < dec.encoded  # negative sorts first


def test_null_and_float_are_rejected():
    with pytest.raises(NullKeyError):
        encode_scalar(pa.scalar(None, pa.int64()))
    with pytest.raises(UnsupportedKeyTypeError):
        encode_scalar(pa.scalar(1.5, pa.float64()))
    assert not key_type_allowed(pa.float32())
    assert not key_type_allowed(pa.list_(pa.int32()))
    assert not key_type_allowed(pa.struct([pa.field("x", pa.int32())]))
    assert key_type_allowed(pa.string())
    assert key_type_allowed(pa.binary())
    assert key_type_allowed(pa.int64())


# ----------------------------------------------------------- end to end types
@pytest.mark.parametrize(
    "arrow_type,left_vals,right_vals",
    [
        (pa.string(), ["a", "b", "c"], ["a", "B", "d"]),
        (pa.binary(), [b"a", b"b", b"c"], [b"a", b"BB", b"d"]),
        (pa.int32(), [10, 20, 30], [10, -20, 40]),
        (pa.uint64(), [1, 2, 3], [1, 22, 4]),
        (pa.bool_(), [True, False], [False, True]),
    ],
)
def test_diff_end_to_end_across_key_types(
    diff_client_factory, diff_server, tmp_path, arrow_type, left_vals, right_vals
):
    schema = pa.schema([
        pa.field("k", arrow_type),
        pa.field("v", pa.string()),
    ])
    lp, rp = str(tmp_path / "kl.parquet"), str(tmp_path / "kr.parquet")
    pq.write_table(pa.table({"k": left_vals, "v": ["x"] * len(left_vals)},
                            schema=schema), lp)
    pq.write_table(pa.table({"k": right_vals, "v": ["y"] * len(right_vals)},
                            schema=schema), rp)
    ld, rd = open(lp, "rb").read(), open(rp, "rb").read()
    le = '"' + hashlib.sha256(ld).hexdigest()[:32] + '"'
    re_ = '"' + hashlib.sha256(rd).hexdigest()[:32] + '"'
    diff_server.put_object("/kl.parquet", ld, le)
    diff_server.put_object("/kr.parquet", rd, re_)
    out = tmp_path / "out"
    out.mkdir(exist_ok=True)
    request = DiffRequest(
        left=SnapshotManifest(diff_server.url_for("/kl.parquet"), le, len(ld)),
        right=SnapshotManifest(diff_server.url_for("/kr.parquet"), re_, len(rd)),
        key_column="k",
        compare_columns=("v",),
        diff_path=str(out / "f.jsonl"),
        report_path=str(out / "r.json"),
        memory_limit=4 << 20,
    )
    report = DiffEngine(diff_client_factory(), request).run()
    # Deterministic canonical order of the facts regardless of input order.
    facts = [__import__("json").loads(line)
             for line in open(request.diff_path)]
    keys = [bytes.fromhex(f["key"]["hex"]) for f in facts]
    assert keys == sorted(keys)
    # Every key present in either side appears exactly once.
    expected_key_count = len(set(map(_hashable, left_vals))
                             | set(map(_hashable, right_vals)))
    assert len(facts) == expected_key_count
    assert report["status"] == "completed"


def _hashable(v):
    return v if not isinstance(v, (bytearray,)) else bytes(v)
