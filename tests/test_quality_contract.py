"""Contract load-time validation: every bad contract fails before any I/O.

All of these raise ContractError (or its UnsupportedPatternError subclass)
with a precise contract location, without touching the network or reading a
single row of data.
"""

from __future__ import annotations

import json

import pytest

from parquet_range import ContractError, QualityContract
from parquet_range.exceptions import UnsupportedPatternError

FIELDS = {
    "i64": "int64",
    "i64b": "int64",
    "u8": "uint8",
    "u64": "uint64",
    "f64": "float64",
    "s": "string",
    "b": "bool",
    "ts": "timestamp[us]",
    "tsz": "timestamp[us, tz=UTC]",
    "d": "date32",
    "bin": "binary",
    "bin2": "binary",
}


def _contract(rules, fields=None):
    return {"spec_version": 1, "fields": fields or dict(FIELDS), "rules": rules}


def _bad(raw, needle):
    with pytest.raises(ContractError) as exc:
        QualityContract.from_dict(raw)
    assert needle in str(exc.value), f"{needle!r} not in {exc.value}"
    assert exc.value.diagnostics, "diagnostics must carry a contract location"
    assert all("location" in d for d in exc.value.diagnostics)
    return exc.value


def test_valid_contract_loads():
    c = QualityContract.from_dict(_contract([
        {"id": "r1", "kind": "not_null", "field": "i64"},
        {"id": "r2", "kind": "range", "field": "f64", "min": 0, "max": 1},
        {"id": "r3", "kind": "enum", "field": "s", "values": ["a"]},
        {"id": "r4", "kind": "pattern", "field": "s", "pattern": "^a+$"},
        {"id": "r5", "kind": "order", "left": "i64", "right": "i64b"},
    ]))
    assert c.rule_order() == ["r1", "r2", "r3", "r4", "r5"]
    assert len(c.sha256) == 64


def test_unknown_top_level_key_rejected():
    _bad({**_contract([{"id": "x", "kind": "not_null", "field": "i64"}]),
          "surprise": 1}, "unknown top-level keys")


def test_unsupported_spec_version():
    _bad(_contract([{"id": "x", "kind": "not_null", "field": "i64"}],
                   ) | {"spec_version": 99}, "spec_version")


def test_fields_must_be_non_empty_object():
    _bad({"spec_version": 1, "fields": {}, "rules": [
        {"id": "x", "kind": "not_null", "field": "i64"}]}, "fields")
    _bad({"spec_version": 1, "fields": ["i64"], "rules": [
        {"id": "x", "kind": "not_null", "field": "i64"}]}, "fields")


def test_unknown_logical_type():
    _bad(_contract([{"id": "x", "kind": "not_null", "field": "z"}],
                   fields={"z": "varchar"}), "unknown logical type")


def test_timestamp_type_forms():
    c = QualityContract.from_dict(_contract(
        [{"id": "x", "kind": "not_null", "field": "a"}],
        fields={"a": "timestamp[ms]", "b": "timestamp[ns, tz=America/New_York]",
                "c": "timestamp[s]"},
    ))
    import pyarrow as pa
    assert c.fields["a"] == pa.timestamp("ms")
    assert c.fields["b"] == pa.timestamp("ns", tz="America/New_York")
    assert c.fields["c"] == pa.timestamp("s")


def test_rules_must_be_non_empty_list():
    _bad({"spec_version": 1, "fields": dict(FIELDS), "rules": []}, "rules")
    _bad({"spec_version": 1, "fields": dict(FIELDS), "rules": {}}, "rules")


def test_unknown_rule_kind():
    _bad(_contract([{"id": "x", "kind": "checksum", "field": "i64"}]),
         "unknown rule kind")


def test_duplicate_rule_ids_rejected_with_position():
    exc = _bad(_contract([
        {"id": "dup", "kind": "not_null", "field": "i64"},
        {"id": "other", "kind": "not_null", "field": "s"},
        {"id": "dup", "kind": "not_null", "field": "f64"},
    ]), "duplicate rule id")
    assert exc.diagnostics[0]["location"] == "rules[2].id"


def test_rule_referencing_undeclared_field():
    exc = _bad(_contract([{"id": "x", "kind": "not_null", "field": "ghost"}]),
               "undeclared field")
    assert "rules[0].field" in str(exc)


def test_unknown_rule_keys_rejected():
    _bad(_contract([{"id": "x", "kind": "not_null", "field": "i64",
                     "min": 3}]), "unknown rule keys")


def test_range_requires_numeric_or_temporal():
    _bad(_contract([{"id": "x", "kind": "range", "field": "s", "min": 1}]),
         "range rules require")
    _bad(_contract([{"id": "x", "kind": "range", "field": "b", "min": 1}]),
         "range rules require")


def test_range_needs_at_least_one_endpoint():
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64"}]),
         "min")


def test_range_bound_type_checked():
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "min": "5"}]), "expected a JSON integer")
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "min": 1.5}]), "expected a JSON integer")
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "min": True}]), "expected a JSON integer")
    _bad(_contract([{"id": "x", "kind": "range", "field": "f64",
                     "min": "1.5"}]), "expected a JSON number")


def test_range_bound_within_integer_width():
    _bad(_contract([{"id": "x", "kind": "range", "field": "u8",
                     "min": 0, "max": 256}]), "outside the range")
    _bad(_contract([{"id": "x", "kind": "range", "field": "u8",
                     "min": -1}]), "outside the range")
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "max": 2 ** 63}]), "outside the range")
    # Exactly at the width boundary is legal.
    QualityContract.from_dict(_contract([
        {"id": "x", "kind": "range", "field": "u64",
         "min": 0, "max": 2 ** 64 - 1}]))
    QualityContract.from_dict(_contract([
        {"id": "x", "kind": "range", "field": "u8", "min": 0, "max": 255}]))


def test_range_min_not_above_max():
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "min": 10, "max": 3}]), "exceeds max")


def test_range_endpoint_inclusivity_flags_must_be_bool():
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "min": 1, "min_inclusive": "yes"}]), "expected a boolean")


def test_timestamp_bound_tz_must_match_field():
    # naive field, aware bound
    _bad(_contract([{"id": "x", "kind": "range", "field": "ts",
                     "min": "2024-01-01T00:00:00+00:00"}]), "timezone-naive")
    # aware field, naive bound
    _bad(_contract([{"id": "x", "kind": "range", "field": "tsz",
                     "min": "2024-01-01T00:00:00"}]), "timezone-aware")
    # matching awareness loads fine
    QualityContract.from_dict(_contract([
        {"id": "x", "kind": "range", "field": "tsz",
         "min": "2024-01-01T00:00:00+00:00",
         "max": "2025-01-01T00:00:00Z"}]))


def test_malformed_timestamp_bound():
    _bad(_contract([{"id": "x", "kind": "range", "field": "ts",
                     "min": "not-a-date"}]), "malformed ISO-8601")
    _bad(_contract([{"id": "x", "kind": "range", "field": "d",
                     "min": "2024-13-40"}]), "malformed ISO-8601")


def test_enum_member_types_checked():
    _bad(_contract([{"id": "x", "kind": "enum", "field": "i64",
                     "values": ["1"]}]), "expected a JSON integer")
    _bad(_contract([{"id": "x", "kind": "enum", "field": "s",
                     "values": [1]}]), "is not a string")
    _bad(_contract([{"id": "x", "kind": "enum", "field": "b",
                     "values": [1]}]), "is not a boolean")
    _bad(_contract([{"id": "x", "kind": "enum", "field": "u8",
                     "values": [256]}]), "outside the range")


def test_enum_requires_non_empty_unique_members():
    _bad(_contract([{"id": "x", "kind": "enum", "field": "s", "values": []}]),
         "non-empty list")
    _bad(_contract([{"id": "x", "kind": "enum", "field": "s",
                     "values": ["a", "a"]}]), "duplicate enum member")


def test_enum_not_supported_for_binary():
    _bad(_contract([{"id": "x", "kind": "enum", "field": "bin",
                     "values": ["a"]}]), "enum rules are not supported")


def test_pattern_requires_string_field():
    _bad(_contract([{"id": "x", "kind": "pattern", "field": "i64",
                     "pattern": "^1$"}]), "pattern rules require a string")


def test_pattern_validated_at_load():
    _bad(_contract([{"id": "x", "kind": "pattern", "field": "s",
                     "pattern": "([unclosed"}]), "unsupported pattern")
    _bad(_contract([{"id": "x", "kind": "pattern", "field": "s",
                     "pattern": "a{5,2}"}]), "min > max")


def test_pattern_unsupported_constructs_named():
    for bad in ("(a|b)\\1", "(?=x)y", "(?<=x)y", "(?P<n>x)", "a*+",
                "\\bword", "(?i)nocase", "a{2001}"):
        exc = _bad(_contract([{"id": "x", "kind": "pattern", "field": "s",
                               "pattern": bad}]), "unsupported pattern")
        assert exc.diagnostics[0]["location"] == "contract.rules[0].pattern"


def test_pattern_error_is_also_unsupported_pattern_error():
    with pytest.raises(UnsupportedPatternError):
        QualityContract.from_dict(_contract([
            {"id": "x", "kind": "pattern", "field": "s", "pattern": "(?=a)b"}
        ]))


def test_order_requires_two_distinct_same_typed_fields():
    _bad(_contract([{"id": "x", "kind": "order",
                     "left": "i64", "right": "i64"}]), "distinct")
    _bad(_contract([{"id": "x", "kind": "order",
                     "left": "i64", "right": "u8"}]), "identical field types")
    _bad(_contract([{"id": "x", "kind": "order",
                     "left": "i64", "right": "ghost"}]), "undeclared field")
    _bad(_contract([{"id": "x", "kind": "order",
                     "left": "bin", "right": "bin2"}]), "order rules require")


def test_nan_flag_requires_float_field():
    _bad(_contract([{"id": "x", "kind": "range", "field": "i64",
                     "min": 0, "nan_is_violation": True}]),
         "nan_is_violation requires a float-typed field")
    _bad(_contract([{"id": "x", "kind": "order", "left": "i64", "right": "i64b",
                     "nan_is_violation": True}]),
         "nan_is_violation requires a float-typed field")


def test_flags_must_be_boolean():
    _bad(_contract([{"id": "x", "kind": "enum", "field": "s",
                     "values": ["a"], "null_is_violation": 1}]),
         "expected a boolean")


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "contract.json"
    path.write_text(
        '{"spec_version": 1, "fields": {"a": "int64", "a": "int32"},'
        ' "rules": [{"id": "x", "kind": "not_null", "field": "a"}]}',
        encoding="utf-8",
    )
    with pytest.raises(ContractError) as exc:
        QualityContract.from_json_file(str(path))
    assert "duplicate key" in str(exc.value)


def test_contract_sha256_is_canonical():
    a = QualityContract.from_dict(_contract(
        [{"id": "x", "kind": "not_null", "field": "i64"}]))
    # Same content, different key order -> same canonical hash.
    shuffled = {
        "rules": [{"field": "i64", "kind": "not_null", "id": "x"}],
        "fields": dict(reversed(list(FIELDS.items()))),
        "spec_version": 1,
    }
    b = QualityContract.from_dict(json.loads(json.dumps(shuffled)))
    assert a.sha256 == b.sha256
