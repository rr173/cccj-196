"""Schema contract checking and field-level drift diagnostics.

Before any data is extracted, the left and right schemas are compared against
the declared key and comparison columns.  A diff is *published only when the
two snapshots are contract-compatible*:

* every declared field exists on both sides;
* the key column is a non-null-capable top-level scalar of an allowed type and
  the type matches on both sides;
* every comparison field has the same Arrow type on both sides (dictionary
  columns must share the same decoded value type; differing index types are
  reported, not silently normalized);
* field *order* differences alone are tolerated -- fields are matched by name,
  never by position, so re-ordered columns cannot be mistaken for drift nor
  hide a real one.

Every problem is appended to a diagnostics list with the offending field and
both observed types; the diff then fails with :class:`SchemaDriftError` and no
result files are written.
"""

from __future__ import annotations

import pyarrow as pa

from .diff_key import key_type_allowed
from .exceptions import SchemaDriftError


def _field_map(schema: pa.Schema) -> dict[str, pa.Field]:
    return {f.name: f for f in schema}


def check_contract(
    left_schema: pa.Schema,
    right_schema: pa.Schema,
    key_column: str,
    compare_columns: tuple[str, ...],
) -> tuple[pa.Field, list[pa.Field]]:
    """Validate the field contract; return (key field, ordered compare fields).

    The returned compare fields use the *left* side's Arrow types; the check
    above guarantees the right side is type-identical for every field.
    """
    diagnostics: list[dict] = []
    lf = _field_map(left_schema)
    rf = _field_map(right_schema)

    # ------------------------------------------------------------ key field
    key_field: pa.Field | None = None
    for side, fields in (("left", lf), ("right", rf)):
        if key_column not in fields:
            diagnostics.append(
                {
                    "field": key_column,
                    "role": "key",
                    "problem": f"missing on {side}",
                    "left_type": str(lf[key_column].type) if key_column in lf else None,
                    "right_type": str(rf[key_column].type) if key_column in rf else None,
                }
            )
    if key_column in lf and key_column in rf:
        lk, rk = lf[key_column], rf[key_column]
        if not key_type_allowed(lk.type):
            diagnostics.append(
                {
                    "field": key_column,
                    "role": "key",
                    "problem": f"unsupported key type {lk.type!s} (must be a "
                               "non-null top-level Arrow scalar; float/nested/list "
                               "types are not allowed)",
                    "left_type": str(lk.type),
                    "right_type": str(rk.type),
                }
            )
        if not key_type_allowed(rk.type):
            diagnostics.append(
                {
                    "field": key_column,
                    "role": "key",
                    "problem": f"unsupported key type {rk.type!s} on right",
                    "left_type": str(lk.type),
                    "right_type": str(rk.type),
                }
            )
        if key_type_allowed(lk.type) and key_type_allowed(rk.type) and lk.type != rk.type:
            diagnostics.append(
                {
                    "field": key_column,
                    "role": "key",
                    "problem": "key type mismatch between snapshots",
                    "left_type": str(lk.type),
                    "right_type": str(rk.type),
                }
            )
        key_field = lk

    # -------------------------------------------------------- compare fields
    compare_fields: list[pa.Field] = []
    for name in compare_columns:
        present_l = name in lf
        present_r = name in rf
        if not present_l or not present_r:
            diagnostics.append(
                {
                    "field": name,
                    "role": "compare",
                    "problem": "missing on "
                    + ("left" if not present_l else "")
                    + (" and " if not present_l and not present_r else "")
                    + ("right" if not present_r else ""),
                    "left_type": str(lf[name].type) if present_l else None,
                    "right_type": str(rf[name].type) if present_r else None,
                }
            )
            continue
        lc, rc = lf[name], rf[name]
        lt, rt = _logical_type(lc.type), _logical_type(rc.type)
        if lt != rt:
            diagnostics.append(
                {
                    "field": name,
                    "role": "compare",
                    "problem": "type mismatch between snapshots",
                    "left_type": str(lc.type),
                    "right_type": str(rc.type),
                }
            )
            continue
        compare_fields.append(lc)

    # Full-schema structural summary is surfaced separately by the engine via
    # schema_summary(); the contract decision above only needs the diagnostics.
    if diagnostics:
        raise SchemaDriftError(
            "schema/type drift detected; diff refused (no silent conversion, "
            "no null-filled fields): "
            + "; ".join(f"{d['field']}: {d['problem']}" for d in diagnostics),
            diagnostics=diagnostics,
        )
    assert key_field is not None  # guaranteed because diagnostics was empty
    return key_field, compare_fields


def _logical_type(t: pa.DataType) -> pa.DataType:
    """Dictionary columns compare on their decoded value type.

    Two dictionary columns with the same value type but different index widths
    are still reported distinctly (the full type string differs in drift rows);
    equality here is on a normalized form so dictionary-vs-plain of the same
    logical value is surfaced as drift rather than accepted.
    """
    if pa.types.is_dictionary(t):
        # Keep dictionary-ness visible: encode as a tuple marker so a plain
        # string never equals a dictionary string.
        return ("dictionary", str(t.index_type), _type_key(t.value_type))
    return _type_key(t)


def _type_key(t: pa.DataType):
    # A stable, order-independent type identity: the full Arrow type string
    # already captures nesting, decimal precision/scale and timestamp units.
    return str(t)


def schema_summary(schema: pa.Schema) -> dict:
    """Compact JSON summary embedded in the audit report."""
    return {
        "fields": [
            {"name": f.name, "type": str(f.type), "nullable": f.nullable}
            for f in schema
        ]
    }
