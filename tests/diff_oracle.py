"""Independent diff oracle: expected facts computed with local PyArrow only.

This module never imports the product diff engine.  It reads the two Parquet
files in full, decodes dictionaries, indexes by key and computes the expected
insert/delete/update/no-change facts and their ordering, providing the
ground-truth that tests compare the product against.
"""

from __future__ import annotations

import io

import pyarrow as pa
import pyarrow.parquet as pq


def scalar_json(value, arrow_type):
    """Independent JSON rendering mirroring the product's tagged output."""
    if value is None:
        return None
    t = str(arrow_type)
    if pa.types.is_binary(arrow_type):
        return {"type": t, "value": bytes(value).hex()}
    if pa.types.is_list(arrow_type) or pa.types.is_large_list(arrow_type):
        return {
            "type": t,
            "value": [scalar_json(v, arrow_type.value_type) for v in value],
        }
    if pa.types.is_struct(arrow_type):
        return {
            "type": t,
            "value": {
                arrow_type.field(i).name: scalar_json(
                    value[arrow_type.field(i).name], arrow_type.field(i).type
                )
                for i in range(arrow_type.num_fields)
            },
        }
    return value


def oracle_expected(left_data: bytes, right_data: bytes, key_column,
                    compare_columns):
    """Return ``[(key, record), ...]`` in the expected canonical key order."""
    lt = pq.read_table(io.BytesIO(left_data))
    rt = pq.read_table(io.BytesIO(right_data))

    def index(table):
        decoded = {}
        for c in table.column_names:
            col = table.column(c)
            decoded[c] = (
                col.cast(col.type.value_type)
                if pa.types.is_dictionary(col.type)
                else col
            )
        out = {}
        for i in range(table.num_rows):
            key = decoded[key_column][i].as_py()
            vals = {
                c: scalar_json(decoded[c][i].as_py(), decoded[c].type)
                for c in compare_columns
            }
            out[key] = vals
        return out

    li, ri = index(lt), index(rt)
    all_keys = sorted(set(li) | set(ri))
    facts = []
    for k in all_keys:
        in_l, in_r = k in li, k in ri
        if in_l and in_r:
            kind = "no-change" if li[k] == ri[k] else "update"
        elif in_r:
            kind = "insert"
        else:
            kind = "delete"
        rec = {"kind": kind, "key": {"type": "int64", "value": k}}
        if in_l:
            rec["before"] = li[k]
        if in_r:
            rec["after"] = ri[k]
        facts.append((k, rec))
    return facts
