"""Parquet metadata planning and on-demand row-group decoding.

Responsibilities of *this* layer (PyArrow only decodes, never does I/O):

1. Locate the Parquet footer ourselves: read the trailing 8 bytes, validate
   ``PAR1`` magic and the footer length, then read the Thrift footer and
   parse it via ``pq.read_metadata`` on an in-memory reconstruction.
2. Map requested top-level columns to their leaf column chunks (nested
   list/struct expansion), using the schema's ``path_in_schema``.
3. Plan exact byte ranges for the selected row groups / leaves (dictionary
   page offset + compressed sizes), prefetch them through the sparse layer,
   then let PyArrow decode with a pre-parsed ``metadata=`` object so its
   construction performs zero remote reads.
"""

from __future__ import annotations

import dataclasses
import io

import pyarrow as pa
import pyarrow.parquet as pq

from .cache import BlockScheduler, QueryToken
from .exceptions import BadSelectionError, FooterError
from .identity import ObjectIdentity
from .sparse_file import SparseObjectFile
from .stats import Stats

PARQUET_MAGIC = b"PAR1"
FOOTER_TAIL_BYTES = 8
# Even an empty Parquet file needs magic(4) + footer_len(4) + magic(4) = 12...
# the smallest plausible footer itself; reject obviously truncated tails.
MIN_FOOTER_ENVELOPE = 12


@dataclasses.dataclass(slots=True)
class InspectResult:
    url: str
    etag: str
    length: int
    version_id: str
    num_rows: int
    num_row_groups: int
    num_columns: int
    schema: pa.Schema
    row_group_rows: list[int]
    row_group_bytes: list[int]
    column_chunk_info: list[dict]

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "etag": self.etag,
            "length": self.length,
            "version_id": self.version_id,
            "num_rows": self.num_rows,
            "num_row_groups": self.num_row_groups,
            "num_columns": self.num_columns,
            "schema": str(self.schema),
            "row_group_rows": self.row_group_rows,
            "row_group_bytes": self.row_group_bytes,
            "column_chunks": self.column_chunk_info,
        }


# ------------------------------------------------------------ footer logic
def read_footer(
    identity: ObjectIdentity,
    scheduler: BlockScheduler,
    stats: Stats,
    *,
    token: QueryToken | None = None,
) -> pq.FileMetaData:
    """Locate, fetch and parse the Parquet footer over sparse Range reads."""
    if identity.length < MIN_FOOTER_ENVELOPE:
        raise FooterError(
            f"object is only {identity.length} bytes, smaller than the minimum "
            f"Parquet footer envelope ({MIN_FOOTER_ENVELOPE})"
        )
    # Step 1: trailing 8 bytes -> footer length + magic.
    tail_start = identity.length - FOOTER_TAIL_BYTES
    tail = scheduler.read(identity, tail_start, FOOTER_TAIL_BYTES, token=token)
    stats.add("range_bytes_requested", FOOTER_TAIL_BYTES)
    if tail[4:] != PARQUET_MAGIC:
        raise FooterError(
            f"bad trailing magic {tail[4:]!r}: not a Parquet file (or truncated)"
        )
    footer_len = int.from_bytes(tail[:4], "little")
    if footer_len <= 0:
        raise FooterError(f"declared footer length {footer_len} is not positive")
    footer_end = tail_start  # footer ends right before the 8-byte envelope
    footer_start = footer_end - footer_len
    # footer + leading 4-byte magic must fit
    if footer_start < 4:
        raise FooterError(
            f"footer length {footer_len} overruns the object "
            f"(footer would start at {footer_start}, before the leading magic)"
        )
    # Step 2: the Thrift footer bytes.
    footer_bytes = scheduler.read(identity, footer_start, footer_len, token=token)
    stats.add("range_bytes_requested", footer_len)
    # Step 3: verify the *leading* magic as well (guards a corrupt envelope).
    leading = scheduler.read(identity, 0, 4, token=token)
    stats.add("range_bytes_requested", 4)
    if leading != PARQUET_MAGIC:
        raise FooterError(f"bad leading magic {leading!r}")
    # Reconstruct a standalone tail blob: footer + len + magic.
    envelope = footer_bytes + footer_len.to_bytes(4, "little") + PARQUET_MAGIC
    try:
        metadata = pq.read_metadata(io.BytesIO(envelope))
    except Exception as exc:  # pq raises various Arrow/Thrift errors
        raise FooterError(f"failed to parse Parquet footer: {exc!r}") from exc
    if metadata.num_rows < 0 or metadata.num_row_groups < 0:
        raise FooterError("parsed footer reports impossible row counts")
    return metadata


# ------------------------------------------------------------ column mapping
def leaf_indices_for(metadata: pq.FileMetaData, columns: list[str]) -> list[int]:
    """Map selected top-level column names to sorted leaf column indices."""
    top_names = metadata.schema.names
    # Build: top-level name -> leaf indices, via path_in_schema of every chunk.
    leaves_of: dict[str, list[int]] = {name: [] for name in top_names}
    for leaf in range(metadata.num_columns):
        path = metadata.row_group(0).column(leaf).path_in_schema
        root = path.split(".", 1)[0]
        if root not in leaves_of:  # schema with zero row groups: fall back below
            for name in top_names:
                if path == name or path.startswith(name + "."):
                    root = name
                    break
        leaves_of.setdefault(root, []).append(leaf)
    selected: list[int] = []
    for col in columns:
        if col not in leaves_of:
            raise BadSelectionError(
                f"unknown column {col!r}; available: {sorted(leaves_of)}"
            )
        selected.extend(leaves_of[col])
    return sorted(set(selected))


def column_chunk_ranges(
    metadata: pq.FileMetaData, row_groups: list[int], leaves: list[int]
) -> list[tuple[int, int]]:
    """Exact [start, end) byte ranges of selected leaves in selected groups.

    A column chunk starts at its dictionary page when one exists, otherwise
    at the first data page.  The end is derived as
    ``min(data_page_offset, dictionary_page_offset) + total_compressed_size``
    because ``total_compressed_size`` covers every page of the chunk
    (dictionary + data) contiguously; we deliberately do not rely on
    ``file_offset`` whose semantics differ across Parquet writers.
    """
    ranges: list[tuple[int, int]] = []
    for rg in row_groups:
        group = metadata.row_group(rg)
        for leaf in leaves:
            chunk = group.column(leaf)
            # A zero-value chunk (row group with zero rows) carries no data
            # pages; its offsets are not meaningful, so there is nothing to
            # prefetch -- PyArrow decodes the empty column from metadata alone.
            if chunk.num_values == 0 and chunk.total_compressed_size == 0:
                continue
            offsets = [chunk.data_page_offset]
            if chunk.dictionary_page_offset is not None:
                offsets.append(chunk.dictionary_page_offset)
            positive = [off for off in offsets if off > 0]
            if not positive:
                if chunk.num_values == 0:
                    # Empty chunk whose only pages are small headers and the
                    # writer reports a zero data offset: nothing readable needed.
                    continue
                raise FooterError(
                    f"row group {rg} leaf {leaf} ({chunk.path_in_schema}) has no "
                    "positive page offset but contains values"
                )
            start = min(positive)
            end = start + chunk.total_compressed_size
            if end <= start:
                raise FooterError(
                    f"row group {rg} leaf {leaf} ({chunk.path_in_schema}) has "
                    f"invalid chunk offsets start={start} end={end}"
                )
            ranges.append((start, end))
    return merge_ranges(ranges)


def merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Merge overlapping/adjacent ranges; never fills holes on its own."""
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [ordered[0]]
    for start, end in ordered[1:]:
        last_start, last_end = merged[-1]
        if start <= last_end:  # adjacent or overlapping
            merged[-1] = (last_start, max(last_end, end))
        else:
            merged.append((start, end))
    return merged


def validate_row_groups(metadata: pq.FileMetaData, row_groups: list[int]) -> list[int]:
    if not row_groups:
        raise BadSelectionError("row_groups must contain at least one index")
    n = metadata.num_row_groups
    for rg in row_groups:
        if not isinstance(rg, int) or rg < 0 or rg >= n:
            raise BadSelectionError(
                f"row group {rg!r} out of range; file has {n} row group(s) [0..{n - 1}]"
            )
    if len(set(row_groups)) != len(row_groups):
        raise BadSelectionError(f"duplicate row group indices in {row_groups}")
    return sorted(row_groups)


# ------------------------------------------------------------ public scan
class ParquetScanClient:
    """Holds transport+scheduler and exposes inspect/scan/cache_clear."""

    def __init__(self, scheduler: BlockScheduler, stats: Stats) -> None:
        self.scheduler = scheduler
        self.stats = stats

    def inspect(
        self, identity: ObjectIdentity, *, token: QueryToken | None = None
    ) -> InspectResult:
        metadata = read_footer(identity, self.scheduler, self.stats, token=token)
        chunk_info: list[dict] = []
        rg_bytes: list[int] = []
        rg_rows: list[int] = []
        for rg in range(metadata.num_row_groups):
            group = metadata.row_group(rg)
            rg_rows.append(group.num_rows)
            rg_bytes.append(group.total_byte_size)
            if rg == 0:
                for leaf in range(metadata.num_columns):
                    c = group.column(leaf)
                    chunk_info.append(
                        {
                            "leaf": leaf,
                            "path": c.path_in_schema,
                            "physical_type": str(c.physical_type),
                            "encodings": [str(e) for e in c.encodings],
                            "dictionary_page_offset": c.dictionary_page_offset,
                            "data_page_offset": c.data_page_offset,
                            "total_compressed_size": c.total_compressed_size,
                        }
                    )
        return InspectResult(
            url=identity.url,
            etag=identity.etag,
            length=identity.length,
            version_id=identity.version_id(),
            num_rows=metadata.num_rows,
            num_row_groups=metadata.num_row_groups,
            num_columns=metadata.num_columns,
            schema=metadata.schema.to_arrow_schema(),
            row_group_rows=rg_rows,
            row_group_bytes=rg_bytes,
            column_chunk_info=chunk_info,
        )

    def scan(
        self,
        identity: ObjectIdentity,
        columns: list[str],
        row_groups: list[int],
        *,
        token: QueryToken | None = None,
    ) -> pa.Table:
        """Return an Arrow table for explicit row groups / top-level columns.

        Empty-column and empty row-group semantics:
          * ``columns=[]`` -> a zero-column table carrying the correct row count
          * a row group with zero rows is legal and returns an empty batch
        """
        metadata = read_footer(identity, self.scheduler, self.stats, token=token)
        groups = validate_row_groups(metadata, row_groups)

        if not columns:
            # Zero-column read: still report exact row counts, no bytes fetched.
            total_rows = sum(metadata.row_group(rg).num_rows for rg in groups)
            return _zero_column_table(total_rows)

        leaves = leaf_indices_for(metadata, columns)
        ranges = column_chunk_ranges(metadata, groups, leaves)

        # Prefetch every planned range through the sparse/cache layer before
        # PyArrow decodes, so coalescing/budgets/cancellation are centralized.
        for start, end in ranges:
            self.stats.add("range_bytes_requested", end - start)
            self.scheduler.read(identity, start, end - start, token=token)

        sparse = SparseObjectFile(identity, self.scheduler, token=token)
        # Pre-parsed metadata => construction performs zero remote I/O.
        pf = pq.ParquetFile(sparse, metadata=metadata)
        tables = [
            pf.read_row_group(rg, columns=columns, use_threads=False)
            for rg in groups
        ]
        if not tables:  # pragma: no cover - validate_row_groups rejects empty
            return _zero_column_table(0)
        result = pa.concat_tables(tables) if len(tables) > 1 else tables[0]
        return result

    def scan_batches(
        self,
        identity: ObjectIdentity,
        columns: list[str],
        row_groups: list[int],
        *,
        batch_size: int,
        token: QueryToken | None = None,
    ):
        """Yield ``(row_group, RecordBatch)`` for explicit groups/columns.

        Unlike :meth:`scan` this never materializes a whole row group in
        memory: each group's column chunks are prefetched through the
        sparse/cache layer right before that group is decoded, and PyArrow
        yields micro-batches of at most ``batch_size`` rows.  Only the
        requested top-level columns' chunks are ever fetched.
        """
        if batch_size <= 0:
            raise BadSelectionError("batch_size must be positive")
        metadata = read_footer(identity, self.scheduler, self.stats, token=token)
        groups = validate_row_groups(metadata, row_groups)
        leaves = leaf_indices_for(metadata, columns)
        sparse = SparseObjectFile(identity, self.scheduler, token=token)
        # Pre-parsed metadata => construction performs zero remote I/O.
        pf = pq.ParquetFile(sparse, metadata=metadata)
        for rg in groups:
            # Prefetch exactly this group's chunks before decoding it.
            ranges = column_chunk_ranges(metadata, [rg], leaves)
            for start, end in ranges:
                self.stats.add("range_bytes_requested", end - start)
                self.scheduler.read(identity, start, end - start, token=token)
            for batch in pf.iter_batches(
                batch_size=batch_size,
                row_groups=[rg],
                columns=columns,
                use_threads=False,
            ):
                yield rg, batch


def _zero_column_table(rows: int) -> pa.Table:
    """A zero-column table that still carries the requested row count."""
    empty_schema = pa.schema([])
    return pa.Table.from_batches(
        [pa.record_batch([], schema=empty_schema) for _ in range(1)],
        schema=empty_schema,
    ).slice(0, rows)
