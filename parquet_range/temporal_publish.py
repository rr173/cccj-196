"""Atomic publication of the temporal-enrich wide table.

The enriched Parquet is delivered as *one* complete file:

1. it is written to a staging temp file in the destination directory
   (``.temporal-out-*.parquet.tmp``), flushed and fsynced;
2. the staged file is re-opened with PyArrow and its footer, row count and
   schema are verified against what the run produced -- a corrupt or
   truncated staging file is never swapped in;
3. a single ``os.replace`` swaps it onto the target path, followed by a
   directory fsync.

A read failure, bad input, cancellation, write failure or a process abort
anywhere before the final replace leaves a previously published target
byte-for-byte intact; the staging file is removed on every failure path.
:func:`recover_staging` runs before every new job and deletes leftover
staging files, so an interrupted earlier attempt is never mistaken for a
published result and run temporaries are never trusted as input.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pyarrow.parquet as pq

from .exceptions import TemporalPublishError

_STAGING_PREFIX = ".temporal-out-"
_STAGING_SUFFIX = ".parquet.tmp"


def staging_path_for(out_dir: Path) -> str:
    fd, path = tempfile.mkstemp(
        prefix=_STAGING_PREFIX, suffix=_STAGING_SUFFIX, dir=str(out_dir)
    )
    os.close(fd)
    return path


def is_staging_file(name: str) -> bool:
    return name.startswith(_STAGING_PREFIX) and name.endswith(_STAGING_SUFFIX)


def recover_staging(out_dir: str | os.PathLike[str]) -> dict:
    """Delete leftover staging files in the output directory.

    Returns a small JSON-able summary for the run report's audit trail.
    Never touches anything that is not a staging temp file of this feature.
    """
    out_dir = Path(out_dir)
    summary: dict = {"removed_staging": 0}
    if not out_dir.exists():
        return summary
    for entry in out_dir.iterdir():
        if entry.is_file() and is_staging_file(entry.name):
            _silent_unlink(entry)
            summary["removed_staging"] += 1
    return summary


def verify_staged_parquet(staging_path: str, *, expected_rows: int,
                          expected_schema) -> None:
    """Re-open the staged file and check footer, schema and row count."""
    try:
        pf = pq.ParquetFile(staging_path)
        metadata = pf.metadata
        if metadata.num_rows != expected_rows:
            raise TemporalPublishError(
                f"staged Parquet has {metadata.num_rows} rows, expected "
                f"{expected_rows}; refusing to publish"
            )
        if pf.schema_arrow != expected_schema:
            raise TemporalPublishError(
                "staged Parquet schema disagrees with the run's output schema; "
                "refusing to publish"
            )
    except TemporalPublishError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parse failure is fatal
        raise TemporalPublishError(
            f"staged Parquet {staging_path!r} failed verification: {exc!r}"
        ) from exc


def publish_parquet(*, staging_path: str, output_path: str,
                    expected_rows: int, expected_schema) -> None:
    """Verify the staged file, then atomically replace the target path."""
    output_path = os.path.abspath(output_path)
    out_dir = Path(os.path.dirname(output_path))
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        verify_staged_parquet(
            staging_path, expected_rows=expected_rows,
            expected_schema=expected_schema,
        )
        os.replace(staging_path, output_path)
    except TemporalPublishError:
        _silent_unlink(staging_path)
        raise
    except OSError as exc:
        _silent_unlink(staging_path)
        raise TemporalPublishError(
            f"atomic publication to {output_path!r} failed: {exc!r}"
        ) from exc
    _fsync_dir(out_dir)


def _silent_unlink(path: str | os.PathLike[str]) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass
    except OSError:
        pass


def _fsync_dir(directory: Path) -> None:
    try:
        fd = os.open(str(directory), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass
