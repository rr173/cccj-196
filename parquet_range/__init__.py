"""Sparse HTTP Range reader for remote Parquet files.

Public surface:
    RemoteParquetClient / ClientConfig / VersionedObject
    scan-equivalent: VersionedObject.scan(columns, row_groups) -> pyarrow.Table
    inspect, cache_clear
"""

from .cache import BlockScheduler, QueryToken
from .client import ClientConfig, RemoteParquetClient, VersionedObject
from .disk_cache import DiskCache
from .diff_engine import DiffEngine, run_diff
from .diff_manifest import DiffRequest, SnapshotManifest
from .diff_publish import explain as explain_diff
from .exceptions import (
    BadSelectionError,
    BudgetExceeded,
    DiffCancelled,
    DiffError,
    DiffPublishError,
    DuplicateKeyError,
    FooterError,
    ManifestError,
    NullKeyError,
    ObjectValidationError,
    ParquetRangeError,
    QueryCancelled,
    RangeProtocolError,
    RangeUnsupportedError,
    RetryExhaustedError,
    SchemaDriftError,
    SegmentIntegrityError,
    TruncatedResponseError,
    UnsupportedKeyTypeError,
    VersionMismatchError,
    WeakOrMissingETagError,
)
from .identity import ObjectIdentity, canonical_url
from .parquet_reader import InspectResult
from .stats import Stats
from .transport import HttpTransport

__all__ = [
    "RemoteParquetClient",
    "ClientConfig",
    "VersionedObject",
    "BlockScheduler",
    "QueryToken",
    "DiskCache",
    "HttpTransport",
    "ObjectIdentity",
    "InspectResult",
    "Stats",
    "canonical_url",
    # diff feature
    "DiffEngine",
    "run_diff",
    "DiffRequest",
    "SnapshotManifest",
    "explain_diff",
    "ParquetRangeError",
    "ObjectValidationError",
    "WeakOrMissingETagError",
    "RangeUnsupportedError",
    "VersionMismatchError",
    "RangeProtocolError",
    "TruncatedResponseError",
    "RetryExhaustedError",
    "FooterError",
    "QueryCancelled",
    "BudgetExceeded",
    "BadSelectionError",
    "DiffError",
    "ManifestError",
    "SchemaDriftError",
    "DuplicateKeyError",
    "NullKeyError",
    "UnsupportedKeyTypeError",
    "SegmentIntegrityError",
    "DiffPublishError",
    "DiffCancelled",
]

__version__ = "1.0.0"
