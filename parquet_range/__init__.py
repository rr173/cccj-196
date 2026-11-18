"""Sparse HTTP Range reader for remote Parquet files.

Public surface:
    RemoteParquetClient / ClientConfig / VersionedObject
    scan-equivalent: VersionedObject.scan(columns, row_groups) -> pyarrow.Table
    inspect, cache_clear
"""

from .cache import BlockScheduler, QueryToken
from .client import ClientConfig, RemoteParquetClient, VersionedObject
from .disk_cache import DiskCache
from .exceptions import (
    BadSelectionError,
    BudgetExceeded,
    FooterError,
    ObjectValidationError,
    ParquetRangeError,
    QueryCancelled,
    RangeProtocolError,
    RangeUnsupportedError,
    RetryExhaustedError,
    TruncatedResponseError,
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
]

__version__ = "1.0.0"
