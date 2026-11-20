"""Sparse HTTP Range reader for remote Parquet files.

Public surface:
    RemoteParquetClient / ClientConfig / VersionedObject
    scan-equivalent: VersionedObject.scan(columns, row_groups) -> pyarrow.Table
    inspect, cache_clear
    version diff: DiffEngine / DiffRequest / explain_diff
    quality audit: QualityAuditEngine / QualityAuditRequest / QualityContract
                   / read_violations / explain_quality
    temporal enrich: TemporalEnrichEngine / TemporalEnrichRequest
                     (bitemporal revision resolution + window matching,
                     materialized as a single Parquet wide table)
"""

from .cache import BlockScheduler, QueryToken
from .client import ClientConfig, RemoteParquetClient, VersionedObject
from .disk_cache import DiskCache
from .diff_engine import DiffEngine, run_diff
from .diff_manifest import DiffRequest, SnapshotManifest
from .diff_publish import explain as explain_diff
from .exceptions import (
    AuditCancelled,
    AuditPublishError,
    AuditRequestError,
    AuditSchemaError,
    BadSelectionError,
    BudgetExceeded,
    ContractError,
    DiffCancelled,
    DiffError,
    DiffPublishError,
    DuplicateKeyError,
    FooterError,
    ManifestError,
    NullKeyError,
    ObjectValidationError,
    ParquetRangeError,
    QualityError,
    QueryCancelled,
    RangeProtocolError,
    RangeUnsupportedError,
    RetryExhaustedError,
    SchemaDriftError,
    SegmentIntegrityError,
    TemporalCancelled,
    TemporalDataError,
    TemporalError,
    TemporalPublishError,
    TemporalRequestError,
    TemporalResourceError,
    TemporalSchemaError,
    TruncatedResponseError,
    UnsupportedKeyTypeError,
    UnsupportedPatternError,
    VersionMismatchError,
    WeakOrMissingETagError,
)
from .identity import ObjectIdentity, canonical_url
from .parquet_reader import InspectResult
from .quality_contract import QualityContract
from .quality_engine import QualityAuditEngine, run_quality_audit
from .quality_publish import explain_quality, read_violations
from .quality_request import QualityAuditRequest
from .stats import Stats
from .temporal_engine import TemporalEnrichEngine, run_temporal_enrich
from .temporal_request import TemporalEnrichRequest
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
    # quality-audit feature
    "QualityAuditEngine",
    "run_quality_audit",
    "QualityAuditRequest",
    "QualityContract",
    "read_violations",
    "explain_quality",
    # temporal-enrich feature
    "TemporalEnrichEngine",
    "run_temporal_enrich",
    "TemporalEnrichRequest",
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
    "QualityError",
    "ContractError",
    "UnsupportedPatternError",
    "AuditSchemaError",
    "AuditRequestError",
    "AuditPublishError",
    "AuditCancelled",
    # temporal-enrich errors
    "TemporalError",
    "TemporalRequestError",
    "TemporalSchemaError",
    "TemporalDataError",
    "TemporalResourceError",
    "TemporalPublishError",
    "TemporalCancelled",
]

__version__ = "1.0.0"
