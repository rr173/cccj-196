"""Exceptions for the sparse remote Parquet reader.

The reader fails loudly on every protocol / version violation.  Nothing here
is ever retried silently against a different object version.
"""

from __future__ import annotations


class ParquetRangeError(Exception):
    """Base class for every error raised by this package."""


class ObjectValidationError(ParquetRangeError):
    """First-contact validation failed (length / strong ETag / Range ability)."""


class WeakOrMissingETagError(ObjectValidationError):
    """The server did not return a strong validator (``W/"..."`` or no ETag)."""


class RangeUnsupportedError(ObjectValidationError):
    """The server answered a ranged probe with something other than 206."""


class VersionMismatchError(ParquetRangeError):
    """A conditional request shows the object changed under a pinned query.

    Raised on HTTP 412 *and* whenever a 206 response carries an ETag that
    differs from the version the query was opened against.  The caller may
    open a fresh query against the new version; cached bytes of the old
    version remain isolated under their own versioned cache key.
    """


class RangeProtocolError(ParquetRangeError):
    """A 206 response violated the Range protocol.

    Covers bad/absent ``Content-Range``, entity length mismatch, missing
    ETag, a 200 whole-object fallback, and a truncated body.
    """


class TruncatedResponseError(RangeProtocolError):
    """Fewer bytes arrived than the validated range required."""


class RetryExhaustedError(ParquetRangeError):
    """A physical request kept failing after the bounded retry budget."""


class FooterError(ParquetRangeError):
    """The Parquet footer is missing, truncated, or has a bad magic number."""


class QueryCancelled(ParquetRangeError):
    """The owning query was cancelled while this read was in flight."""


class BudgetExceeded(ParquetRangeError):
    """An in-flight byte/request quota would be exceeded (non-retryable)."""


class EmptyObjectError(ParquetRangeError):
    """The remote object is zero bytes long, so it cannot be a Parquet file."""


class BadSelectionError(ParquetRangeError):
    """Caller asked for unknown columns or out-of-range row groups."""


# --------------------------------------------------------------------- diff
class DiffError(ParquetRangeError):
    """Base class for every failure raised by the version-diff feature."""


class ManifestError(DiffError):
    """A snapshot manifest is malformed or disagrees with the server identity.

    Raised before any data fetch when the canonical URL, strong ETag or entity
    length declared in the manifest cannot be proven against the live object.
    """


class SchemaDriftError(DiffError):
    """The two snapshots are not contract-compatible for the declared fields.

    Carries field-level diagnostics (missing field on a side, type mismatch,
    non-scalar key, ...).  A drift failure always aborts the diff: nothing is
    silently cast, null-filled or re-ordered.
    """

    def __init__(self, message: str, diagnostics: list[dict] | None = None) -> None:
        super().__init__(message)
        self.diagnostics: list[dict] = diagnostics or []


class DuplicateKeyError(DiffError):
    """A business key is not unique within one snapshot.

    Every occurrence carries its source identity, row group and row number so
    the offending records can be located offline.
    """

    def __init__(self, message: str, occurrences: list[dict] | None = None) -> None:
        super().__init__(message)
        self.occurrences: list[dict] = occurrences or []


class NullKeyError(DiffError):
    """A key value is null; business keys must be non-null top-level scalars."""


class UnsupportedKeyTypeError(DiffError):
    """The key column does not use an allowed scalar Arrow type."""


class SegmentIntegrityError(DiffError):
    """A spilled external-sort segment fails its format/checksum verification.

    The untrusted segment is quarantined (never merged) and the diff fails
    rather than publishing facts derived from corrupt temporary storage.
    """


class DiffPublishError(DiffError):
    """The diff facts could not be atomically published to the target paths."""


class DiffCancelled(QueryCancelled, DiffError):
    """A diff run was cancelled; all of its temporary segments are removed."""


# ------------------------------------------------------------- quality audit
class QualityError(ParquetRangeError):
    """Base class for every failure raised by the quality-audit feature."""


class ContractError(QualityError):
    """The quality contract is malformed, inconsistent or unsupported.

    Raised before a single row of data is read.  Carries ``diagnostics``: a
    list of ``{"location", "problem", ...}`` entries whose ``location`` is a
    JSON-path-like position inside the contract document (for example
    ``rules[2].min`` or ``fields.price``).
    """

    def __init__(self, message: str, diagnostics: list[dict] | None = None) -> None:
        super().__init__(message)
        self.diagnostics: list[dict] = diagnostics or []


class UnsupportedPatternError(ContractError):
    """A string pattern uses regex constructs the bounded engine rejects."""


class AuditSchemaError(QualityError):
    """The contract does not match the Parquet file's top-level schema.

    Raised after the footer is read but before any row data is fetched: a
    declared field is missing from the file, or the contract-declared logical
    type is not exactly the file's Arrow type.  Carries field-level
    ``diagnostics``; field *order* differences are never an error.
    """

    def __init__(self, message: str, diagnostics: list[dict] | None = None) -> None:
        super().__init__(message)
        self.diagnostics: list[dict] = diagnostics or []


class AuditRequestError(QualityError):
    """The audit request (identity / paths / limits) is malformed."""


class AuditPublishError(QualityError):
    """The violations/report pair could not be atomically published."""


class AuditCancelled(QueryCancelled, QualityError):
    """A quality audit was cancelled; its staging files are removed."""
