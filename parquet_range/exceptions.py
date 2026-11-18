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
