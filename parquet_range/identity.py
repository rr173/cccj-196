"""Object identity: canonical URL + strong ETag + length.

A query's identity is fixed for its whole lifetime as the triple
``(canonical_url, etag, length)`` and every cache key contains the version,
so two versions of the same URL can never be stitched together.
"""

from __future__ import annotations

import dataclasses
import hashlib
import posixpath
from urllib.parse import urlsplit, urlunsplit, quote

from .exceptions import WeakOrMissingETagError


def canonical_url(url: str) -> str:
    """Normalize a URL into the canonical form used for identity/cache keys.

    - scheme/host lower-cased
    - default port removed
    - path percent-encoding normalized (``quote`` keeps reserved chars)
    - query/fragment kept (fragment is normally absent for http(s))
    """
    parts = urlsplit(url.strip())
    if parts.scheme not in ("http", "https"):
        raise ValueError(f"only http(s) URLs are supported, got scheme={parts.scheme!r}")
    host = parts.hostname or ""
    port = parts.port
    if port is not None and not (
        (parts.scheme == "http" and port == 80)
        or (parts.scheme == "https" and port == 443)
    ):
        netloc = f"{host}:{port}"
    else:
        netloc = host
    # Re-quote the path: unquote reserved sub-delims handled by quote's safe.
    raw_path = parts.path or "/"
    # Collapse "." / ".." dot segments (RFC 3986 §5.2.4) before re-quoting.
    normalized = posixpath.normpath(raw_path)
    if raw_path.endswith("/") and not normalized.endswith("/"):
        normalized += "/"
    path = quote(normalized, safe="/%!$&'()*+,;=:@~")
    return urlunsplit((parts.scheme.lower(), netloc, path, parts.query, parts.fragment))


def require_strong_etag(etag: str | None) -> str:
    """Return the strong ETag or raise :class:`WeakOrMissingETagError`.

    A weak validator (``W/"..."``) cannot prove byte identity, so it is
    rejected outright -- version pinning would be meaningless.
    """
    if not etag:
        raise WeakOrMissingETagError("server returned no ETag; version pinning impossible")
    etag = etag.strip()
    if etag.startswith(("W/", "w/")):
        raise WeakOrMissingETagError(
            f"weak ETag {etag!r} cannot pin an exact object version"
        )
    if not (etag.startswith('"') and etag.endswith('"') and len(etag) >= 2):
        raise WeakOrMissingETagError(f"malformed ETag {etag!r}")
    return etag


@dataclasses.dataclass(frozen=True, slots=True)
class ObjectIdentity:
    """The immutable identity of one concrete object version."""

    url: str          # canonical URL
    etag: str         # strong ETag, including the double quotes
    length: int       # entity length in bytes

    def __post_init__(self) -> None:
        if self.length <= 0:
            from .exceptions import EmptyObjectError

            raise EmptyObjectError(f"object {self.url} has non-positive length {self.length}")
        # Defensive: ensure the ETag is strong.
        require_strong_etag(self.etag)
        if not self.url.startswith(("http://", "https://")):
            raise ValueError("identity url must be canonical http(s) URL")

    def version_id(self) -> str:
        """Short stable version discriminator used in cache keys/disk paths."""
        digest = hashlib.sha256(f"{self.url}\n{self.etag}\n{self.length}".encode()).digest()
        return digest.hex()[:32]

    def block_key(self, index: int) -> str:
        """Cache key for one block; the version hash is always embedded."""
        return f"{self.version_id()}/b{index:012d}"

    @property
    def if_match(self) -> str:
        """Value for the ``If-Match`` conditional header."""
        return self.etag

    def describe(self) -> str:
        return f"{self.url} etag={self.etag} length={self.length} v={self.version_id()[:12]}"
