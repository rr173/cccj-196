"""HTTP transport: strict conditional Range GETs over HTTPX.

Only this module performs remote I/O.  There is no fsspec, no S3 filesystem,
no remote dataset scanner anywhere in the package.

Contract for every data GET:
  * ``Range: bytes=start-end`` plus ``If-Match: <pinned strong etag>``
  * success requires HTTP 206 with:
      - matching strong ETag (else VersionMismatchError, even on 206)
      - ``Content-Range: bytes start-end/total`` with exact start/end/total
      - a body of exactly ``end-start+1`` bytes (truncation is an error)
  * HTTP 412 -> VersionMismatchError (never retried)
  * HTTP 200 -> RangeProtocolError: whole-object fallback is forbidden
  * 416 / bad Content-Range / missing headers -> RangeProtocolError
Transient network errors and 5xx are retried at most ``max_retries`` times,
always against the *same* pinned version; exhaustion raises RetryExhaustedError.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Callable

import httpx

from .exceptions import (
    RangeProtocolError,
    RetryExhaustedError,
    TruncatedResponseError,
    VersionMismatchError,
    WeakOrMissingETagError,
)
from .identity import ObjectIdentity, require_strong_etag
from .stats import Stats

# Statuses worth a bounded retry; 412/200/416/bad headers are never retried.
_RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


def _content_length(head) -> int:
    length_hdr = head.headers.get("content-length")
    if length_hdr is None:
        raise RangeProtocolError("HEAD response lacks Content-Length")
    try:
        length = int(length_hdr)
    except ValueError as exc:
        raise RangeProtocolError(
            f"non-integer Content-Length {length_hdr!r}"
        ) from exc
    if length <= 0:
        from .exceptions import EmptyObjectError

        raise EmptyObjectError(f"object reports length {length}")
    return length


@dataclasses.dataclass(slots=True)
class ProbeResult:
    etag: str
    length: int
    accepts_ranges: bool


class HttpTransport:
    def __init__(
        self,
        stats: Stats,
        *,
        timeout: float = 30.0,
        max_retries: int = 3,
        backoff: float = 0.05,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        self.stats = stats
        self.max_retries = max_retries
        self.backoff = backoff
        # One shared client; HTTPX is safe for concurrent requests on it.
        self._client = httpx.Client(timeout=timeout, transport=transport)
        # Test hook: barrier() is invoked with (method, url, headers) before
        # every physical request, allowing deterministic concurrency tests.
        self.barrier: Callable[[str, str, dict[str, str]], None] | None = None

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "HttpTransport":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    # -------------------------------------------------------- first contact
    def validate_object(self, url: str) -> ProbeResult:
        """HEAD for length/ETag, then a 1-byte GET proving real Range support.

        Some servers advertise ``Accept-Ranges: bytes`` on HEAD yet return 200
        on GET, so Range ability is proven, not assumed.
        """
        head = self._head(url, if_match=None)
        etag = require_strong_etag(head.headers.get("etag"))
        length = _content_length(head)
        return self._prove_range(url, etag, length)

    def validate_pinned(
        self, url: str, expected_etag: str, expected_length: int
    ) -> ProbeResult:
        """Verify that the live object is *exactly* the manifest-declared version.

        Unlike :meth:`validate_object`, every request is conditional on the
        expected strong ETag from the snapshot manifest:

        * ``HEAD`` carries ``If-Match: <expected_etag>`` -- a version that
          changed before first contact answers 412 and the diff aborts;
        * the one-byte Range probe likewise carries the same ``If-Match``;
        * the observed strong ETag and entity length must equal the manifest.

        This makes the two snapshots of a diff two *explicit* identities even
        when they share a URL: probing one can never silently observe the
        other's current bytes.
        """
        etag = require_strong_etag(expected_etag)
        head = self._head(url, if_match=etag)
        if head.status_code == 412:
            self.stats.inc("status_412")
            raise VersionMismatchError(
                f"conditional HEAD for {url} returned 412: the pinned ETag "
                f"{etag!r} no longer matches the live object"
            )
        if head.status_code != 200:
            raise RangeProtocolError(
                f"conditional HEAD {url} returned {head.status_code}, expected 200"
            )
        observed_etag = require_strong_etag(head.headers.get("etag"))
        if observed_etag != etag:
            raise VersionMismatchError(
                f"manifest ETag {etag!r} disagrees with live ETag {observed_etag!r}"
            )
        length = _content_length(head)
        if length != expected_length:
            raise VersionMismatchError(
                f"manifest entity length {expected_length} disagrees with live "
                f"Content-Length {length} for {url}"
            )
        result = self._prove_range(url, etag, length)
        if result.length != expected_length or result.etag != etag:  # defensive
            raise VersionMismatchError("pinned probe returned an unexpected identity")
        return result

    def _head(self, url: str, *, if_match: str | None):
        self.stats.inc("head_requests")
        headers = {"If-Match": if_match} if if_match is not None else None
        head = self._client.head(
            url, headers=headers, follow_redirects=True
        )
        return head

    def _prove_range(self, url: str, etag: str, length: int) -> ProbeResult:
        """One-byte conditional GET proving the server really honours Range."""
        self.stats.inc("probe_requests")
        headers = {"Range": "bytes=0-0", "If-Match": etag}
        attempt = 0
        while True:
            self._run_barrier("PROBE", url, headers)
            try:
                resp = self._client.get(url, headers=headers, follow_redirects=True)
                break
            except (httpx.HTTPError, OSError) as exc:
                if attempt >= self.max_retries:
                    self.stats.inc("retry_exhausted")
                    raise RetryExhaustedError(
                        f"range probe failed after {attempt + 1} attempts: {exc!r}"
                    ) from exc
                attempt += 1
                self.stats.inc("retries")
                time.sleep(min(self.backoff * (2 ** (attempt - 1)), 2.0))
        if resp.status_code == 412:
            self.stats.inc("status_412")
            raise VersionMismatchError("object changed between HEAD and range probe")
        if resp.status_code == 200:
            raise RangeProtocolError(
                "server answered a Range probe with 200 (whole-object fallback); "
                "sparse reading is impossible"
            )
        if resp.status_code != 206:
            if resp.status_code in _RETRY_STATUS:
                raise RangeProtocolError(
                    f"range probe returned retryable {resp.status_code} without "
                    "recovery; server is unhealthy"
                )
            raise RangeProtocolError(
                f"range probe returned {resp.status_code}, Range not supported"
            )
        self._check_content_range(resp, 0, 0, length)
        probe_etag = require_strong_etag(resp.headers.get("etag"))
        if probe_etag != etag:
            raise VersionMismatchError(
                f"ETag changed during probe: HEAD {etag!r} vs GET {probe_etag!r}"
            )
        if len(resp.content) != 1:
            raise TruncatedResponseError(
                f"range probe body has {len(resp.content)} bytes, expected 1"
            )
        self.stats.add("wire_bytes", len(resp.content))
        return ProbeResult(etag=etag, length=length, accepts_ranges=True)

    # ------------------------------------------------------------- data GET
    def get_range(
        self,
        identity: ObjectIdentity,
        start: int,
        end: int,
        *,
        cancel_event=None,
    ) -> bytes:
        """Fetch inclusive [start, end] of a pinned object version.

        ``cancel_event`` (threading.Event) aborts only when *no other query*
        waits on the physical request; the scheduler decides that, here we
        simply honor an abort signal between/inside attempts.
        """
        if start < 0 or end < start or end >= identity.length:
            raise RangeProtocolError(
                f"refusing invalid range bytes={start}-{end} for length {identity.length}"
            )
        headers = {
            "Range": f"bytes={start}-{end}",
            "If-Match": identity.if_match,
        }
        attempt = 0
        last_exc: Exception | None = None
        while True:
            if cancel_event is not None and cancel_event.is_set():
                self.stats.inc("cancelled_requests")
                from .exceptions import QueryCancelled

                raise QueryCancelled("physical request aborted: last waiter left")
            self._run_barrier("GET", identity.url, headers)
            try:
                with self._client.stream(
                    "GET", identity.url, headers=headers, follow_redirects=True
                ) as resp:
                    return self._consume_stream(
                        resp, identity, start, end, cancel_event
                    )
            except VersionMismatchError:
                raise  # never retry a version conflict
            except RangeProtocolError as exc:
                # Protocol violations are deterministic; no point retrying.
                raise
            except (httpx.HTTPError, OSError) as exc:
                last_exc = exc
            if attempt >= self.max_retries:
                self.stats.inc("retry_exhausted")
                raise RetryExhaustedError(
                    f"range bytes={start}-{end} failed after {attempt + 1} attempts: "
                    f"{last_exc!r}"
                ) from last_exc
            attempt += 1
            self.stats.inc("retries")
            # Bounded exponential backoff; same version stays pinned.
            if cancel_event is not None:
                if cancel_event.wait(min(self.backoff * (2 ** (attempt - 1)), 2.0)):
                    self.stats.inc("cancelled_requests")
                    from .exceptions import QueryCancelled

                    raise QueryCancelled("cancelled during retry backoff")
            else:
                time.sleep(min(self.backoff * (2 ** (attempt - 1)), 2.0))

    def _consume_stream(self, resp, identity, start: int, end: int,
                        cancel_event) -> bytes:
        status = resp.status_code
        if status == 412:
            self.stats.inc("status_412")
            resp.read()
            raise VersionMismatchError(
                f"412 Precondition Failed for {identity.url}: object version changed "
                f"(pinned {identity.etag!r}); current query must fail"
            )
        if status == 200:
            resp.read()
            raise RangeProtocolError(
                "server returned 200 instead of 206 for a Range request; "
                "whole-object download fallback is forbidden"
            )
        if status == 416:
            resp.read()
            raise RangeProtocolError(
                f"416 Range Not Satisfiable for bytes={start}-{end}"
            )
        if status != 206:
            if status in _RETRY_STATUS:
                resp.read()
                raise httpx.HTTPStatusError(
                    f"retryable status {status}", request=resp.request, response=resp
                )
            self.stats.inc("status_other_errors")
            resp.read()
            raise RangeProtocolError(f"unexpected status {status} for Range GET")

        # 206: validate every protocol element before trusting a single byte.
        self._check_content_range(resp, start, end, identity.length)
        etag = resp.headers.get("etag")
        if not etag:
            raise RangeProtocolError("206 response missing ETag")
        if etag.startswith(("W/", "w/")):
            raise WeakOrMissingETagError(f"206 carries weak ETag {etag!r}")
        if etag != identity.etag:
            raise VersionMismatchError(
                f"206 ETag {etag!r} differs from pinned {identity.etag!r}; "
                "refusing to mix object versions"
            )
        expected = end - start + 1
        chunks: list[bytes] = []
        received = 0
        try:
            for chunk in resp.iter_bytes():
                if cancel_event is not None and cancel_event.is_set():
                    self.stats.inc("cancelled_requests")
                    from .exceptions import QueryCancelled

                    raise QueryCancelled("stream aborted: last waiter left")
                chunks.append(chunk)
                received += len(chunk)
                if received > expected:
                    raise RangeProtocolError(
                        f"oversized response: got {received} bytes, expected {expected}"
                    )
        except httpx.HTTPError as exc:
            raise exc  # handled by the bounded-retry loop in get_range
        if received != expected:
            raise TruncatedResponseError(
                f"truncated body for bytes={start}-{end}: "
                f"received {received}/{expected} bytes"
            )
        body = b"".join(chunks)
        self.stats.inc("physical_requests")
        self.stats.add("wire_bytes", received)
        return body

    @staticmethod
    def _check_content_range(resp, start: int, end: int, total: int) -> None:
        cr = resp.headers.get("content-range")
        if not cr:
            raise RangeProtocolError("206 response missing Content-Range header")
        prefix = f"bytes {start}-{end}/{total}"
        if cr.strip() != prefix:
            raise RangeProtocolError(
                f"bad Content-Range {cr!r}, expected {prefix!r}"
            )
        # Declared entity length must agree with the pinned length too.
        clen = resp.headers.get("content-length")
        if clen is not None and int(clen) != end - start + 1:
            raise RangeProtocolError(
                f"Content-Length {clen} disagrees with range size {end - start + 1}"
            )

    def _run_barrier(self, method: str, url: str, headers: dict[str, str]) -> None:
        if self.barrier is not None:
            self.barrier(method, url, dict(headers))
