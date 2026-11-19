"""Real local HTTP test server with deterministic barriers and fault injection.

This is a genuine HTTP server (``http.server.ThreadingHTTPServer``) speaking
the Range/conditional protocol -- nothing about the network is mocked.  It:

* serves in-memory object versions keyed by path;
* records every request: method, path, Range/If-Match headers, status,
  number of bytes returned and the ETag that was served;
* supports blocking barriers so concurrency tests can pause a physical GET
  at known points;
* supports version swap (same URL, new bytes + new strong ETag);
* supports per-request fault injection: weak/missing ETag, 200 fallback,
  bad Content-Range, truncated body, no-Range support, 5xx, 412.
"""

from __future__ import annotations

import dataclasses
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit


@dataclasses.dataclass(slots=True)
class ObjectVersion:
    data: bytes
    etag: str
    accept_ranges: bool = True


@dataclasses.dataclass(slots=True)
class LogEntry:
    method: str
    path: str
    range_header: str | None
    if_match: str | None
    status: int
    bytes_sent: int
    etag_served: str | None
    content_range: str | None


class VersionedObjectStore:
    """Keeps the *full version history* of a path behind one current pointer.

    A conditional request (``If-Match``) can therefore address an older
    version even after a swap -- this lets a diff pin two explicit identities
    of the same URL without the second pin observing the other's bytes.  The
    default (no ``If-Match``) behaviour is unchanged: the current version is
    served exactly as before.
    """

    def __init__(self, current: ObjectVersion) -> None:
        self.current = current
        self.history: dict[str, ObjectVersion] = {current.etag: current}

    def swap(self, version: ObjectVersion) -> None:
        self.history[version.etag] = version
        self.current = version

    def select(self, if_match: str | None) -> ObjectVersion:
        if if_match is None:
            return self.current
        return self.history.get(if_match, self.current)


class FaultSpec:
    """Per-request fault configuration inspected by the handler."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.weak_etag = False
        self.missing_etag = False
        self.force_200 = False            # answer Range GET with whole object
        self.no_range_support = False    # ignore Range, answer 200 on probe too
        self.bad_content_range = False
        self.truncate_body = 0           # trim this many bytes off the body
        self.fail_times = 0              # answer 503 this many times first
        self.force_412 = False
        self.min_footer_too_small = False  # serve an object shorter than 12 B
        # When true, truncation/412/200 faults skip the 1-byte validation probe
        # so first-contact validation stays clean and only data GETs are faulted.
        self.skip_fault_on_probe = True

    def reset(self) -> None:
        with self._lock:
            self.weak_etag = False
            self.missing_etag = False
            self.force_200 = False
            self.no_range_support = False
            self.bad_content_range = False
            self.truncate_body = 0
            self.fail_times = 0
            self.force_412 = False
            self.min_footer_too_small = False
            self.skip_fault_on_probe = True


class Barrier:
    """Named rendezvous points with block/release and call counters.

    By default an *unarmed* barrier just records arrivals and lets requests
    pass.  A test calls ``arm(name)`` to make subsequent arrivals block until
    it calls ``release(name)`` -- this keeps ordinary tests free of hangs
    while giving concurrency tests a deterministic pause point.
    """

    def __init__(self) -> None:
        self._cond = threading.Condition()
        self._blocked: dict[str, int] = {}
        self._arrivals: dict[str, int] = {}
        self._released: set[str] = set()
        self._armed: set[str] = set()

    def arm(self, name: str) -> None:
        with self._cond:
            self._armed.add(name)
            self._released.discard(name)

    def arrive(self, name: str) -> None:
        with self._cond:
            self._arrivals[name] = self._arrivals.get(name, 0) + 1
            if name not in self._armed or name in self._released:
                self._cond.notify_all()
                return
            self._blocked[name] = self._blocked.get(name, 0) + 1
            self._cond.notify_all()
            while name not in self._released and self._blocked.get(name, 0) > 0:
                self._cond.wait(timeout=0.1)

    def wait_for_arrivals(self, name: str, count: int, timeout: float = 5.0) -> bool:
        import time

        deadline = time.monotonic() + timeout
        with self._cond:
            while self._arrivals.get(name, 0) < count:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not self._cond.wait(timeout=min(remaining, 0.05)):
                    return False
            return True

    def release(self, name: str) -> None:
        with self._cond:
            self._released.add(name)
            self._armed.discard(name)  # one-shot: later tests see default pass-through
            self._blocked.pop(name, None)
            self._cond.notify_all()

    def arrivals(self, name: str) -> int:
        with self._cond:
            return self._arrivals.get(name, 0)

    def reset(self) -> None:
        with self._cond:
            self._blocked.clear()
            self._arrivals.clear()
            self._released.clear()
            self._armed.clear()


class _Handler(BaseHTTPRequestHandler):
    server_version = "ParquetRangeTest/1.0"

    def log_message(self, *args: object) -> None:  # silence stderr noise
        pass

    # ------------------------------------------------------------ helpers
    @property
    def store(self) -> dict[str, ObjectVersion]:
        return self.server.store  # type: ignore[attr-defined]

    @property
    def fault(self) -> FaultSpec:
        return self.server.fault  # type: ignore[attr-defined]

    @property
    def barrier(self) -> Barrier:
        return self.server.barrier  # type: ignore[attr-defined]

    def _record(self, method, status, nbytes, etag, content_range) -> None:
        self.server.log.append(  # type: ignore[attr-defined]
            LogEntry(
                method=method,
                path=urlsplit(self.path).path,
                range_header=self.headers.get("Range"),
                if_match=self.headers.get("If-Match"),
                status=status,
                bytes_sent=nbytes,
                etag_served=etag,
                content_range=content_range,
            )
        )

    def _etag_headers(self, etag: str) -> list[tuple[str, str]]:
        if self.fault.missing_etag:
            return []
        if self.fault.weak_etag:
            return [("ETag", 'W/"weak-validator"')]
        return [("ETag", etag)]

    def _store_entry(self) -> "VersionedObjectStore | None":
        return self.store.get(urlsplit(self.path).path)

    def _current(self) -> ObjectVersion | None:
        entry = self._store_entry()
        if entry is None:
            return None
        return entry.select(self.headers.get("If-Match"))

    # --------------------------------------------------------------- HEAD
    def do_HEAD(self) -> None:
        entry = self._store_entry()
        if entry is None:
            self.send_error(404)
            self._record("HEAD", 404, 0, None, None)
            return
        if_match = self.headers.get("If-Match")
        if if_match is not None and if_match not in entry.history:
            self.send_response(412)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record("HEAD", 412, 0, entry.current.etag, None)
            return
        obj = entry.select(if_match)
        self.send_response(200)
        for k, v in self._etag_headers(obj.etag):
            self.send_header(k, v)
        self.send_header("Content-Length", str(len(obj.data)))
        if obj.accept_ranges and not self.fault.no_range_support:
            self.send_header("Accept-Ranges", "bytes")
        self.end_headers()
        self._record("HEAD", 200, 0, obj.etag, None)

    # ---------------------------------------------------------------- GET
    def do_GET(self) -> None:
        obj = self._current()
        if obj is None:
            self.send_error(404)
            self._record("GET", 404, 0, None, None)
            return
        total = len(obj.data)
        range_hdr = self.headers.get("Range")
        if_match = self.headers.get("If-Match")
        is_probe = range_hdr == "bytes=0-0"
        faults_active = not (self.fault.skip_fault_on_probe and is_probe)

        # Deterministic 5xx burst for retry tests.
        if faults_active and self.fault.fail_times > 0:
            self.fault.fail_times -= 1
            self.send_response(503)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record("GET", 503, 0, obj.etag, None)
            return
        if faults_active and self.fault.force_412:
            self.send_response(412)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record("GET", 412, 0, obj.etag, None)
            return
        # Conditional header enforcement: If-Match must equal current ETag.
        if if_match is not None and if_match != obj.etag:
            self.send_response(412)
            self.send_header("Content-Length", "0")
            self.end_headers()
            self._record("GET", 412, 0, obj.etag, None)
            return

        # Barrier for physical GETs (data fetches carry a Range header).
        if range_hdr:
            self.barrier.arrive(f"GET:{self.path}")

        no_range = self.fault.no_range_support or not obj.accept_ranges
        if not range_hdr or no_range or self.fault.force_200:
            # Whole-object fallback path (also used by misbehaving-server tests).
            self.send_response(200)
            for k, v in self._etag_headers(obj.etag):
                self.send_header(k, v)
            body = obj.data
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                return  # client cancelled; nothing more to send
            self._record("GET", 200, len(body), obj.etag, None)
            return

        # Parse "bytes=start-end".
        try:
            spec = range_hdr.strip()
            assert spec.startswith("bytes=")
            start_s, end_s = spec[len("bytes="):].split("-")
            start = int(start_s)
            end = int(end_s) if end_s else total - 1
        except (AssertionError, ValueError):
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()
            self._record("GET", 416, 0, obj.etag, None)
            return
        if start < 0 or end >= total or start > end:
            self.send_response(416)
            self.send_header("Content-Range", f"bytes */{total}")
            self.end_headers()
            self._record("GET", 416, 0, obj.etag, None)
            return

        body = obj.data[start: end + 1]
        if faults_active and self.fault.truncate_body:
            body = body[: max(0, len(body) - self.fault.truncate_body)]
        cr = f"bytes {start}-{end}/{total}"
        if faults_active and self.fault.bad_content_range:
            cr = f"bytes {start}-{end + 999}/{total}"
        self.send_response(206)
        for k, v in self._etag_headers(obj.etag):
            self.send_header(k, v)
        self.send_header("Content-Range", cr)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            return
        self._record("GET", 206, len(body), obj.etag, cr)


class TestHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, addr: tuple[str, int] = ("127.0.0.1", 0)) -> None:
        super().__init__(addr, _Handler)
        self.store: dict[str, ObjectVersion] = {}
        self.log: list[LogEntry] = []
        self.fault = FaultSpec()
        self.barrier = Barrier()
        self._lock = threading.Lock()

    def put_object(self, path: str, data: bytes, etag: str,
                   accept_ranges: bool = True) -> None:
        with self._lock:
            self.store[path] = VersionedObjectStore(
                ObjectVersion(bytes(data), etag, accept_ranges)
            )

    def swap_version(self, path: str, data: bytes, etag: str, *,
                     retain_history: bool = False) -> None:
        """Replace the current version.

        By default the old version is dropped (so stale conditional requests
        get a real 412).  Pass ``retain_history=True`` to keep every version
        addressable by its own ETag -- used by diff tests that pin two
        explicit identities of the same URL simultaneously.
        """
        with self._lock:
            entry = self.store.get(path)
            if entry is None:
                raise KeyError(path)
            new_version = ObjectVersion(bytes(data), etag, True)
            if retain_history:
                entry.swap(new_version)
            else:
                self.store[path] = VersionedObjectStore(new_version)

    def base_url(self) -> str:
        host, port = self.server_address[:2]
        return f"http://{host}:{port}"

    def url_for(self, path: str) -> str:
        if not path.startswith("/"):
            path = "/" + path
        return self.base_url() + path

    def range_gets(self, path: str | None = None) -> list[LogEntry]:
        with self._lock:
            return [
                e
                for e in self.log
                if e.method == "GET" and e.range_header and (path is None or e.path == path)
            ]

    def physical_bytes(self, path: str | None = None) -> int:
        return sum(e.bytes_sent for e in self.range_gets(path))

    def clear_log(self) -> None:
        with self._lock:
            self.log.clear()

    def served_etags(self, path: str | None = None) -> list[str | None]:
        with self._lock:
            return [e.etag_served for e in self.log if path is None or e.path == path]
