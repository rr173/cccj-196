"""Public entry point: open a remote Parquet object and query a fixed version.

Typical use::

    from parquet_range import RemoteParquetClient

    with RemoteParquetClient(disk_dir="./.pqr-cache") as client:
        obj = client.open("http://127.0.0.1:8000/big.parquet")
        info = obj.inspect()
        table = obj.scan(columns=["id", "events"], row_groups=[0, 3])

``open`` performs the mandatory first-contact validation (length, strong
ETag, proven Range support) and returns a :class:`VersionedObject` pinned to
exactly ``(canonical_url, etag, length)`` for its whole lifetime.  A version
change mid-query fails that query; the caller may call ``open`` again to get
a new, separately isolated versioned object.
"""

from __future__ import annotations

import dataclasses

from .cache import BlockScheduler, QueryToken
from .disk_cache import DiskCache
from .exceptions import VersionMismatchError
from .identity import ObjectIdentity, canonical_url
from .parquet_reader import InspectResult, ParquetScanClient
from .stats import Stats
from .transport import HttpTransport


@dataclasses.dataclass(slots=True)
class ClientConfig:
    block_size: int = 1 << 20            # 1 MiB fixed cache blocks
    mem_budget: int = 64 << 20           # 64 MiB
    disk_budget: int = 512 << 20         # 512 MiB
    max_inflight_bytes: int = 16 << 20   # 16 MiB held by in-flight GETs
    max_inflight_requests: int = 8
    max_merge_gap_blocks: int = 2
    timeout: float = 30.0
    max_retries: int = 3
    backoff: float = 0.05


class RemoteParquetClient:
    def __init__(
        self,
        *,
        disk_dir: str | None = ".parquet_range_cache",
        config: ClientConfig | None = None,
        transport: HttpTransport | None = None,
    ) -> None:
        self.config = config or ClientConfig()
        self.stats = Stats()
        self._owns_transport = transport is None
        self.transport = transport or HttpTransport(
            self.stats,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            backoff=self.config.backoff,
        )
        self.disk = DiskCache(disk_dir, self.stats) if disk_dir else None
        self.scheduler = BlockScheduler(
            self.transport,
            self.stats,
            block_size=self.config.block_size,
            mem_budget=self.config.mem_budget,
            disk_budget=self.config.disk_budget,
            max_inflight_bytes=self.config.max_inflight_bytes,
            max_inflight_requests=self.config.max_inflight_requests,
            max_merge_gap_blocks=self.config.max_merge_gap_blocks,
            disk_cache=self.disk,
        )
        self.reader = ParquetScanClient(self.scheduler, self.stats)
        # url -> currently known identity, purely advisory for error messages
        self._known: dict[str, ObjectIdentity] = {}

    # ------------------------------------------------------------- open
    def open(self, url: str) -> "VersionedObject":
        canon = canonical_url(url)
        probe = self.transport.validate_object(canon)
        identity = ObjectIdentity(
            url=canon, etag=probe.etag, length=probe.length
        )
        previous = self._known.get(canon)
        if previous is not None and previous.etag != identity.etag:
            # Not an error: a *new* versioned object is created; old caches
            # stay isolated under the old version hash.  Flag it explicitly.
            self._known[canon] = identity
            raise VersionMismatchError(
                f"{canon} changed since it was first opened: "
                f"{previous.etag!r} -> {identity.etag!r}. "
                "open_fresh() to accept the new version; cached old bytes "
                "remain version-isolated and are never mixed."
            )
        self._known[canon] = identity
        return VersionedObject(identity, self)

    def open_fresh(self, url: str) -> "VersionedObject":
        """Accept whatever the current version is and pin a new object."""
        canon = canonical_url(url)
        probe = self.transport.validate_object(canon)
        identity = ObjectIdentity(url=canon, etag=probe.etag, length=probe.length)
        self._known[canon] = identity
        return VersionedObject(identity, self)

    def open_identity(
        self, url: str, expected_etag: str, expected_length: int
    ) -> "VersionedObject":
        """Pin the *exact* manifest-declared identity of a snapshot.

        Both the conditional HEAD and the Range probe carry the expected
        strong ETag; any 412, ETag drift or entity-length drift fails before a
        single data byte is read.  The advisory ``_known`` table is deliberately
        not updated, so the same URL can back two distinct snapshot identities
        in one diff run without one pin overwriting the other.
        """
        canon = canonical_url(url)
        probe = self.transport.validate_pinned(canon, expected_etag, expected_length)
        identity = ObjectIdentity(
            url=canon, etag=probe.etag, length=probe.length
        )
        return VersionedObject(identity, self)

    def new_token(self) -> QueryToken:
        return QueryToken(self.scheduler)

    def cache_clear(self, url: str | None = None) -> dict[str, int]:
        identity = None
        if url is not None:
            identity = self._known.get(canonical_url(url))
        return self.scheduler.clear_caches(identity)

    def close(self) -> None:
        self.scheduler.close()
        if self._owns_transport:
            self.transport.close()

    def __enter__(self) -> "RemoteParquetClient":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class VersionedObject:
    """A handle pinned to one concrete object version."""

    def __init__(self, identity: ObjectIdentity, client: RemoteParquetClient) -> None:
        self.identity = identity
        self._client = client

    @property
    def version_id(self) -> str:
        return self.identity.version_id()

    def inspect(self, token: QueryToken | None = None) -> InspectResult:
        return self._client.reader.inspect(self.identity, token=token)

    def scan(
        self,
        columns: list[str],
        row_groups: list[int],
        token: QueryToken | None = None,
    ):
        return self._client.reader.scan(
            self.identity, columns, row_groups, token=token
        )

    def __repr__(self) -> str:
        return f"VersionedObject({self.identity.describe()})"
