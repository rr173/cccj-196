"""Seekable/sparse random-access file over a remote versioned object.

This is the only object PyArrow sees.  All reads go through the
:class:`~parquet_range.cache.BlockScheduler`, so every byte is either an
already-validated cache overlay or a conditional Range GET pinned to the
object's version -- PyArrow never performs any remote I/O itself.

Implements the bare ``RawIOBase`` interface (``read``/``readinto``/``seek``/
``tell``/``seekable``) that ``pyarrow.PythonFile`` expects.
"""

from __future__ import annotations

import io

from .cache import BlockScheduler, QueryToken
from .identity import ObjectIdentity


class SparseObjectFile(io.RawIOBase):
    def __init__(
        self,
        identity: ObjectIdentity,
        scheduler: BlockScheduler,
        *,
        token: QueryToken | None = None,
    ) -> None:
        super().__init__()
        self.identity = identity
        self._scheduler = scheduler
        self._token = token
        self._pos = 0
        # Logical read accounting (distinct from physical wire counters).
        self.logical_reads: list[tuple[int, int]] = []

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def writable(self) -> bool:
        return False

    # ------------------------------------------------------------ seeking
    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        if whence == io.SEEK_SET:
            target = offset
        elif whence == io.SEEK_CUR:
            target = self._pos + offset
        elif whence == io.SEEK_END:
            target = self.identity.length + offset
        else:  # pragma: no cover - stdlib guarantees whence in 0..2
            raise ValueError(f"invalid whence {whence}")
        if target < 0:
            raise ValueError(f"negative seek position {target}")
        self._pos = target
        return target

    def tell(self) -> int:
        return self._pos

    # -------------------------------------------------------------- reading
    def readinto(self, buffer) -> int:
        n = len(buffer)
        remaining = self.identity.length - self._pos
        n = min(n, max(remaining, 0))
        if n == 0:
            return 0
        data = self._scheduler.read(
            self.identity, self._pos, n, token=self._token
        )
        buffer[: len(data)] = data
        self.logical_reads.append((self._pos, len(data)))
        self._pos += len(data)
        return len(data)

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.identity.length - self._pos
        if size == 0:
            return b""
        data = self._scheduler.read(
            self.identity, self._pos, size, token=self._token
        )
        self.logical_reads.append((self._pos, len(data)))
        self._pos += len(data)
        return data

    def readall(self) -> bytes:  # pragma: no cover - RawIOBase default is fine
        return super().readall()

    def close(self) -> None:
        # We do not own the scheduler/transport; closing the file is a no-op
        # for shared resources.  The Client/Store lifecycle owns those.
        super().close()
