"""Local-file HiCache storage client on io_uring (``mooncake.uring``).

A drop-in ``Hf3fsClient`` for a local NVMe file: page slots are addressed by
byte offset, ``batch_read`` / ``batch_write`` submit one io_uring batch per
call, and O_DIRECT is used whenever every buffer, length and offset of a call
is 4 KiB aligned (a pinned host pool with page-aligned strides qualifies);
anything else goes through a buffered ring on the same file. The pinned host
pool can be registered once as the ring's fixed buffer (``register_buffer``).
"""

import logging
import os
import threading
from typing import List, Optional

import torch

from sglang.srt.mem_cache.storage.hf3fs.hf3fs_client import Hf3fsClient

logger = logging.getLogger(__name__)

DIRECT_IO_ALIGNMENT = 4096
DEFAULT_QUEUE_DEPTH = 32


def import_mooncake_uring():
    """Import ``mooncake.uring``, raising an ImportError that says what to do."""
    try:
        from mooncake import uring
    except ImportError as e:
        raise ImportError(
            "HiCache io_uring storage needs the `mooncake.uring` module; install "
            "a mooncake-transfer-engine build with io_uring support (Linux, "
            "liburing) or select client='usrbio' / 'mock'."
        ) from e
    if not uring.SUPPORT_URING:
        raise ImportError(
            "mooncake.uring was built without io_uring support "
            "(SUPPORT_URING is False); rebuild Mooncake with liburing."
        )
    return uring


def _is_aligned(*values: int) -> bool:
    return all(v % DIRECT_IO_ALIGNMENT == 0 for v in values)


class UringLocalClient(Hf3fsClient):
    """``Hf3fsClient`` over a local file driven by ``mooncake.uring.UringFile``."""

    def __init__(
        self,
        path: str,
        size: int,
        bytes_per_page: int,
        entries: int,
        queue_depth: Optional[int] = None,
        page_aligned: Optional[bool] = None,
    ):
        self._uring = import_mooncake_uring()
        self.path = path
        self.size = size
        self.bytes_per_page = bytes_per_page
        self.entries = entries
        self.queue_depth = queue_depth or max(DEFAULT_QUEUE_DEPTH, entries)

        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            os.ftruncate(fd, size)
        finally:
            os.close(fd)

        # O_DIRECT needs page-multiple lengths; pointers and offsets are checked
        # per call unless the host pool is known to be misaligned.
        self._direct_capable = _is_aligned(bytes_per_page) and page_aligned is not False
        self._direct_file = None
        self._buffered_file = None
        self._lock = threading.Lock()
        logger.info(
            "UringLocalClient initialized: path=%s, size=%d, bytes_per_page=%d, "
            "entries=%d, direct_io=%s",
            path,
            size,
            bytes_per_page,
            entries,
            self._direct_capable,
        )

    @staticmethod
    def register_buffer(ptr: int, length: int) -> bool:
        """Register a pinned host region (the HiCache host pool) as the ring's
        fixed buffer so submissions from it skip per-I/O page pinning."""
        uring = import_mooncake_uring()
        ok = bool(uring.register_global_buffer(ptr, length))
        if not ok:
            logger.warning(
                "io_uring fixed-buffer registration failed for %d bytes at 0x%x; "
                "continuing with unregistered buffers.",
                length,
                ptr,
            )
        return ok

    def set_page_aligned_hint(self, aligned: bool) -> None:
        """``is_stride_page_aligned`` of the host pool: False turns O_DIRECT off;
        True keeps the per-call check, since a caller may pass a staging tensor."""
        if not aligned:
            self._direct_capable = False

    def _open(self, direct: bool):
        flags = os.O_RDWR | os.O_DIRECT if direct else os.O_RDWR
        return self._uring.UringFile(
            self.path, flags, queue_depth=self.queue_depth, direct_io=direct
        )

    def _file_for(self, buf_ptrs: List[int], lengths: List[int], offsets: List[int]):
        if self._direct_capable and _is_aligned(*buf_ptrs, *lengths, *offsets):
            if self._direct_file is None:
                try:
                    self._direct_file = self._open(direct=True)
                except OSError as e:
                    # e.g. tmpfs; buffered for the client's lifetime.
                    logger.warning(
                        "O_DIRECT unavailable for %s (%s); using buffered io_uring.",
                        self.path,
                        e,
                    )
                    self._direct_capable = False
            if self._direct_file is not None:
                return self._direct_file
        if self._buffered_file is None:
            self._buffered_file = self._open(direct=False)
        return self._buffered_file

    @staticmethod
    def _staging(tensor: torch.Tensor, copy_in: bool) -> torch.Tensor:
        if tensor.is_contiguous():
            return tensor
        if copy_in:
            return tensor.contiguous()
        return torch.empty(tensor.shape, dtype=tensor.dtype)

    def _results(self, raw: List[int], lengths: List[int], op: str) -> List[int]:
        results = []
        for ret, length in zip(raw, lengths):
            if ret < 0:
                logger.error(
                    "io_uring %s failed: %s",
                    op,
                    os.strerror(-ret) if -ret < 200 else ret,
                )
                results.append(0)
            else:
                if ret != length:
                    logger.warning("Short %s: expected %d, got %d", op, length, ret)
                results.append(ret)
        return results

    def batch_read(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)
        bufs = [self._staging(t, copy_in=False) for t in tensors]
        buf_ptrs = [b.data_ptr() for b in bufs]
        lengths = [b.numel() * b.itemsize for b in bufs]
        with self._lock:
            uring_file = self._file_for(buf_ptrs, lengths, offsets)
            try:
                raw = uring_file.batch_read(buf_ptrs, lengths, list(offsets))
            except Exception as e:
                logger.error(f"Error submitting io_uring batch read: {e}")
                return [0] * len(offsets)
        for tensor, buf in zip(tensors, bufs):
            if buf is not tensor:
                tensor.copy_(buf)
        return self._results(raw, lengths, "read")

    def batch_write(self, offsets: List[int], tensors: List[torch.Tensor]) -> List[int]:
        self.check(offsets, tensors)
        bufs = [self._staging(t, copy_in=True) for t in tensors]
        buf_ptrs = [b.data_ptr() for b in bufs]
        lengths = [b.numel() * b.itemsize for b in bufs]
        with self._lock:
            uring_file = self._file_for(buf_ptrs, lengths, offsets)
            try:
                raw = uring_file.batch_write(buf_ptrs, lengths, list(offsets))
            except Exception as e:
                logger.error(f"Error submitting io_uring batch write: {e}")
                return [0] * len(offsets)
        return self._results(raw, lengths, "write")

    def check(self, offsets: List[int], tensors: List[torch.Tensor]) -> None:
        sizes = [t.numel() * t.itemsize for t in tensors]
        if (
            len(offsets) > self.entries
            or len(offsets) != len(sizes)
            or any(
                offset < 0 or offset + size > self.size
                for offset, size in zip(offsets, sizes)
            )
            or any(size > self.bytes_per_page for size in sizes)
            or any(t.device.type != "cpu" for t in tensors)
        ):
            raise ValueError(f"UringLocalClient.check: {offsets=}, {sizes=}")

    def get_size(self) -> int:
        return self.size

    def close(self) -> None:
        with self._lock:
            for uring_file in (self._direct_file, self._buffered_file):
                if uring_file is not None:
                    try:
                        uring_file.close()
                    except Exception as e:
                        logger.error(f"Error closing UringLocalClient: {e}")
            self._direct_file = None
            self._buffered_file = None

    def flush(self) -> None:
        with self._lock:
            for uring_file in (self._direct_file, self._buffered_file):
                if uring_file is not None:
                    ret = uring_file.datasync()
                    if ret < 0:
                        logger.error(f"Error flushing UringLocalClient: {ret}")
