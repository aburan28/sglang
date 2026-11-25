"""Unit tests for the HiCache io_uring local-file client -- no server, no GPU.

``mooncake.uring`` is a native module that may be absent, so a fake module is
injected into ``sys.modules`` for the duration of each test. The fake
``UringFile`` records the open mode it was created with and serves I/O from
the raw buffer pointers with ``pread`` / ``pwrite``, which is what the real
ring does, so the O_DIRECT selection and the data round trip are both checked.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import ctypes
import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.mem_cache.storage.hf3fs.storage_hf3fs import (
    HiCacheHF3FS,
    create_hf3fs_client,
    resolve_hf3fs_client_kind,
)
from sglang.srt.mem_cache.storage.hf3fs.uring_local_client import (
    DIRECT_IO_ALIGNMENT,
    UringLocalClient,
)
from sglang.test.test_utils import CustomTestCase

PAGE = 4096


class _FakeUringFile:
    instances = []

    def __init__(self, path, flags, queue_depth=32, direct_io=False):
        self.path = path
        self.flags = flags
        self.queue_depth = queue_depth
        self.direct_io = direct_io
        self.fd = os.open(path, os.O_RDWR)
        self.datasync_calls = 0
        self.closed = False
        _FakeUringFile.instances.append(self)

    def read_aligned(self, buf_ptr, length, offset):
        return self.batch_read([buf_ptr], [length], [offset])[0]

    def write_aligned(self, buf_ptr, length, offset):
        return self.batch_write([buf_ptr], [length], [offset])[0]

    def batch_read(self, buf_ptrs, lengths, offsets):
        out = []
        for ptr, length, offset in zip(buf_ptrs, lengths, offsets):
            data = os.pread(self.fd, length, offset)
            ctypes.memmove(ptr, data, len(data))
            out.append(len(data))
        return out

    def batch_write(self, buf_ptrs, lengths, offsets):
        out = []
        for ptr, length, offset in zip(buf_ptrs, lengths, offsets):
            data = (ctypes.c_char * length).from_address(ptr).raw
            out.append(os.pwrite(self.fd, data, offset))
        return out

    def datasync(self):
        self.datasync_calls += 1
        return 0

    def close(self):
        if not self.closed:
            os.close(self.fd)
            self.closed = True


def _fake_mooncake(support=True):
    uring = types.ModuleType("mooncake.uring")
    uring.SUPPORT_URING = support
    uring.UringFile = _FakeUringFile
    uring.registered = []
    uring.register_global_buffer = lambda ptr, length: (
        uring.registered.append((ptr, length)) or True
    )
    uring.unregister_global_buffer = lambda: None
    pkg = types.ModuleType("mooncake")
    pkg.__path__ = []
    pkg.uring = uring
    return pkg, uring


def _aligned_bytes(n, misalign=0):
    """A uint8 view of ``n`` bytes whose data_ptr is 4 KiB aligned (+misalign)."""
    backing = torch.empty(n + 2 * DIRECT_IO_ALIGNMENT, dtype=torch.uint8)
    start = (-backing.data_ptr()) % DIRECT_IO_ALIGNMENT + misalign
    return backing[start : start + n]


class _UringTestBase(CustomTestCase):
    def setUp(self):
        super().setUp()
        _FakeUringFile.instances = []
        self.pkg, self.uring = _fake_mooncake()
        patcher = patch.dict(
            sys.modules, {"mooncake": self.pkg, "mooncake.uring": self.uring}
        )
        patcher.start()
        self.addCleanup(patcher.stop)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "hicache", "pages.bin")

    def _client(self, bytes_per_page=PAGE, entries=8, num_pages=16, **kw):
        client = UringLocalClient(
            self.path, num_pages * bytes_per_page, bytes_per_page, entries, **kw
        )
        self.addCleanup(client.close)
        return client


class TestUringLocalClientIO(_UringTestBase):
    def test_round_trip_uses_direct_io_for_aligned_batches(self):
        client = self._client()
        self.assertEqual(os.path.getsize(self.path), 16 * PAGE)
        pages = [_aligned_bytes(PAGE) for _ in range(3)]
        for page in pages:
            page.copy_(torch.randint(0, 256, (PAGE,), dtype=torch.uint8))
        offsets = [0, 5 * PAGE, 9 * PAGE]

        self.assertEqual(client.batch_write(offsets, pages), [PAGE] * 3)
        readback = [_aligned_bytes(PAGE) for _ in range(3)]
        self.assertEqual(client.batch_read(offsets, readback), [PAGE] * 3)
        for want, got in zip(pages, readback):
            self.assertTrue(torch.equal(want, got))

        self.assertEqual(len(_FakeUringFile.instances), 1)
        ring = _FakeUringFile.instances[0]
        self.assertTrue(ring.direct_io)
        self.assertTrue(ring.flags & os.O_DIRECT)
        self.assertEqual(ring.queue_depth, 32)

    def test_misaligned_buffers_fall_back_to_buffered_ring(self):
        client = self._client()
        page = _aligned_bytes(PAGE, misalign=1)
        page.copy_(torch.randint(0, 256, (PAGE,), dtype=torch.uint8))
        self.assertEqual(client.batch_write([PAGE], [page]), [PAGE])
        got = _aligned_bytes(PAGE, misalign=1)
        self.assertEqual(client.batch_read([PAGE], [got]), [PAGE])
        self.assertTrue(torch.equal(page, got))
        self.assertEqual([r.direct_io for r in _FakeUringFile.instances], [False])

        # An aligned call afterwards still gets the direct ring.
        aligned = _aligned_bytes(PAGE)
        client.batch_read([PAGE], [aligned])
        self.assertTrue(torch.equal(page, aligned))
        self.assertEqual(
            sorted(r.direct_io for r in _FakeUringFile.instances), [False, True]
        )

    def test_misaligned_offset_falls_back_to_buffered_ring(self):
        client = self._client(bytes_per_page=PAGE, num_pages=4)
        page = _aligned_bytes(PAGE)
        client.batch_write([100], [page])
        self.assertEqual([r.direct_io for r in _FakeUringFile.instances], [False])

    def test_page_size_not_multiple_of_4k_never_uses_direct_io(self):
        client = self._client(bytes_per_page=1000)
        client.batch_write([0], [_aligned_bytes(1000)])
        self.assertEqual([r.direct_io for r in _FakeUringFile.instances], [False])

    def test_host_pool_hint_disables_direct_io(self):
        client = self._client()
        client.set_page_aligned_hint(False)
        client.batch_write([0], [_aligned_bytes(PAGE)])
        self.assertEqual([r.direct_io for r in _FakeUringFile.instances], [False])

    def test_direct_open_failure_falls_back_for_the_client_lifetime(self):
        client = self._client()
        real_open = client._open

        def failing_open(direct):
            if direct:
                raise OSError(22, "Invalid argument")
            return real_open(direct)

        client._open = failing_open
        page = _aligned_bytes(PAGE)
        page.copy_(torch.randint(0, 256, (PAGE,), dtype=torch.uint8))
        self.assertEqual(client.batch_write([0], [page]), [PAGE])
        client.batch_write([PAGE], [page])
        self.assertEqual([r.direct_io for r in _FakeUringFile.instances], [False])

    def test_non_contiguous_tensors_are_staged(self):
        client = self._client()
        src = torch.randint(0, 256, (64, 64), dtype=torch.uint8).t()
        self.assertFalse(src.is_contiguous())
        self.assertEqual(client.batch_write([0], [src]), [PAGE])
        dst = torch.zeros((64, 64), dtype=torch.uint8).t()
        self.assertEqual(client.batch_read([0], [dst]), [PAGE])
        self.assertTrue(torch.equal(src, dst))

    def test_negative_ring_status_reports_zero_bytes(self):
        client = self._client()
        page = _aligned_bytes(PAGE)
        client.batch_write([0], [page])
        ring = _FakeUringFile.instances[0]
        ring.batch_read = lambda ptrs, lens, offs: [-5] * len(ptrs)
        self.assertEqual(client.batch_read([0], [page]), [0])

    def test_check_rejects_bad_batches(self):
        client = self._client(entries=2, num_pages=4)
        page = _aligned_bytes(PAGE)
        with self.assertRaises(ValueError):  # more entries than the ring holds
            client.check([0, PAGE, 2 * PAGE], [page, page, page])
        with self.assertRaises(ValueError):  # past the end of the file
            client.check([4 * PAGE], [page])
        with self.assertRaises(ValueError):  # negative offset
            client.check([-PAGE], [page])
        with self.assertRaises(ValueError):  # larger than a page slot
            client.check([0], [_aligned_bytes(PAGE + 1)])
        with self.assertRaises(ValueError):  # offsets / tensors mismatch
            client.check([0, PAGE], [page])
        client.check([0, PAGE], [page, page])

    def test_flush_and_close(self):
        client = self._client()
        client.flush()  # nothing open yet: no-op
        client.batch_write([0], [_aligned_bytes(PAGE)])
        client.batch_write([100], [_aligned_bytes(PAGE)])
        client.flush()
        rings = _FakeUringFile.instances
        self.assertEqual([r.datasync_calls for r in rings], [1, 1])
        self.assertEqual(client.get_size(), 16 * PAGE)
        client.close()
        self.assertTrue(all(r.closed for r in rings))
        client.close()  # idempotent

    def test_register_buffer_uses_global_registration(self):
        buf = torch.empty(2 * PAGE, dtype=torch.uint8)
        self.assertTrue(UringLocalClient.register_buffer(buf.data_ptr(), 2 * PAGE))
        self.assertEqual(self.uring.registered, [(buf.data_ptr(), 2 * PAGE)])


class TestUringImportErrors(CustomTestCase):
    def test_missing_module_gives_actionable_import_error(self):
        with patch.dict(sys.modules, {"mooncake": None, "mooncake.uring": None}):
            with self.assertRaisesRegex(ImportError, "mooncake.uring"):
                UringLocalClient("/tmp/x.bin", PAGE, PAGE, 1)

    def test_build_without_uring_support_is_rejected(self):
        pkg, uring = _fake_mooncake(support=False)
        with patch.dict(sys.modules, {"mooncake": pkg, "mooncake.uring": uring}):
            with self.assertRaisesRegex(ImportError, "SUPPORT_URING"):
                UringLocalClient("/tmp/x.bin", PAGE, PAGE, 1)


class TestHf3fsClientSelection(_UringTestBase):
    def test_explicit_client_and_env_selection(self):
        self.assertEqual(resolve_hf3fs_client_kind(None, use_mock=True), "mock")
        self.assertEqual(resolve_hf3fs_client_kind("uring"), "uring")
        self.assertEqual(resolve_hf3fs_client_kind("USRBIO"), "usrbio")
        with self.assertRaises(ValueError):
            resolve_hf3fs_client_kind("aio")
        with envs.SGLANG_HICACHE_FILE_BACKEND_IO.override("posix"):
            self.assertEqual(resolve_hf3fs_client_kind(), "usrbio")
        with envs.SGLANG_HICACHE_FILE_BACKEND_IO.override("io_uring"):
            self.assertEqual(resolve_hf3fs_client_kind(), "uring")
            # An explicit client still wins over the env selection.
            self.assertEqual(resolve_hf3fs_client_kind("usrbio"), "usrbio")
        with envs.SGLANG_HICACHE_FILE_BACKEND_IO.override("libaio"):
            with self.assertRaises(ValueError):
                resolve_hf3fs_client_kind()

    def test_factory_builds_uring_client(self):
        for kwargs in ({"client": "uring"}, {}):
            with envs.SGLANG_HICACHE_FILE_BACKEND_IO.override("io_uring"):
                client = create_hf3fs_client(
                    self.path, 4 * PAGE, PAGE, 2, client_timeout=5, **kwargs
                )
            self.addCleanup(client.close)
            self.assertIsInstance(client, UringLocalClient)

    def test_register_mem_pool_host_registers_pinned_pool(self):
        storage = HiCacheHF3FS.__new__(HiCacheHF3FS)
        storage.is_mla_model = False
        storage.clients = [self._client(), self._client()]

        class _HostPool:
            layout = "page_first"
            kv_buffer = torch.empty(4 * PAGE, dtype=torch.uint8)

            def __init__(self, aligned):
                self.aligned = aligned

            def is_stride_page_aligned(self, page_size_bytes=4096):
                return self.aligned

        pool = _HostPool(aligned=True)
        HiCacheHF3FS.register_mem_pool_host(storage, pool)
        self.assertTrue(storage.is_zero_copy)
        self.assertEqual(self.uring.registered, [(pool.kv_buffer.data_ptr(), 4 * PAGE)])
        self.assertTrue(all(c._direct_capable for c in storage.clients))

        HiCacheHF3FS.register_mem_pool_host(storage, _HostPool(aligned=False))
        self.assertFalse(any(c._direct_capable for c in storage.clients))


if __name__ == "__main__":
    unittest.main()
