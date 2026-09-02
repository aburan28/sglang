"""Async (submit/poll) data plane of the mooncake PD manager -- no engine.

Covers the ``MooncakeTransferEngine`` wrappers over the pybind API, the
``_transfer_data`` submit + poll loop with its sleep backoff (non-blocking
``batch_transfer_poll`` + ``batch_transfer_free`` on newer wheels, blocking
``get_batch_transfer_status`` otherwise), the outstanding counter the
unified-memory move gate reads, and the fallback to synchronous transfers when
the installed wheel has no async API.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import threading
import unittest
from unittest.mock import patch

from sglang.srt.disaggregation.mooncake import conn as conn_mod
from sglang.srt.disaggregation.mooncake.conn import (
    ASYNC_POLL_MAX_SLEEP_S,
    ASYNC_POLL_MIN_SLEEP_S,
    MooncakeKVManager,
)
from sglang.srt.distributed.device_communicators.mooncake_transfer_engine import (
    MooncakeTransferEngine,
)
from sglang.srt.environ import envs
from sglang.test.test_utils import CustomTestCase


class _StatusSequence:
    """Yields the given statuses in order and repeats the last one."""

    def __init__(self, statuses):
        self._statuses = list(statuses)

    def next(self):
        if len(self._statuses) > 1:
            return self._statuses.pop(0)
        return self._statuses[0]


class _FakePybindEngine:
    """Mimics ``mooncake.engine.TransferEngine``: batch id 0 = submit failure;
    status 0 = done, -1 = failed, 1 = still in flight (non-blocking binding)."""

    def __init__(self, statuses=(0,), submit_id=7):
        self.statuses = _StatusSequence(statuses)
        self.submit_id = submit_id
        self.submits = []
        self.polls = []

    def batch_transfer_async_write(self, session, bufs, peers, lens):
        self.submits.append((session, bufs, peers, lens))
        return self.submit_id

    def get_batch_transfer_status(self, batch_ids):
        self.polls.append(list(batch_ids))
        return self.statuses.next()


class _FakePollingPybindEngine(_FakePybindEngine):
    """Newer binding that also exposes the non-blocking poll / free pair."""

    def __init__(self, statuses=(0,), submit_id=7):
        super().__init__(statuses, submit_id)
        self.frees = []

    def batch_transfer_poll(self, batch_ids):
        self.polls.append(list(batch_ids))
        return [self.statuses.next() for _ in batch_ids]

    def batch_transfer_free(self, batch_ids):
        self.frees.append(list(batch_ids))


def _make_transfer_engine(engine):
    te = MooncakeTransferEngine.__new__(MooncakeTransferEngine)
    te.engine = engine
    return te


class TestTransferEngineAsyncWrappers(CustomTestCase):
    def test_capability_probe(self):
        self.assertTrue(
            _make_transfer_engine(_FakePybindEngine()).supports_async_transfer()
        )
        self.assertFalse(_make_transfer_engine(object()).supports_async_transfer())

    def test_submit_returns_batch_id_or_negative_sentinel(self):
        te = _make_transfer_engine(_FakePybindEngine(submit_id=42))
        self.assertEqual(te.batch_transfer_async("s", [1], [2], [3]), 42)
        # The binding reports failure as batch id 0 (unsigned); the wrapper
        # maps it to a negative sentinel so "id > 0" means submitted.
        te = _make_transfer_engine(_FakePybindEngine(submit_id=0))
        self.assertEqual(
            te.batch_transfer_async("s", [1], [2], [3]),
            MooncakeTransferEngine.ASYNC_SUBMIT_FAILED,
        )
        self.assertLess(MooncakeTransferEngine.ASYNC_SUBMIT_FAILED, 0)

    def test_submit_without_async_api_raises(self):
        te = _make_transfer_engine(object())
        with self.assertRaises(RuntimeError):
            te.batch_transfer_async("s", [1], [2], [3])

    def test_status_passthrough_and_exception(self):
        te = _make_transfer_engine(_FakePybindEngine(statuses=(-1,)))
        self.assertEqual(te.get_batch_transfer_status([7]), -1)
        te = _make_transfer_engine(_FakePybindEngine(statuses=(1,)))
        self.assertEqual(te.get_batch_transfer_status([7]), 1)

        class Boom:
            def get_batch_transfer_status(self, ids):
                raise RuntimeError("engine gone")

        self.assertEqual(
            _make_transfer_engine(Boom()).get_batch_transfer_status([7]), -1
        )

    def test_nonblocking_poll_probe(self):
        self.assertTrue(
            _make_transfer_engine(
                _FakePollingPybindEngine()
            ).supports_nonblocking_poll()
        )
        # A wheel with only the blocking status call.
        self.assertFalse(
            _make_transfer_engine(_FakePybindEngine()).supports_nonblocking_poll()
        )

    def test_nonblocking_poll_passthrough_and_exception(self):
        te = _make_transfer_engine(_FakePollingPybindEngine(statuses=(1,)))
        self.assertEqual(te.batch_transfer_poll([7]), [1])
        te = _make_transfer_engine(_FakePollingPybindEngine(statuses=(-1,)))
        self.assertEqual(te.batch_transfer_poll([7]), [-1])

        class Boom:
            def batch_transfer_poll(self, ids):
                raise RuntimeError("engine gone")

            def batch_transfer_free(self, ids):
                raise RuntimeError("engine gone")

        te = _make_transfer_engine(Boom())
        self.assertEqual(te.batch_transfer_poll([7, 8]), [-1, -1])
        te.batch_transfer_free([7, 8])  # swallowed, best effort


class _FakeTransferEngine:
    """What the manager sees (``MooncakeTransferEngine`` shape). Samples the
    manager's outstanding counter inside every poll."""

    def __init__(self, statuses=(0,), submit_id=7, supports=True, nonblocking=False):
        self.statuses = _StatusSequence(statuses)
        self.submit_id = submit_id
        self.supports = supports
        self.nonblocking = nonblocking
        self.submits = []
        self.polls = []
        self.nonblocking_polls = []
        self.frees = []
        self.outstanding_at_free = []
        self.sync_calls = []
        self.observed_outstanding = []
        self.manager = None

    def supports_async_transfer(self):
        return self.supports

    def supports_nonblocking_poll(self):
        return self.nonblocking

    def batch_transfer_async(self, session, srcs, dsts, lens):
        self.submits.append((session, srcs, dsts, lens))
        return self.submit_id

    def get_batch_transfer_status(self, batch_ids):
        self.polls.append(list(batch_ids))
        self.observed_outstanding.append(self.manager.outstanding_async_transfers())
        return self.statuses.next()

    def batch_transfer_poll(self, batch_ids):
        assert self.nonblocking, "non-blocking poll used on a wheel without it"
        self.nonblocking_polls.append(list(batch_ids))
        self.observed_outstanding.append(self.manager.outstanding_async_transfers())
        return [self.statuses.next() for _ in batch_ids]

    def batch_transfer_free(self, batch_ids):
        assert self.nonblocking, "batch_transfer_free used on a wheel without it"
        self.frees.append(list(batch_ids))
        self.outstanding_at_free.append(self.manager.outstanding_async_transfers())

    def batch_transfer_sync(self, session, srcs, dsts, lens):
        self.sync_calls.append((session, srcs, dsts, lens))
        return 0


def _make_manager(engine, enable_async=True):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.engine = engine
    engine.manager = mgr
    mgr.enable_async_transfer = enable_async
    mgr.use_nonblocking_poll = enable_async and engine.supports_nonblocking_poll()
    mgr._async_outstanding = 0
    mgr._async_outstanding_lock = threading.Lock()
    return mgr


BLOCKS = [(1, 2, 3), (4, 5, 6)]


class TestAsyncTransferData(CustomTestCase):
    def test_submit_then_poll_until_complete(self):
        engine = _FakeTransferEngine(statuses=(1, 1, 0))
        mgr = _make_manager(engine)
        with patch.object(conn_mod.time, "sleep") as sleep:
            ret = mgr._transfer_data("sess", BLOCKS)
        self.assertEqual(ret, 0)
        self.assertEqual(engine.submits, [("sess", [1, 4], [2, 5], [3, 6])])
        self.assertEqual(engine.polls, [[7], [7], [7]])
        # The batch counts as outstanding across every poll, and is released
        # once the terminal status is known.
        self.assertEqual(engine.observed_outstanding, [1, 1, 1])
        self.assertEqual(mgr.outstanding_async_transfers(), 0)
        self.assertEqual(sleep.call_count, 2)

    def test_backoff_doubles_from_50us_and_caps_at_2ms(self):
        engine = _FakeTransferEngine(statuses=(1,) * 8 + (0,))
        mgr = _make_manager(engine)
        with patch.object(conn_mod.time, "sleep") as sleep:
            self.assertEqual(mgr._transfer_data("sess", BLOCKS), 0)
        sleeps = [call.args[0] for call in sleep.call_args_list]
        expected = [50e-6, 100e-6, 200e-6, 400e-6, 800e-6, 1.6e-3, 2e-3, 2e-3]
        self.assertEqual(len(sleeps), len(expected))
        for got, want in zip(sleeps, expected):
            self.assertAlmostEqual(got, want, places=9)
        self.assertEqual(sleeps[0], ASYNC_POLL_MIN_SLEEP_S)
        self.assertEqual(max(sleeps), ASYNC_POLL_MAX_SLEEP_S)

    def test_failure_status_propagates_and_releases_counter(self):
        engine = _FakeTransferEngine(statuses=(1, -1))
        mgr = _make_manager(engine)
        with patch.object(conn_mod.time, "sleep"):
            self.assertEqual(mgr._transfer_data("sess", BLOCKS), -1)
        self.assertEqual(engine.observed_outstanding, [1, 1])
        self.assertEqual(mgr.outstanding_async_transfers(), 0)

    def test_submit_failure_is_a_transfer_failure(self):
        engine = _FakeTransferEngine(submit_id=-1)
        mgr = _make_manager(engine)
        self.assertEqual(mgr._transfer_data("sess", BLOCKS), -1)
        self.assertEqual(engine.polls, [])
        self.assertEqual(mgr.outstanding_async_transfers(), 0)

    def test_sync_path_when_disabled(self):
        engine = _FakeTransferEngine()
        mgr = _make_manager(engine, enable_async=False)
        self.assertEqual(mgr._transfer_data("sess", BLOCKS), 0)
        self.assertEqual(engine.sync_calls, [("sess", [1, 4], [2, 5], [3, 6])])
        self.assertEqual(engine.submits, [])

    def test_empty_blocks_touch_nothing(self):
        engine = _FakeTransferEngine()
        mgr = _make_manager(engine)
        self.assertEqual(mgr._transfer_data("sess", []), 0)
        self.assertEqual(engine.submits, [])
        self.assertEqual(engine.sync_calls, [])

    def test_nonblocking_poll_completes_after_several_polls(self):
        engine = _FakeTransferEngine(statuses=(1, 1, 0), nonblocking=True)
        mgr = _make_manager(engine)
        with patch.object(conn_mod.time, "sleep") as sleep:
            self.assertEqual(mgr._transfer_data("sess", BLOCKS), 0)
        self.assertEqual(engine.nonblocking_polls, [[7], [7], [7]])
        self.assertEqual(engine.polls, [])
        # Freed exactly once, after the terminal poll, while still counted as
        # outstanding; released only once the whole transfer is done.
        self.assertEqual(engine.frees, [[7]])
        self.assertEqual(engine.observed_outstanding, [1, 1, 1])
        self.assertEqual(engine.outstanding_at_free, [1])
        self.assertEqual(mgr.outstanding_async_transfers(), 0)
        sleeps = [call.args[0] for call in sleep.call_args_list]
        self.assertEqual(sleeps, [ASYNC_POLL_MIN_SLEEP_S, 2 * ASYNC_POLL_MIN_SLEEP_S])

    def test_nonblocking_poll_failure_frees_once(self):
        engine = _FakeTransferEngine(statuses=(1, -1), nonblocking=True)
        mgr = _make_manager(engine)
        with patch.object(conn_mod.time, "sleep"):
            self.assertEqual(mgr._transfer_data("sess", BLOCKS), -1)
        self.assertEqual(engine.nonblocking_polls, [[7], [7]])
        self.assertEqual(engine.frees, [[7]])
        self.assertEqual(mgr.outstanding_async_transfers(), 0)

    def test_nonblocking_submit_failure_never_polls_or_frees(self):
        engine = _FakeTransferEngine(submit_id=-1, nonblocking=True)
        mgr = _make_manager(engine)
        self.assertEqual(mgr._transfer_data("sess", BLOCKS), -1)
        self.assertEqual(engine.nonblocking_polls, [])
        self.assertEqual(engine.frees, [])

    def test_blocking_status_when_poll_api_missing(self):
        engine = _FakeTransferEngine(statuses=(1, 0), nonblocking=False)
        mgr = _make_manager(engine)
        self.assertFalse(mgr.use_nonblocking_poll)
        with patch.object(conn_mod.time, "sleep"):
            self.assertEqual(mgr._transfer_data("sess", BLOCKS), 0)
        # The blocking call frees the id itself: no explicit free.
        self.assertEqual(engine.polls, [[7], [7]])
        self.assertEqual(engine.nonblocking_polls, [])
        self.assertEqual(engine.frees, [])


class TestResolveAsyncTransfer(CustomTestCase):
    @staticmethod
    def _manager_with(engine):
        mgr = MooncakeKVManager.__new__(MooncakeKVManager)
        mgr.engine = engine
        return mgr

    def test_off_by_default(self):
        with envs.SGLANG_DISAGGREGATION_ASYNC_TRANSFER.override(False):
            mgr = self._manager_with(_FakeTransferEngine())
            self.assertFalse(mgr._resolve_async_transfer())

    def test_on_with_capable_engine(self):
        with envs.SGLANG_DISAGGREGATION_ASYNC_TRANSFER.override(True):
            mgr = self._manager_with(_FakeTransferEngine())
            self.assertTrue(mgr._resolve_async_transfer())

    def test_falls_back_when_wheel_lacks_async_api(self):
        with envs.SGLANG_DISAGGREGATION_ASYNC_TRANSFER.override(True):
            mgr = self._manager_with(_FakeTransferEngine(supports=False))
            with self.assertLogs(conn_mod.logger, level="WARNING") as logs:
                self.assertFalse(mgr._resolve_async_transfer())
            self.assertIn("falling back to synchronous", logs.output[0])


if __name__ == "__main__":
    unittest.main()
