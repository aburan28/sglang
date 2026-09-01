"""Mooncake PD transfer of the page-major (whole-envelope) KV layout.

With ``kv_layout_page_major`` the prefill registers ONE region whose
``item_len`` is a full page across all layers and K/V, so ``send_kvcache``
must take the flat branch of ``_send_kvcache_generic`` (one descriptor per run
of contiguous pages, addressed ``ptr + page * item_len``) and refuse a peer
that registered a different shape. No server / engine: the manager is built
bare and the engine call is captured.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

import numpy as np

from sglang.srt.disaggregation.mooncake.conn import MooncakeKVManager
from sglang.srt.runtime_context import get_context
from sglang.test.test_utils import CustomTestCase

SESSION = "10.0.0.2:12345"
ITEM_LEN = 4096
PREFILL_PAGES = np.array([2, 3, 4, 7], dtype=np.int32)
DECODE_PAGES = np.array([5, 6, 7, 1], dtype=np.int32)


def _make_manager(src_ptrs, item_lens, page_major):
    mgr = MooncakeKVManager.__new__(MooncakeKVManager)
    mgr.is_mla_backend = False
    mgr.is_hybrid_mla_backend = False
    mgr.is_page_major_kv = page_major
    mgr.pp_size = 1
    mgr.attn_tp_size = 1
    mgr.enable_custom_mem_pool = False
    mgr.enable_deferred_decode_kv_release = False
    mgr.kv_args = SimpleNamespace(
        kv_data_ptrs=list(src_ptrs),
        kv_item_lens=list(item_lens),
        kv_layer_ids=[],
        prefill_start_layer=0,
        prefill_end_layer=None,
        mla_compression_ratios=None,
    )
    mgr.sent = []

    def capture(session, blocks):
        mgr.sent.append((session, list(blocks)))
        return 0

    mgr._transfer_data = capture
    return mgr


class TestPageMajorSendKVCache(CustomTestCase):
    def test_contiguous_pages_become_one_descriptor(self):
        src, dst = 0x10000, 0x80000
        mgr = _make_manager([src], [ITEM_LEN], page_major=True)
        ret = mgr.send_kvcache(
            SESSION,
            PREFILL_PAGES,
            [dst],
            DECODE_PAGES,
            executor=None,
            dst_kv_item_len=ITEM_LEN,
            dst_attn_tp_size=1,
        )
        self.assertEqual(ret, 0)
        # Runs [2,3,4]->[5,6,7] and [7]->[1]; each run is one whole-page block.
        self.assertEqual(
            mgr.sent,
            [
                (
                    SESSION,
                    [
                        (src + 2 * ITEM_LEN, dst + 5 * ITEM_LEN, 3 * ITEM_LEN),
                        (src + 7 * ITEM_LEN, dst + 1 * ITEM_LEN, ITEM_LEN),
                    ],
                )
            ],
        )

    def test_rejects_peer_on_per_layer_layout(self):
        mgr = _make_manager([0x10000], [ITEM_LEN], page_major=True)
        with self.assertRaisesRegex(RuntimeError, "KV layout mismatch"):
            mgr.send_kvcache(
                SESSION,
                PREFILL_PAGES,
                [0x80000, 0x90000],
                DECODE_PAGES,
                executor=None,
                dst_kv_item_len=ITEM_LEN,
            )
        self.assertEqual(mgr.sent, [])

    def test_rejects_peer_with_other_page_bytes(self):
        mgr = _make_manager([0x10000], [ITEM_LEN], page_major=True)
        with self.assertRaisesRegex(RuntimeError, "KV layout mismatch"):
            mgr.send_kvcache(
                SESSION,
                PREFILL_PAGES,
                [0x80000],
                DECODE_PAGES,
                executor=None,
                dst_kv_item_len=2 * ITEM_LEN,
            )

    def test_rejects_heterogeneous_attention_tp(self):
        mgr = _make_manager([0x10000], [ITEM_LEN], page_major=True)
        with self.assertRaisesRegex(RuntimeError, "attention TP"):
            mgr.send_kvcache(
                SESSION,
                PREFILL_PAGES,
                [0x80000],
                DECODE_PAGES,
                executor=None,
                dst_kv_item_len=ITEM_LEN,
                dst_attn_tp_size=2,
            )

    def test_per_layer_layout_keeps_k_v_split(self):
        """Without the flag the MHA branch stays: [K0, K1, V0, V1] regions are
        paired per layer, so the same pages give one block pair per region."""
        override = get_context().override_server_args()
        override.install()
        self.addCleanup(override.restore)
        src = [0x10000, 0x20000, 0x30000, 0x40000]
        dst = [0x80000, 0x90000, 0xA0000, 0xB0000]
        mgr = _make_manager(src, [ITEM_LEN] * 4, page_major=False)
        ret = mgr.send_kvcache(SESSION, PREFILL_PAGES, dst, DECODE_PAGES, executor=None)
        self.assertEqual(ret, 0)
        expected = []
        for s, d in zip(src, dst):
            expected.append((s + 2 * ITEM_LEN, d + 5 * ITEM_LEN, 3 * ITEM_LEN))
            expected.append((s + 7 * ITEM_LEN, d + 1 * ITEM_LEN, ITEM_LEN))
        self.assertEqual(mgr.sent, [(SESSION, expected)])


if __name__ == "__main__":
    unittest.main()
