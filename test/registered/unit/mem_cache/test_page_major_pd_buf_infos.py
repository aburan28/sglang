"""PD transfer addressing of the page-major MHA pool (CPU, no server).

``PageMajorMHATokenToKVPool.get_contiguous_buf_infos`` exposes the single
``_raw`` buffer as ONE region whose ``item_len`` is the byte size of a page
across all layers and K/V, and mooncake's ``_send_kvcache_generic`` then
addresses ``ptr + page_index * item_len``. These tests pin that contract to the
strided views the kernels read through: page ``p`` of every layer's K/V view
must live inside ``[ptr + p * item_len, ptr + (p + 1) * item_len)`` at the
layer-major offsets, and nothing of page ``p`` may live outside it.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

import unittest

import torch

from sglang.srt.mem_cache.layout.page_major import mha_entry_bytes
from sglang.srt.mem_cache.memory_pool import (
    MHATokenToKVPool,
    PageMajorMHATokenToKVPool,
)
from sglang.test.test_utils import CustomTestCase

_DT = torch.float32


def _make_pool(layer_num=3, head_num=2, head_dim=8, v_head_dim=4, page_size=4, size=12):
    return PageMajorMHATokenToKVPool(
        size=size,
        page_size=page_size,
        dtype=_DT,
        head_num=head_num,
        head_dim=head_dim,
        layer_num=layer_num,
        device="cpu",
        enable_memory_saver=False,
        v_head_dim=v_head_dim,
        enable_alt_stream=False,
    )


class TestPageMajorContiguousBufInfos(CustomTestCase):
    def test_single_region_with_page_item_len(self):
        pool = _make_pool()
        ptrs, lens, item_lens = pool.get_contiguous_buf_infos()
        self.assertEqual((len(ptrs), len(lens), len(item_lens)), (1, 1, 1))
        entry = mha_entry_bytes(
            layer_num=3, head_num=2, head_dim=8, v_head_dim=4, itemsize=_DT.itemsize
        )
        self.assertEqual(item_lens[0], pool.page_size * entry)
        self.assertEqual(ptrs[0], pool._raw.data_ptr())
        self.assertEqual(lens[0], pool._raw.numel())
        self.assertEqual(lens[0], pool.num_pages * item_lens[0])
        # (size + page_size) slots -> one padding page on top of the 3 data pages.
        self.assertEqual(pool.num_pages, 4)
        # The flag rides on KVArgs so both PD peers agree on the layout.
        self.assertTrue(pool.kv_layout_page_major)
        self.assertFalse(MHATokenToKVPool.kv_layout_page_major)

    def test_page_address_matches_strided_views(self):
        """``ptr + p * item_len`` is where page p's envelope starts (layer-0 K),
        and each layer's K then V block of page p follows layer-major inside
        that envelope, ending exactly at the next page."""
        pool = _make_pool()
        (ptr,), _, (item_len,) = pool.get_contiguous_buf_infos()
        itemsize = _DT.itemsize
        k_row = pool.head_num * pool.head_dim * itemsize
        v_row = pool.head_num * pool.v_head_dim * itemsize
        ps = pool.page_size
        for p in range(pool.num_pages):
            page_base = ptr + p * item_len
            self.assertEqual(pool.k_buffer[0][p].data_ptr(), page_base)
            for layer in range(pool.layer_num):
                layer_base = page_base + layer * ps * (k_row + v_row)
                self.assertEqual(pool.k_buffer[layer][p].data_ptr(), layer_base)
                self.assertEqual(
                    pool.v_buffer[layer][p].data_ptr(), layer_base + ps * k_row
                )
                self.assertEqual(
                    pool.v_buffer[layer][p, ps - 1].data_ptr() + v_row,
                    layer_base + ps * (k_row + v_row),
                )
            self.assertEqual(
                pool.v_buffer[-1][p, ps - 1].data_ptr() + v_row, page_base + item_len
            )

    def test_page_writes_stay_inside_their_envelope(self):
        """Writing page p through every layer's views changes exactly the raw
        slice ``[p * item_len, (p + 1) * item_len)`` and fills all of it, so
        one descriptor of ``item_len`` bytes carries the whole page."""
        pool = _make_pool()
        _, _, (item_len,) = pool.get_contiguous_buf_infos()
        raw = pool._raw
        torch.manual_seed(0)
        for p in range(pool.num_pages):
            raw.zero_()
            for layer in range(pool.layer_num):
                k_view, v_view = pool.k_buffer[layer][p], pool.v_buffer[layer][p]
                k_view.copy_(torch.rand(k_view.shape, dtype=_DT) + 1.0)
                v_view.copy_(torch.rand(v_view.shape, dtype=_DT) + 1.0)
            envelope = raw[p * item_len : (p + 1) * item_len].view(_DT)
            outside = torch.cat([raw[: p * item_len], raw[(p + 1) * item_len :]])
            self.assertTrue(bool(outside.eq(0).all()), f"page {p} bled outside")
            self.assertTrue(bool(envelope.ne(0).all()), f"page {p} not filled")


if __name__ == "__main__":
    unittest.main()
