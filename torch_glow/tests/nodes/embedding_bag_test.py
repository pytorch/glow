from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests.utils import jitVsGlow
import unittest


class TestEmbeddingBag(unittest.TestCase):
    def test_embedding_bag_basic(self):
        """Test of aten::embedding_bag node on glow"""

        def embedding_bag_basic(input, offsets, per_sample_weights):
            weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]])
            embedding_sum = torch.nn.EmbeddingBag.from_pretrained(
                weight, mode="sum")
            # in jit mode we need to discard the end offset
            a = embedding_sum(input, offsets[:-1])
            b = embedding_sum(input, offsets[:-1], per_sample_weights)
            return a, b

        input = torch.LongTensor([1, 0, 0, 1, 1])
        offsets = torch.LongTensor([0, 1, 5])  # final item is endOffset
        per_sample_weights = torch.FloatTensor([1, 2, 3, 4, 5])

        jitVsGlow(
            embedding_bag_basic,
            input,
            offsets,
            per_sample_weights,
            expected_fused_ops={"aten::embedding_bag"},
        )
