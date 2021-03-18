from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils
from tests.utils import check_skip


class TestEmbedding(unittest.TestCase):
    supported_backends = {"Interpreter", "NNPI"}

    def test_embedding_wt_float32_ind_int64(self):
        """Test of aten::embedding node on glow"""

        check_skip(self)

        class TestModule(torch.nn.Module):
            def forward(self, indices):
                weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3], [3.3, -4, 5.7]])
                embedding = torch.nn.Embedding.from_pretrained(weight)
                a = embedding(indices)
                return a

        indices = torch.LongTensor([1, 0, 2, 0, 1])

        utils.compare_tracing_methods(
            TestModule(),
            indices,
            fusible_ops={"aten::embedding"},
            skip_to_glow=True,  # to_glow doesn't support include_last_offset=False
        )

    def test_embedding_wt_float16_ind_int64(self):
        """Test of aten::embedding node on glow"""

        check_skip(self)

        class TestModule(torch.nn.Module):
            def forward(self, indices):
                weight = torch.HalfTensor([[1, 2.3, 3], [4, 5.1, 6.3], [3.3, -4, 5.7]])
                embedding = torch.nn.Embedding.from_pretrained(weight)
                a = embedding(indices)
                return a

        indices = torch.LongTensor([1, 0, 2, 0, 1])

        utils.compare_tracing_methods(
            TestModule(),
            indices,
            fusible_ops={"aten::embedding"},
            skip_to_glow=True,  # to_glow doesn't support include_last_offset=False
        )

    def test_embedding_2d_indices(self):
        """Test of aten::embedding node on glow"""

        check_skip(self)

        class TestModule(torch.nn.Module):
            def forward(self, indices):
                weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3], [3.3, -4, 5.7]])
                embedding = torch.nn.Embedding.from_pretrained(weight)
                a = embedding(indices)
                return a

        indices = torch.LongTensor([[1, 2], [0, 1]])

        utils.compare_tracing_methods(
            TestModule(),
            indices,
            fusible_ops={"aten::embedding"},
            skip_to_glow=True,  # to_glow doesn't support include_last_offset=False
        )

    def test_embedding_3d_indices(self):
        """Test of aten::embedding node on glow"""

        check_skip(self)

        class TestModule(torch.nn.Module):
            def forward(self, indices):
                weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3], [3.3, -4, 5.7]])
                embedding = torch.nn.Embedding.from_pretrained(weight)
                a = embedding(indices)
                return a

        indices = torch.LongTensor([[[1, 2], [0, 1]]])

        utils.compare_tracing_methods(
            TestModule(),
            indices,
            fusible_ops={"aten::embedding"},
            skip_to_glow=True,  # to_glow doesn't support include_last_offset=False
        )
