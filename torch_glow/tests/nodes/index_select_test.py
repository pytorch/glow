import unittest

import torch
from parameterized import parameterized
from tests import utils


class IndexSelectModule(torch.nn.Module):
    def __init__(self, dimension):
        super(IndexSelectModule, self).__init__()
        self.dimension = dimension

    def forward(self, tensor, index):
        return torch.index_select(tensor, self.dimension, index)


class TestIndexSelect(unittest.TestCase):
    @parameterized.expand(
        [
            ("0-dim", torch.randn(3, 4), 0, torch.tensor([0, 2])),
            ("1-dim", torch.randn(3, 4), 1, torch.tensor([0, 2])),
            ("repeat index", torch.randn(3, 4), 1, torch.tensor([2, 2])),
        ]
    )
    def test_index_select(self, _, tensor, dimension, index):
        utils.compare_tracing_methods(
            IndexSelectModule(dimension),
            tensor,
            index,
            skip_to_glow=True,
            fusible_ops={"aten::index_select"},
        )
