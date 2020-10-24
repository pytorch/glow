from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleGatherModule(torch.nn.Module):
    def __init__(self, axis):
        super(SimpleGatherModule, self).__init__()
        self.axis = axis

    def forward(self, data, indices):
        return torch.gather(data + data, self.axis, indices)


class TestGather(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic_1d",
                SimpleGatherModule(axis=0),
                torch.tensor([1, 2, 3, 4]),
                torch.tensor([0, 0, 1, 0], dtype=torch.long),
            ),
            (
                "2d_axis_eq_0",
                SimpleGatherModule(axis=0),
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([[0, 0], [1, 0]], dtype=torch.long),
            ),
            (
                "2d_axis_eq_1",
                SimpleGatherModule(axis=1),
                torch.tensor([[1, 2], [3, 4]]),
                torch.tensor([[1, 1], [1, 0]], dtype=torch.long),
            ),
            (
                "3d_axis_eq_2",
                SimpleGatherModule(axis=2),
                torch.reshape(torch.tensor(list(range(30))), (2, 3, 5)),
                torch.empty(2, 3, 3).random_(2).long(),
            ),
            (
                "4d_axis_eq_neg_2",
                SimpleGatherModule(axis=-2),
                torch.reshape(torch.tensor(list(range(120))), (2, 3, 4, 5)),
                torch.empty(2, 3, 4, 3).random_(2).long(),
            ),
        ]
    )
    def test_gather(self, _, module, tensor, indices):
        """Test of the PyTorch gather Node on Glow."""
        utils.compare_tracing_methods(
            module, tensor, indices, fusible_ops={"aten::gather"}
        )
