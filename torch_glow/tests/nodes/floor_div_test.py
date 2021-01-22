from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleFloorDivideModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleFloorDivideModule, self).__init__()
        self.inplace = inplace

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            b = b.item()
        if self.inplace:
            return (a + a).floor_divide_(b)
        else:
            return (a + a).floor_divide(b)


class TestFloorDiv(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleFloorDivideModule(),
                torch.Tensor(4).random_(0, 5),
                torch.Tensor(4).random_(1, 5),
            ),
            (
                "inplace",
                SimpleFloorDivideModule(True),
                torch.Tensor(4).random_(0, 5),
                torch.Tensor(4).random_(1, 5),
            ),
            (
                "positive_float",
                SimpleFloorDivideModule(),
                torch.Tensor(4).random_(0, 5),
                torch.tensor(3.9),
            ),
            (
                "negative_float",
                SimpleFloorDivideModule(),
                torch.tensor([-4.0]),
                torch.tensor([3.0]),
            ),
            (
                "positive_broadcast",
                SimpleFloorDivideModule(),
                torch.Tensor(8, 3, 4, 2).random_(0, 5),
                torch.Tensor(4, 2).random_(1, 5),
            ),
            (
                "positive_broadcast",
                SimpleFloorDivideModule(),
                torch.Tensor(8, 3, 4, 2).random_(0, 5),
                torch.Tensor(1, 2).random_(1, 5),
            ),
            (
                "positive_broadcast",
                SimpleFloorDivideModule(),
                torch.Tensor(4, 2).random_(0, 5),
                torch.Tensor(8, 3, 4, 2).random_(1, 5),
            ),
            (
                "positive_int",
                SimpleFloorDivideModule(),
                torch.tensor([5]),
                torch.tensor([4]),
            ),
            (
                "negative_int",
                SimpleFloorDivideModule(),
                torch.tensor([-5]),
                torch.tensor([4]),
            ),
        ]
    )
    def test_floor_div(self, _, module, left, right):
        utils.compare_tracing_methods(
            module, left, right, fusible_ops={"aten::floor_divide"}
        )
