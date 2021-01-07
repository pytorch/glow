from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleClampModel(torch.nn.Module):
    def __init__(self, min, max, inplace=False):
        super(SimpleClampModel, self).__init__()
        self.min = min
        self.max = max
        self.inplace = inplace

    def forward(self, input):
        if self.inplace:
            return input.clamp_(min=self.min, max=self.max)
        else:
            return torch.clamp(input, self.min, self.max)


class TestClamp(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleClampModel(0.0, 0.8), torch.randn(7), {"aten::clamp"}),
            ("no_min", SimpleClampModel(None, 0.8), torch.randn(7), {"aten::clamp"}),
            ("no_max", SimpleClampModel(0.0, None), torch.randn(7), {"aten::clamp"}),
            (
                "inplace",
                SimpleClampModel(0.1, 0.6, inplace=True),
                torch.randn(7),
                {"aten::clamp_"},
            ),
            ("int", SimpleClampModel(0, 2), torch.arange(7), {"aten::clamp"}),
            (
                "int_inplace",
                SimpleClampModel(0, 2, inplace=True),
                torch.arange(7),
                {"aten::clamp_"},
            ),
            (
                "int_no_min",
                SimpleClampModel(None, 1),
                torch.arange(7),
                {"aten::clamp"},
            ),
            (
                "int_no_max",
                SimpleClampModel(0, None),
                torch.arange(7),
                {"aten::clamp"},
            ),
        ]
    )
    def test_clamp(self, _, model, tensor, fusible):
        """Test of the PyTorch clamp Node on Glow."""

        utils.compare_tracing_methods(model, tensor, fusible_ops=fusible)
