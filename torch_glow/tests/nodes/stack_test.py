from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleStackModel(torch.nn.Module):
    def __init__(self, *args):
        super(SimpleStackModel, self).__init__()
        self.axis = args[0] if args else None

    def forward(self, a, b):
        if not self.axis:
            c = torch.stack((a, b), 0)
            d = torch.stack((c, c), 1)
            return torch.stack((d, d), 2)
        else:
            a = a + a
            return torch.stack((a, b), dim=self.axis)


class TestStack(unittest.TestCase):
    def test_stack_basic(self):
        """Basic test of the PyTorch aten::stack Node on Glow."""

        utils.compare_tracing_methods(
            SimpleStackModel(),
            torch.randn(2, 3, 4),
            torch.randn(2, 3, 4),
            skip_to_glow=True,
        )

    def test_stack_different_types(self):
        """Test stack between fp16 and fp32, which is supported in pytorch."""

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4, dtype=torch.half)

        utils.compare_tracing_methods(
            SimpleStackModel(),
            x,
            y,
            skip_to_glow=True,
        )

    def test_stack_on_new_dim(self):
        x = torch.randn(4)
        y = torch.randn(4)

        utils.compare_tracing_methods(
            SimpleStackModel(1),
            x,
            y,
            skip_to_glow=True,
        )
