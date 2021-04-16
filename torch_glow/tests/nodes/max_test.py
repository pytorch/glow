from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleMaxModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMaxModule, self).__init__()

    def forward(self, a, b):
        return torch.max(a + a, b + b)


class UnaryMaxModule(torch.nn.Module):
    def __init__(self):
        super(UnaryMaxModule, self).__init__()

    def forward(self, a):
        return torch.max(a + a)


class TestMax(utils.TorchGlowTestCase):
    def test_elementwise_max(self):
        """Test of the PyTorch max Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxModule(), torch.randn(4), torch.randn(4), fusible_ops={"aten::max"}
        )

    def test_elementwise_max_broadcast(self):
        """Test of the PyTorch max Node with broadcast on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxModule(),
            torch.randn(2, 4),
            torch.randn(4),
            fusible_ops={"aten::max"},
        )

    def test_unary_max(self):
        """Test of the PyTorch unary max Node on Glow."""

        utils.compare_tracing_methods(
            UnaryMaxModule(),
            torch.randint(
                50,
                (
                    10,
                    10,
                ),
                dtype=torch.int,
            ),
            fusible_ops={"aten::max"},
        )
