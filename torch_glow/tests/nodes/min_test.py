from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleMinModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMinModule, self).__init__()

    def forward(self, a, b):
        return torch.min(a + a, b + b)


class UnaryMinModule(torch.nn.Module):
    def __init__(self):
        super(UnaryMinModule, self).__init__()

    def forward(self, a):
        return torch.min(a + a)


class TestMin(utils.TorchGlowTestCase):
    def test_elementwise_min(self):
        """Test of the PyTorch min Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMinModule(), torch.randn(7), torch.randn(7), fusible_ops={"aten::min"}
        )

    def test_elementwise_min_broadcast(self):
        """Test of the PyTorch min Node with broadcast on Glow."""

        utils.compare_tracing_methods(
            SimpleMinModule(),
            torch.randn(2, 7),
            torch.randn(7),
            fusible_ops={"aten::min"},
        )

    def test_unary_min(self):
        """Test of the PyTorch unary min Node on Glow."""

        utils.compare_tracing_methods(
            UnaryMinModule(),
            torch.randint(
                20,
                (
                    10,
                    10,
                ),
                dtype=torch.int32,
            ),
            fusible_ops={"aten::min"},
        )
