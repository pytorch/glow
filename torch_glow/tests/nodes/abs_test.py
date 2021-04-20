from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleAbsModule(torch.nn.Module):
    def __init__(self):
        super(SimpleAbsModule, self).__init__()

    def forward(self, a):
        return torch.abs(a + a)


class TestAbs(utils.TorchGlowTestCase):
    def test_abs_basic(self):
        """Basic test of the PyTorch Abs Node on Glow."""

        x = torch.randn(10)
        utils.run_comparison_tests(
            SimpleAbsModule(),
            x,
            fusible_ops={"aten::abs"},
        )

    def test_abs_3d(self):
        """Test multidimensional tensor for the PyTorch Abs Node on Glow."""

        x = torch.randn(2, 3, 5)
        utils.run_comparison_tests(
            SimpleAbsModule(),
            x,
            fusible_ops={"aten::abs"},
        )
