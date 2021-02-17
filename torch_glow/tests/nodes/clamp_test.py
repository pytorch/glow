from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleClampModel(torch.nn.Module):
    def __init__(self, min, max):
        super(SimpleClampModel, self).__init__()
        self.min = min
        self.max = max

    def forward(self, input):
        return torch.clamp(input, self.min, self.max)


class TestClamp(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", 0.0, 0.8),
            lambda: ("no_min", None, 0.8),
            lambda: ("no_max", 0.0, None),
        ]
    )
    def test_clamp(self, _, min, max):
        """Test of the PyTorch clamp Node on Glow."""

        utils.compare_tracing_methods(
            SimpleClampModel(min, max), torch.randn(7), fusible_ops={"aten::clamp"}
        )
