from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleErfModule(torch.nn.Module):
    def forward(self, input):
        return torch.special.erf(input)


class TestErf(utils.TorchGlowTestCase):
    def test_erf_basic(self):
        """Test of the PyTorch erf Node on Glow."""

        utils.compare_tracing_methods(
            SimpleErfModule(), torch.randn(4)
        )
