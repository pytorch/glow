from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleClampMinModel(torch.nn.Module):
    def __init__(self, min):
        super(SimpleClampMinModel, self).__init__()
        self.min = min

    def forward(self, input):
        return torch.clamp_min(input, self.min)


class TestClamp(unittest.TestCase):
    def test_clamp_min(self):
        """Test of the PyTorch clamp_min Node on Glow."""

        utils.compare_tracing_methods(
            SimpleClampMinModel(0.1), torch.randn(7), fusible_ops={"aten::clamp_min"}
        )
