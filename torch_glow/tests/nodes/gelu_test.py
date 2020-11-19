from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleGeluModule(torch.nn.Module):
    def forward(self, tensor):
        return F.gelu(tensor + tensor)


class TestGelu(unittest.TestCase):
    def test_gelu_basic(self):
        """Basic test of the PyTorch gelu Node on Glow."""

        def test_f(a):
            return F.gelu(a + a)

        for _ in range(100):
            x = torch.randn(10)
            utils.compare_tracing_methods(
                SimpleGeluModule(),
                x,
                check_trace=False,
                atol=1e-3,
                fusible_ops={"aten::gelu"},
            )
