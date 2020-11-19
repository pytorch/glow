from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimplePreluModule(torch.nn.Module):
    def __init__(self):
        super(SimplePreluModule, self).__init__()

    def forward(self, inputs, weights):
        return F.prelu(inputs + inputs, weights)


class TestPrelu(unittest.TestCase):
    def test_prelu_basic(self):
        """Basic test of the PyTorch prelu Node on Glow."""

        utils.compare_tracing_methods(
            SimplePreluModule(),
            torch.randn(1, 4, 5, 5),
            torch.tensor([0.25]),
            fusible_ops={"aten::prelu"},
        )
