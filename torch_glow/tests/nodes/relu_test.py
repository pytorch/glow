from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleReluModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleReluModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        other = F.relu(tensor, inplace=self.inplace)
        return F.relu(other, inplace=self.inplace)


class TestRelu(unittest.TestCase):
    def test_relu_basic(self):
        """Basic test of the PyTorch relu Node on Glow."""

        x = torch.randn(4)
        # make sure we have at least one negative
        x[0] = -2.0

        utils.compare_tracing_methods(SimpleReluModel(), x, fusible_ops={"aten::relu"})

    def test_relu_inplace(self):
        """Test of the PyTorch relu_ Node on Glow."""

        x = torch.randn(4)
        # make sure we have at least one negative
        x[0] = -2.0

        utils.compare_tracing_methods(
            SimpleReluModel(inplace=True), x, fusible_ops={"aten::relu_"}
        )
