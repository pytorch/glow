from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests.utils import jitVsGlow


class TestRelu(unittest.TestCase):
    def test_relu_basic(self):
        """Basic test of the PyTorch relu Node on Glow."""

        def test_f(a):
            b = F.relu(a)
            return F.relu(b)

        x = torch.randn(4)
        # make sure we have at least one negative
        x[0] = -2.0

        jitVsGlow(test_f, x, expected_fused_ops={"aten::relu"})

    def test_relu_inplace(self):
        """Test of the PyTorch relu_ Node on Glow."""

        def test_f(a):
            b = F.relu(a, inplace=True)
            return F.relu(b, inplace=True)

        x = torch.randn(4)
        # make sure we have at least one negative
        x[0] = -2.0

        jitVsGlow(test_f, x, expected_fused_ops={"aten::relu_"})
