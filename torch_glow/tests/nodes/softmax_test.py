from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.nn.functional as F

from tests.utils import jitVsGlow
import unittest


class TestSoftmax(unittest.TestCase):
    def test_softmax_basic(self):
        """Basic test of the PyTorch SoftMax Node on Glow."""
        def softmax_basic(inputs):
            return F.softmax(inputs, dim=1)

        inputs = torch.randn(2, 3)
        jitVsGlow(softmax_basic, inputs, expected_fused_ops={"aten::softmax"})
