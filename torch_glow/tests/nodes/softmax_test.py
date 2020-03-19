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

    def test_softmax_neg_dim(self):
        """Test negative dimension index for the PyTorch SoftMax Node on Glow."""
        def softmax_neg_dim(inputs):
            # Note: dims in the range [-size, -2] cause an assert from flattenCdr() as currently implemented
            return F.softmax(inputs, dim=-1)

        inputs = torch.randn(2, 3)
        jitVsGlow(softmax_neg_dim, inputs,
                  expected_fused_ops={"aten::softmax"})

    def test_softmax_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch SoftMax Node on Glow."""
        def test_f(inputs):
            with self.assertRaises(IndexError):
                return F.softmax(inputs, dim=-3)

        inputs = torch.randn(2, 3)
        with self.assertRaises(RuntimeError):
            jitVsGlow(test_f, inputs, expected_fused_ops={"aten::softmax"})
