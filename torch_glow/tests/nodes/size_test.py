from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestSize(unittest.TestCase):
    # Need to be able to export lists from Glow fused nodes
    # Commented out both test cases for not triggering internal CI
    # @unittest.skip(reason="not ready")
    # def test_size_basic(self):
    #    """Test of the PyTorch aten::size Node on Glow."""

    #    def test_f(a):
    #        b = a + a.size(0)
    #        return b

    #    x = torch.zeros([4], dtype=torch.int32)

    #    jitVsGlow(test_f, x, expected_fused_ops={"aten::size"})

    # @unittest.skip(reason="not ready")
    # def test_size_neg_dim(self):
    #    """Test negative dimension index for the PyTorch aten::size Node on Glow."""

    #    def test_f(a):
    #        return a.size(-1)

    #    x = torch.randn(2, 3, 4, dtype=torch.float32)

    #    jitVsGlow(test_f, x, expected_fused_ops={"aten::size"})

    def test_size_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch aten::size Node on Glow."""

        def test_f(a):
            return a.size(-4)

        x = torch.randn(2, 3, 4, dtype=torch.float32)

        with self.assertRaises(IndexError):
            jitVsGlow(test_f, x, expected_fused_ops={"aten::size"})
