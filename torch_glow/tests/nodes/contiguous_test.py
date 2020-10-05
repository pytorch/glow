from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestContiguous(unittest.TestCase):
    def test_contiguous_basic(self):
        """Test of the PyTorch contiguous Node on Glow."""

        def contiguous_basic(a):
            return a.contiguous()

        x = torch.randn(2, 2, 2)

        jitVsGlow(contiguous_basic, x, expected_fused_ops={"aten::contiguous"})
