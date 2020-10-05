from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestTopk(unittest.TestCase):
    def test_topk_basic(self):
        """Test of the PyTorch TopK Node on Glow."""

        def test_topk(x):
            x + x
            return torch.topk(x, 3)

        x = torch.arange(1.0, 6.0)

        jitVsGlow(test_topk, x, expected_fused_ops={"aten::topk"})
