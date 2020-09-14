from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests.utils import jitVsGlow


class TestDropout(unittest.TestCase):
    def test_dropout(self):
        """Basic test of the PyTorch aten::dropout Node on Glow."""

        def test_f(a):
            return F.dropout(a + a, p=0.5, training=False)

        x = torch.randn(6, 4, 10)

        jitVsGlow(test_f, x, expected_fused_ops={"aten::dropout"})

    def test_dropout_inplace(self):
        """Basic test of the PyTorch aten::dropout_ Node on Glow."""

        def test_f(a):
            return F.dropout(a + a, p=0.5, training=False, inplace=True)

        x = torch.randn(6, 4, 10)

        # Expect fuser to out-of-place the operator
        jitVsGlow(test_f, x, expected_fused_ops={"aten::dropout"})
