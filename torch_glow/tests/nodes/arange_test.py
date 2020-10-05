from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests.utils import jitVsGlow


class TestArange(unittest.TestCase):
    """
    Tests for torch.arange glow fusion.

    Note that torch.arange is effectively a constant, so torch jit will try to
    compile it down to said constant. The tests in this class utilize a test
    function which takes a tensor as input, so that we can prevent that from
    happening. Otherwise, there would be nothing to fuse.
    """

    def test_arange_simple(self):
        """Testing arange with minimum parameters"""

        def test_f(dummy):
            return torch.arange(dummy.size(0))

        dummy = torch.randn(10)
        jitVsGlow(test_f, dummy, expected_fused_ops={"aten::arange"})

    def test_arange_multiple_args(self):
        """Testing arange with multiple parameters"""

        def test_f(dummy):
            return torch.arange(dummy.size(0), end=30)

        dummy = torch.randn(10)
        jitVsGlow(test_f, dummy, expected_fused_ops={"aten::arange"})

    def test_arange_with_all_args(self):
        """Testing arange with all args provided"""

        def test_f(dummy):
            return torch.arange(dummy.size(0), 30, 1)

        dummy = torch.randn(10)
        jitVsGlow(test_f, dummy, expected_fused_ops={"aten::arange"})

    def test_arange_with_floats(self):
        """Testing arange with all floats as input"""

        def test_f(dummy):
            return torch.arange(dummy.size(0), 30.5, 0.8)

        dummy = torch.randn(10)
        jitVsGlow(test_f, dummy, expected_fused_ops={"aten::arange"})

    def test_arange_with_negative_step(self):
        """Testing arange with negative step"""

        def test_f(dummy):
            return torch.arange(dummy.size(0), dummy.size(1), -1.2)

        dummy = torch.randn(10, 2)
        jitVsGlow(test_f, dummy, expected_fused_ops={"aten::arange"})
