from __future__ import absolute_import, division, print_function, unicode_literals

import random
import unittest

import torch
from tests.utils import jitVsGlow


class TestMatMul(unittest.TestCase):
    def test_matmul_1d_1d(self):
        """Test of aten::matmul with two 1d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_2d_1d(self):
        """Test of aten::matmul with 2d and 1d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(9, 4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_3d_1d(self):
        """Test of aten::matmul with 2d and 1d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(6, 9, 4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_4d_1d(self):
        """Test of aten::matmul with 2d and 1d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(2, 6, 9, 4)
        y = torch.randn(4)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_1d_2d(self):
        """Test of aten::matmul with 1d and 2d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(4)
        y = torch.randn(4, 9)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_1d_3d(self):
        """Test of aten::matmul with 1d and 2d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(4)
        y = torch.randn(3, 4, 9)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_1d_4d(self):
        """Test of aten::matmul with 1d and 2d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        x = torch.randn(4)
        y = torch.randn(5, 3, 4, 9)

        jitVsGlow(test_f, x, y, expected_fused_ops={"aten::matmul"})

    def test_matmul_nd_nd(self):
        """Test of aten::matmul with >2d and >2d inputs Glow."""

        def test_f(a, b):
            return a.matmul(b + b)

        def do_test(lhsDims, rhsDims):
            lhs = torch.randn(lhsDims)
            rhs = torch.randn(rhsDims)

            jitVsGlow(test_f, lhs, rhs, expected_fused_ops={"aten::matmul"})

        def randomDimsOfRank(rank):
            dims = []
            for i in range(rank):
                dim = random.randint(2, 9)
                dims.append(dim)
            return dims

        # Dimensions of base tensors that lhs and rhs will be built from
        lhsBase = [3, 4]
        rhsBase = [4, 2]

        for additional_dims in range(3):
            extension = randomDimsOfRank(additional_dims)

            do_test(extension + lhsBase, rhsBase)
            do_test([1] + extension + lhsBase, rhsBase)
            do_test(extension + [1] + lhsBase, rhsBase)

            do_test(lhsBase, extension + rhsBase)
            do_test(lhsBase, [1] + extension + rhsBase)
            do_test(lhsBase, extension + [1] + rhsBase)

            do_test(extension + lhsBase, extension + rhsBase)
