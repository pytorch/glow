from __future__ import absolute_import, division, print_function, unicode_literals

import random
import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleMatmulModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMatmulModule, self).__init__()

    def forward(self, a, b):
        return a.matmul(b + b)


class TestMatMul(unittest.TestCase):
    @parameterized.expand(
        [
            ("1d_1d", torch.randn(4), torch.randn(4)),
            ("1d_2d", torch.randn(4), torch.randn(4, 9)),
            ("1d_3d", torch.randn(4), torch.randn(3, 4, 9)),
            ("1d_4d", torch.randn(4), torch.randn(5, 3, 4, 9)),
            ("2d_1d", torch.randn(9, 4), torch.randn(4)),
            ("3d_1d", torch.randn(6, 9, 4), torch.randn(4)),
            ("4d_1d", torch.randn(2, 6, 9, 4), torch.randn(4)),
        ]
    )
    def test_matmul(self, _, left, right):
        """Test of aten::matmul with two 1d inputs Glow."""

        utils.compare_tracing_methods(
            SimpleMatmulModule(), left, right, fusible_ops={"aten::matmul"}
        )

    def test_matmul_nd_nd(self):
        """Test of aten::matmul with >2d and >2d inputs Glow."""

        def do_test(lhsDims, rhsDims):
            lhs = torch.randn(lhsDims)
            rhs = torch.randn(rhsDims)

            utils.compare_tracing_methods(
                SimpleMatmulModule(), lhs, rhs, fusible_ops={"aten::matmul"}
            )

        def randomDimsOfRank(rank):
            dims = []
            for _ in range(rank):
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
