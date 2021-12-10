# Copyright (c) Glow Contributors. See CONTRIBUTORS file.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import, division, print_function, unicode_literals

import random

import torch
from tests import utils


class SimpleMatmulModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMatmulModule, self).__init__()

    def forward(self, a, b):
        return a.matmul(b + b)


class TestMatMul(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("1d_1d", torch.randn(4), torch.randn(4)),
            lambda: ("1d_2d", torch.randn(4), torch.randn(4, 9)),
            lambda: ("1d_3d", torch.randn(4), torch.randn(3, 4, 9)),
            lambda: ("1d_4d", torch.randn(4), torch.randn(5, 3, 4, 9)),
            lambda: ("2d_1d", torch.randn(9, 4), torch.randn(4)),
            lambda: ("3d_1d", torch.randn(6, 9, 4), torch.randn(4)),
            lambda: ("4d_1d", torch.randn(2, 6, 9, 4), torch.randn(4)),
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
