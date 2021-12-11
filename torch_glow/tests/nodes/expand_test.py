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

import torch
from tests import utils


class ExpandModel(torch.nn.Module):
    def __init__(self, shape):
        super(ExpandModel, self).__init__()
        self.shape = shape

    def forward(self, a):
        return a.expand(self.shape)


class TestExpand(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "unit_vector",
                ExpandModel([3]),
                torch.randn(1),
            ),
            lambda: (
                "unit_matrix",
                ExpandModel([3, 4]),
                torch.randn(1, 1),
            ),
            lambda: (
                "singleton_matrix",
                ExpandModel([2, 4]),
                torch.randn(2, 1),
            ),
            lambda: (
                "singleton_matrix_minus_one",
                ExpandModel([-1, 4]),
                torch.randn(2, 1),
            ),
            lambda: (
                "fourD",
                ExpandModel([2, 4, 5, 8]),
                torch.randn(2, 1, 5, 8),
            ),
            lambda: (
                "fourD_two_singleton",
                ExpandModel([2, 4, 5, 8]),
                torch.randn(2, 1, 5, 1),
            ),
            lambda: (
                "fourD_minus_ones",
                ExpandModel([2, 4, -1, -1]),
                torch.randn(2, 1, 5, 8),
            ),
            lambda: (
                "add_dim",
                ExpandModel([3, 4, 2]),
                torch.randn(4, 2),
            ),
            lambda: (
                "add_two_dims",
                ExpandModel([8, 3, 4, 2]),
                torch.randn(4, 2),
            ),
            lambda: (
                "add_dim_minus_one",
                ExpandModel([3, -1, 2]),
                torch.randn(4, 2),
            ),
            lambda: (
                "add_dim_minus_ones",
                ExpandModel([3, -1, -1]),
                torch.randn(4, 2),
            ),
            lambda: (
                "add_dims_minus_one",
                ExpandModel([8, 3, -1, 2]),
                torch.randn(4, 2),
            ),
            lambda: (
                "add_dims_minus_ones",
                ExpandModel([8, 3, -1, -1]),
                torch.randn(4, 2),
            ),
        ]
    )
    def test_expand(self, _, module, a):
        """Test of the PyTorch expand Node on Glow."""
        utils.compare_tracing_methods(
            module,
            a,
            fusible_ops={"aten::expand"},
        )


class TestExpandError(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "no_singleton",
                ExpandModel([3, 3]),
                torch.randn(2, 2),
            ),
            lambda: (
                "shape_too_small",
                ExpandModel([3]),
                torch.randn(2, 2),
            ),
            lambda: (
                "invalid_zero",
                ExpandModel([0, 3]),
                torch.randn(1, 2),
            ),
            lambda: (
                "invalid_negative",
                ExpandModel([-2, 3]),
                torch.randn(1, 2),
            ),
            lambda: (
                "add_dims_undefined_m1",
                ExpandModel([-1, 2, 3]),
                torch.randn(1, 2),
            ),
            lambda: (
                "add_dims_undefined_zero",
                ExpandModel([0, 2, 3]),
                torch.randn(1, 2),
            ),
            lambda: (
                "add_dims_undefined_m2",
                ExpandModel([-2, 2, 3]),
                torch.randn(1, 2),
            ),
        ]
    )
    def test_expand_error(self, _, module, a):
        """Test of the PyTorch expand Node on Glow."""
        utils.compare_tracing_methods_error(
            module,
            a,
            fusible_ops={"aten::expand"},
        )
