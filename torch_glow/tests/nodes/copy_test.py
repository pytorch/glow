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


class CopyModel(torch.nn.Module):
    def __init__(self, shape):
        super(CopyModel, self).__init__()
        self.other = torch.randn(shape)

    def forward(self, a):
        b = a.copy_(self.other)
        return a + b


class TestCopy(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("1x1 => 1x3", [1, 1], [1, 3]),
            lambda: ("1x3x5 => 1x3x5", [1, 3, 5], [1, 3, 5]),
            lambda: ("1x3 => 4x4x3", [1, 3], [4, 4, 3]),
        ]
    )
    def test_copy_(self, _, other_shape, tensor_shape):
        """Test of the PyTorch copy_ method on Glow."""

        utils.compare_tracing_methods(
            CopyModel(other_shape),
            torch.randn(tensor_shape),
            fusible_ops={"aten::copy_"},
        )

    @utils.deterministic_expand(
        [
            lambda: ("1x1x1 => 1x3", [1, 1, 1], [1, 3]),
            lambda: ("1x4 => 4x4x3", [1, 4], [4, 4, 3]),
            lambda: ("4x4x3 => 1x3", [4, 4, 3], [1, 3]),
        ]
    )
    def test_copy_broadcast_failure(self, _, other_shape, tensor_shape):
        """Test of the PyTorch copy_ method on Glow."""

        with self.assertRaises(RuntimeError):
            utils.compare_tracing_methods(
                CopyModel(other_shape),
                torch.randn(tensor_shape),
                fusible_ops={"aten::copy_"},
            )
