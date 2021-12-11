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


class CloneModel(torch.nn.Module):
    def __init__(self, memory_format=torch.contiguous_format):
        super(CloneModel, self).__init__()
        self.memory_format = memory_format

    def forward(self, a):
        b = a.clone(memory_format=self.memory_format)
        return b + a


class TestClone(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("1x3", [1, 3]),
            lambda: ("8x3x5", [8, 3, 5]),
        ]
    )
    def test_clone(self, _, tensor_shape):
        """Test of the PyTorch clone method on Glow."""

        utils.compare_tracing_methods(
            CloneModel(),
            torch.randn(tensor_shape),
            fusible_ops={"aten::clone"},
        )

    @utils.deterministic_expand(
        [
            lambda: ("8x3x5x10", [8, 3, 5, 10]),
        ]
    )
    def test_clone_alt_memory_format(self, _, tensor_shape):
        """Test of the PyTorch clone method on Glow."""

        utils.compare_tracing_methods(
            CloneModel(memory_format=torch.channels_last),
            torch.randn(tensor_shape),
            fusible_ops={"aten::clone"},
        )
