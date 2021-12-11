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


class SimpleContiguousModel(torch.nn.Module):
    def __init__(self, memory_format=torch.contiguous_format):
        super(SimpleContiguousModel, self).__init__()
        self.memory_format = memory_format

    def forward(self, input):
        formatted = input.contiguous(memory_format=self.memory_format)
        return formatted + formatted


class TestContiguous(utils.TorchGlowTestCase):
    def test_contiguous_basic(self):
        """Test of the PyTorch contiguous Node on Glow."""

        x = torch.randn(2, 2, 2)

        utils.compare_tracing_methods(
            SimpleContiguousModel(), x, fusible_ops={"aten::contiguous"}
        )

    def test_with_alternate_memory_format(self):

        x = torch.randn(3, 4, 5, 6)

        utils.compare_tracing_methods(
            SimpleContiguousModel(torch.channels_last),
            x,
            fusible_ops={"aten::contiguous"},
        )
