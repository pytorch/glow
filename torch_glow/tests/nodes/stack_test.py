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


class SimpleStackModel(torch.nn.Module):
    def __init__(self, dim):
        super(SimpleStackModel, self).__init__()
        self.dim = dim

    def forward(self, a, b):
        c = b + b
        return torch.stack((a, c), dim=self.dim)


class TestStack(utils.TorchGlowTestCase):
    def test_stack_basic(self):
        """Basic test of the PyTorch aten::stack Node on Glow."""

        for d in range(0, 4):
            utils.compare_tracing_methods(
                SimpleStackModel(d),
                torch.randn(2, 3, 4),
                torch.randn(2, 3, 4),
                skip_to_glow=True,
            )

    def test_stack_different_types(self):
        """Test stack between fp16 and fp32, which is supported in pytorch."""

        x = torch.randn(2, 3, 4)
        y = torch.randn(2, 3, 4, dtype=torch.half)

        for d in range(0, 4):
            utils.compare_tracing_methods(
                SimpleStackModel(d),
                x,
                y,
                skip_to_glow=True,
            )
