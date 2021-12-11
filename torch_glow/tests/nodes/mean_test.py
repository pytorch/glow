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


class SimpleMeanModule(torch.nn.Module):
    def __init__(self, dim=None, keepdim=False):
        super(SimpleMeanModule, self).__init__()
        self.dim = dim
        self.keepdim = keepdim

    def forward(self, a, b):
        if self.dim:
            return torch.mean(a + b, self.dim, keepdim=self.keepdim)
        else:
            return torch.mean(a + b)


class TestMean(utils.TorchGlowTestCase):
    def test_basic(self):
        """Test of the PyTorch mean Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMeanModule(),
            torch.randn(7),
            torch.randn(7),
            fusible_ops={"aten::mean"},
        )

    def test_with_dims(self):
        """Test of the PyTorch mean node with dims on Glow."""

        utils.compare_tracing_methods(
            SimpleMeanModule((1, 2)),
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
            fusible_ops={"aten::mean"},
        )

    def test_with_keepdim(self):
        """Test of the PyTorch mean node with dims and keepdim=True on Glow."""

        utils.compare_tracing_methods(
            SimpleMeanModule((2, 1), keepdim=True),
            torch.randn([1, 2, 3, 4]),
            torch.randn([1, 2, 3, 4]),
            fusible_ops={"aten::mean"},
        )
