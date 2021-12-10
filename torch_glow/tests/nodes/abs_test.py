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


class SimpleAbsModule(torch.nn.Module):
    def __init__(self):
        super(SimpleAbsModule, self).__init__()

    def forward(self, a):
        return torch.abs(a + a)


class TestAbs(utils.TorchGlowTestCase):
    def test_abs_basic(self):
        """Basic test of the PyTorch Abs Node on Glow."""

        x = torch.randn(10)
        utils.run_comparison_tests(
            SimpleAbsModule(),
            x,
            fusible_ops={"aten::abs"},
        )

    def test_abs_3d(self):
        """Test multidimensional tensor for the PyTorch Abs Node on Glow."""

        x = torch.randn(2, 3, 5)
        utils.run_comparison_tests(
            SimpleAbsModule(),
            x,
            fusible_ops={"aten::abs"},
        )
