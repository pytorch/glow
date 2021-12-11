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
import torch.nn.functional as F
from tests import utils


class SimpleReluModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleReluModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        other = F.relu(tensor, inplace=self.inplace)
        return F.relu(other, inplace=self.inplace)


class TestRelu(utils.TorchGlowTestCase):
    def test_relu_basic(self):
        """Basic test of the PyTorch relu Node on Glow."""

        x = torch.randn(4)
        # make sure we have at least one negative
        x[0] = -2.0

        utils.compare_tracing_methods(SimpleReluModel(), x, fusible_ops={"aten::relu"})

    def test_relu_inplace(self):
        """Test of the PyTorch relu_ Node on Glow."""

        x = torch.randn(4)
        # make sure we have at least one negative
        x[0] = -2.0

        utils.compare_tracing_methods(
            SimpleReluModel(inplace=True), x, fusible_ops={"aten::relu_"}
        )
