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


class SimpleExpModule(torch.nn.Module):
    def forward(self, input):
        other = torch.exp(input)
        return torch.exp(other)


class TestExp(utils.TorchGlowTestCase):
    def test_exp_basic(self):
        """Test of the PyTorch exp Node on Glow."""

        utils.compare_tracing_methods(
            SimpleExpModule(), torch.randn(4), fusible_ops={"aten::exp"}
        )
