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


class SimpleClampMinModel(torch.nn.Module):
    def __init__(self, min):
        super(SimpleClampMinModel, self).__init__()
        self.min = min

    def forward(self, input):
        return torch.clamp_min(input, self.min)


class TestClamp(utils.TorchGlowTestCase):
    def test_clamp_min(self):
        """Test of the PyTorch clamp_min Node on Glow."""

        utils.compare_tracing_methods(
            SimpleClampMinModel(0.1), torch.randn(7), fusible_ops={"aten::clamp_min"}
        )
