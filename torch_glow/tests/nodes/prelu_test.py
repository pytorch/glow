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


class SimplePreluModule(torch.nn.Module):
    def __init__(self):
        super(SimplePreluModule, self).__init__()

    def forward(self, inputs, weights):
        return F.prelu(inputs + inputs, weights)


class TestPrelu(utils.TorchGlowTestCase):
    def test_prelu_basic(self):
        """Basic test of the PyTorch prelu Node on Glow."""

        utils.compare_tracing_methods(
            SimplePreluModule(),
            torch.randn(1, 4, 5, 5),
            torch.tensor([0.25]),
            fusible_ops={"aten::prelu"},
        )
