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


class SimpleGeluModule(torch.nn.Module):
    def forward(self, tensor):
        return F.gelu(tensor + tensor)


class TestGelu(utils.TorchGlowTestCase):
    def test_gelu_basic(self):
        """Basic test of the PyTorch gelu Node on Glow."""

        def test_f(a):
            return F.gelu(a + a)

        for _ in range(100):
            x = torch.randn(10)
            utils.compare_tracing_methods(
                SimpleGeluModule(),
                x,
                check_trace=False,
                atol=1e-3,
                fusible_ops={"aten::gelu"},
            )
