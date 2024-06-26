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

# pyre-ignore-all-errors

from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from glow.glow.torch_glow.tests.tests import utils


class SimpleMmModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMmModule, self).__init__()

    def forward(self, a, b, t):
        r = torch.mm(a, b)
        return r.mm(t)


class TestMm(utils.TorchGlowTestCase):
    def test_mm_basic(self):
        """Test of the PyTorch mm Node on Glow."""

        x = torch.randn(2, 3)
        y = torch.randn(4, 3).t()
        t = torch.randn(4, 2)

        utils.compare_tracing_methods(
            SimpleMmModule(), x, y, t, fusible_ops={"aten::mm"}, skip_to_glow=True
        )
