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


class ExpandAsModel(torch.nn.Module):
    def __init__(self, shape):
        super(ExpandAsModel, self).__init__()
        self.other = torch.randn(shape)

    def forward(self, a):
        return a.expand_as(self.other)


class TestClamp(utils.TorchGlowTestCase):
    def test_clamp_min(self):
        """Test of the PyTorch expand_as Node on Glow."""

        utils.compare_tracing_methods(
            ExpandAsModel([2, 2, 4]), torch.randn(1, 4), fusible_ops={"aten::expand_as"}
        )
