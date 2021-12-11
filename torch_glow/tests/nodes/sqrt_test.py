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


class SimpleSqrtModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleSqrtModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            other = tensor.sqrt_()
            return other.sqrt_()
        else:
            tensor = torch.sqrt(tensor)
            return torch.sqrt(tensor)


class TestSqrt(utils.TorchGlowTestCase):
    def test_sqrt_basic(self):
        """Test of the PyTorch sqrt Node on Glow."""

        # Make sure the input is positive and not super close to zero.
        utils.compare_tracing_methods(SimpleSqrtModel(), torch.rand(4) + 5)

    def test_sqrt_inplace(self):
        """Test of the PyTorch inplace sqrt Node on Glow."""

        # Make sure the input is positive and not super close to zero.
        utils.compare_tracing_methods(SimpleSqrtModel(inplace=True), torch.rand(4) + 5)
