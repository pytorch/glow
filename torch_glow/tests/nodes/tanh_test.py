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


class SimpleTanhModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleTanhModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        tensor = tensor + tensor
        return tensor.tanh_() if self.inplace else tensor.tanh()


class TestTanh(utils.TorchGlowTestCase):
    def test_tanh(self):
        """Basic test of the PyTorch aten::tanh Node on Glow."""

        utils.compare_tracing_methods(
            SimpleTanhModel(), torch.randn(4), fusible_ops={"aten::tanh"}
        )

    def test_tanh_inplace(self):
        """Basic test of the PyTorch aten::tanh_ Node on Glow."""

        utils.compare_tracing_methods(
            SimpleTanhModel(inplace=True), torch.randn(4), fusible_ops={"aten::tanh"}
        )
