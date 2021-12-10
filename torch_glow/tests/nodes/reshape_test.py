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


class SimpleReshapeModel(torch.nn.Module):
    def __init__(self, shape):
        super(SimpleReshapeModel, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        combined = tensor + tensor
        return combined.reshape(self.shape)


class TestReshape(utils.TorchGlowTestCase):
    def test_reshape(self):
        """Test of the PyTorch reshape Node on Glow."""

        utils.compare_tracing_methods(
            SimpleReshapeModel([2, -1]),
            torch.rand(2, 3, 4),
            fusible_ops={"aten::reshape"},
        )
