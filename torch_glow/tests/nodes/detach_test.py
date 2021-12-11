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


class DetachModel(torch.nn.Module):
    def __init__(self):
        super(DetachModel, self).__init__()

    def forward(self, a):
        b = a.detach()
        return b + b


class TestDetach(utils.TorchGlowTestCase):
    def test_detach(self):
        """Test of the PyTorch detach Node on Glow."""

        x = torch.randn(5, 6, 7)
        x.requires_grad = True

        utils.compare_tracing_methods(DetachModel(), x, fusible_ops={"aten::detach"})
