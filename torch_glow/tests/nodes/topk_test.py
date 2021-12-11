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


class SimpleTopkModel(torch.nn.Module):
    def __init__(self, count):
        super(SimpleTopkModel, self).__init__()
        self.count = count

    def forward(self, tensor):
        tensor = tensor + tensor
        return torch.topk(tensor, self.count)


class TestTopk(utils.TorchGlowTestCase):
    def test_topk_basic(self):
        """Test of the PyTorch TopK Node on Glow."""
        utils.compare_tracing_methods(
            SimpleTopkModel(3), torch.arange(1.0, 6.0), fusible_ops={"aten::topk"}
        )
