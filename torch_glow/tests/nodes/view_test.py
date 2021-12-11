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


class SimpleViewModule(torch.nn.Module):
    def __init__(self, *shape):
        super(SimpleViewModule, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        return (tensor + tensor).view(self.shape)


class TestView(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (SimpleViewModule(2, -1), torch.rand(2, 3, 4)),
            lambda: (SimpleViewModule(-1, 2), torch.rand(2, 3, 4)),
        ]
    )
    def test_simple(self, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::view"})
