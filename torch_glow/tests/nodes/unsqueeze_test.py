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


class SimpleUnsqueezeModel(torch.nn.Module):
    def __init__(self, dimension, inplace=False):
        super(SimpleUnsqueezeModel, self).__init__()
        self.dimension = dimension
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            other = tensor + tensor
            return other.unsqueeze_(self.dimension)
        else:
            return torch.unsqueeze(tensor + tensor, self.dimension)


class TestUnsqueeze(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("dim0", SimpleUnsqueezeModel(0), torch.randn(2, 3, 4)),
            lambda: ("dim1", SimpleUnsqueezeModel(1), torch.randn(2, 3, 4)),
            lambda: ("dim2", SimpleUnsqueezeModel(2), torch.randn(2, 3, 4)),
            lambda: ("dim3", SimpleUnsqueezeModel(3), torch.randn(2, 3, 4)),
            lambda: ("dim_negative", SimpleUnsqueezeModel(-1), torch.randn(2, 3, 4)),
            lambda: (
                "inplace",
                SimpleUnsqueezeModel(-1, inplace=True),
                torch.randn(2, 3, 4),
            ),
        ]
    )
    def test_unsqueeze(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops=["aten::unsqueeze"])
