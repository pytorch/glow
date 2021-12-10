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

import torch
from tests import utils


class IndexSelectModule(torch.nn.Module):
    def __init__(self, dimension):
        super(IndexSelectModule, self).__init__()
        self.dimension = dimension

    def forward(self, tensor, index):
        return torch.index_select(tensor, self.dimension, index)


class TestIndexSelect(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("0-dim", torch.randn(3, 4), 0, torch.tensor([0, 2])),
            lambda: ("1-dim", torch.randn(3, 4), 1, torch.tensor([0, 2])),
            lambda: ("repeat index", torch.randn(3, 4), 1, torch.tensor([2, 2])),
        ]
    )
    def test_index_select(self, _, tensor, dimension, index):
        utils.compare_tracing_methods(
            IndexSelectModule(dimension),
            tensor,
            index,
            skip_to_glow=True,
            fusible_ops={"aten::index_select"},
        )
