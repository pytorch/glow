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


class GatherModule(torch.nn.Module):
    def __init__(self, dimension):
        super(GatherModule, self).__init__()
        self.dimension = dimension

    def forward(self, tensor, index):
        return torch.gather(tensor, self.dimension, index)


class TestGather(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic-1dim",
                torch.tensor([1, 2, 3, 4]),
                0,
                torch.tensor([0, 0, 1, 0]),
            ),
            lambda: (
                "0-dim",
                torch.tensor([[1, 2], [3, 4]]),
                0,
                torch.tensor([[0, 1], [0, 1]]),
            ),
            lambda: (
                "1-dim",
                torch.tensor([[1, 2], [3, 4]]),
                1,
                torch.tensor([[0, 0], [0, 0]]),
            ),
            lambda: (
                "2-dim",
                torch.randn(3, 4, 2),
                2,
                torch.empty(3, 4, 2).random_(2).long(),
            ),
        ]
    )
    def test_gather(self, _, tensor, dimension, index):
        utils.compare_tracing_methods(
            GatherModule(dimension),
            tensor,
            index,
            skip_to_glow=True,
            fusible_ops={"aten::gather"},
        )
