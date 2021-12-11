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


class SimpleFloorDivideModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleFloorDivideModule, self).__init__()
        self.inplace = inplace

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            b = b.item()
        if self.inplace:
            return (a + a).floor_divide_(b)
        else:
            return (a + a).floor_divide(b)


class TestFloorDiv(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                SimpleFloorDivideModule(),
                torch.Tensor(4).random_(0, 5),
                torch.Tensor(4).random_(1, 5),
            ),
            lambda: (
                "inplace",
                SimpleFloorDivideModule(True),
                torch.Tensor(4).random_(0, 5),
                torch.Tensor(4).random_(1, 5),
            ),
            lambda: (
                "positive_float",
                SimpleFloorDivideModule(),
                torch.Tensor(4).random_(0, 5),
                torch.tensor(3.9),
            ),
            lambda: (
                "negative_float",
                SimpleFloorDivideModule(),
                torch.tensor([-4.0]),
                torch.tensor([3.0]),
            ),
            lambda: (
                "positive_broadcast",
                SimpleFloorDivideModule(),
                torch.Tensor(8, 3, 4, 2).random_(0, 5),
                torch.Tensor(4, 2).random_(1, 5),
            ),
            lambda: (
                "positive_broadcast",
                SimpleFloorDivideModule(),
                torch.Tensor(8, 3, 4, 2).random_(0, 5),
                torch.Tensor(1, 2).random_(1, 5),
            ),
            lambda: (
                "positive_broadcast",
                SimpleFloorDivideModule(),
                torch.Tensor(4, 2).random_(0, 5),
                torch.Tensor(8, 3, 4, 2).random_(1, 5),
            ),
            lambda: (
                "positive_int",
                SimpleFloorDivideModule(),
                torch.tensor([5]),
                torch.tensor([4]),
            ),
            lambda: (
                "negative_int",
                SimpleFloorDivideModule(),
                torch.tensor([-5]),
                torch.tensor([4]),
            ),
            lambda: (
                "int64",
                SimpleFloorDivideModule(),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
            ),
        ]
    )
    def test_floor_div(self, _, module, left, right):
        utils.run_comparison_tests(
            module,
            (left, right),
            fusible_ops={"aten::floor_divide"},
            skip_for_backends="NNPI",
        )
