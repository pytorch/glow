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

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleBitwiseNotModule(torch.nn.Module):
    def __init__(self):
        super(SimpleBitwiseNotModule, self).__init__()

    def forward(self, a):
        b = torch.bitwise_not(a)
        return torch.bitwise_not(b)


class TestBitwiseNot(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", torch.tensor([-1, -2, 3], dtype=torch.int32)),
            lambda: ("basic_int64", torch.tensor([-1, -2, 3], dtype=torch.int64)),
            lambda: (
                "rand_int",
                torch.randint(-1000000000, 1000000000, (2, 3), dtype=torch.int64),
            ),
            lambda: ("bool_ts", torch.zeros((2, 2, 3), dtype=torch.bool)),
            lambda: ("bool_fs", torch.ones((2, 2, 3), dtype=torch.bool)),
            lambda: ("bool_tf", torch.tensor([False, True], dtype=torch.bool)),
        ]
    )
    def test_bitwise_not(self, _, x):
        """Tests of the PyTorch Bitwise Not Node on Glow."""
        utils.compare_tracing_methods(
            SimpleBitwiseNotModule(),
            x,
            fusible_ops={"aten::bitwise_not"},
        )
