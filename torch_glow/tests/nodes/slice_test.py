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


class SimpleSliceModel(torch.nn.Module):
    def __init__(self, start, end):
        super(SimpleSliceModel, self).__init__()
        self.start = start
        self.end = end

    def forward(self, x):
        x = x + x
        if self.start is None and self.end is None:
            return x[:]
        elif self.start is None:
            return x[: self.end]
        elif self.end is None:
            return x[self.start :]
        else:
            return x[self.start : self.end]


class TestSlice(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (0, 1),
            lambda: (0, 2),
            lambda: (0, 3),
            lambda: (1, 2),
            lambda: (1, 3),
            lambda: (2, 3),
            lambda: (-3, 1),
            lambda: (-2, 2),
            lambda: (-1, 3),
            lambda: (-2, -1),
            lambda: (0, -1),
            lambda: (1, -1),
            lambda: (None, 2),
            lambda: (None, -1),
            lambda: (0, None),
            lambda: (-2, None),
            lambda: (None, None),
        ]
    )
    def test_slice(self, start, end):
        """Test of the PyTorch slice Node on Glow."""

        input = torch.rand(3, 2, 2)
        utils.compare_tracing_methods(
            SimpleSliceModel(start, end), input, fusible_ops={"aten::slice"}
        )
