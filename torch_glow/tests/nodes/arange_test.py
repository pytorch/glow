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


class SimpleArangeModule(torch.nn.Module):
    def __init__(self, end, start=0, step=1):
        super(SimpleArangeModule, self).__init__()
        self.start = start
        self.end = end
        self.step = step

    def forward(self, dummy):
        start = self.start(dummy) if callable(self.start) else self.start
        end = self.end(dummy) if callable(self.end) else self.end
        step = self.step(dummy) if callable(self.step) else self.step
        return torch.arange(start=start, end=end, step=step)


class TestArange(utils.TorchGlowTestCase):
    """
    Tests for torch.arange glow fusion.

    Note that torch.arange is effectively a constant, so torch jit will try to
    compile it down to said constant. The tests in this class utilize a test
    function which takes a tensor as input, so that we can prevent that from
    happening. Otherwise, there would be nothing to fuse.
    """

    @utils.deterministic_expand(
        [
            lambda: (
                "simple",
                SimpleArangeModule(end=lambda x: x.size(0)),
                torch.randn(10),
            ),
            lambda: (
                "all_args",
                SimpleArangeModule(start=lambda x: x.size(0), end=30, step=1),
                torch.randn(10),
            ),
            lambda: (
                "floats",
                SimpleArangeModule(start=lambda x: x.size(0), end=30.5, step=0.8),
                torch.randn(10),
            ),
            lambda: (
                "negative_step",
                SimpleArangeModule(
                    start=lambda x: x.size(0), end=lambda x: x.size(1), step=-1.2
                ),
                torch.randn(10, 2),
            ),
        ]
    )
    def test_arange(self, _, module, dummy):
        """Testing arange with minimum parameters"""
        utils.run_comparison_tests(module, dummy, fusible_ops={"aten::arange"})
