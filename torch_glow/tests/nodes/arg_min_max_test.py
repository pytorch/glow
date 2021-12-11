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


class ArgMinModule(torch.nn.Module):
    def __init__(self, dim=None, keepDims=True):
        super(ArgMinModule, self).__init__()
        self.dim = dim
        self.keepDims = keepDims

    def forward(self, tensor):
        if self.dim:
            return torch.argmin(tensor, self.dim, self.keepDims)
        else:
            return torch.argmin(tensor)


class ArgMaxModule(torch.nn.Module):
    def __init__(self, dim=None, keepDims=True):
        super(ArgMaxModule, self).__init__()
        self.dim = dim
        self.keepDims = keepDims

    def forward(self, tensor):
        if self.dim:
            return torch.argmax(tensor, self.dim, self.keepDims)
        else:
            return torch.argmax(tensor)


class TestArgMin(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", ArgMinModule(), torch.randn(4)),
            lambda: ("dimensions1", ArgMinModule(1, False), torch.randn(4, 4)),
            lambda: ("dimensions2", ArgMinModule(1), torch.randn(5, 5)),
        ]
    )
    def test_argmin_node(self, _, module, tensor):
        """Test of the PyTorch ArgMin node on Glow."""
        utils.run_comparison_tests(module, tensor, fusible_ops={"aten::argmin"})


class TestArgMax(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", ArgMaxModule(), torch.randn(4)),
            lambda: ("dimensions1", ArgMaxModule(1, False), torch.randn(4, 4)),
            lambda: ("dimensions2", ArgMaxModule(1), torch.randn(5, 5)),
        ]
    )
    def test_argmax_node(self, _, module, tensor):
        """Test of the PyTorch ArgMax node on Glow."""
        utils.run_comparison_tests(module, tensor, fusible_ops={"aten::argmax"})
