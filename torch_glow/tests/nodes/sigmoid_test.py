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

# isort:skip_file
from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleSigmoidModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleSigmoidModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            other = tensor + tensor
            return other.sigmoid_()
        else:
            other = tensor + tensor
            return other.sigmoid()


class TestSigmoid(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleSigmoidModel(), torch.randn(6)),
            lambda: ("inplace", SimpleSigmoidModel(inplace=True), torch.randn(6)),
        ]
    )
    def test_sigmoid(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor, fusible_ops={"aten::sigmoid"})
