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


class SimpleLogModule(torch.nn.Module):
    def __init__(self, *dimensions):
        super(SimpleLogModule, self).__init__()

    def forward(
        self,
        a,
    ):
        b = torch.log(a)
        return torch.log(b)


class TestLog(utils.TorchGlowTestCase):
    def test_log_basic(self):

        x = 1 / torch.rand(3, 4, 5)

        utils.compare_tracing_methods(
            SimpleLogModule(),
            x,
            fusible_ops={"aten::log"},
        )
