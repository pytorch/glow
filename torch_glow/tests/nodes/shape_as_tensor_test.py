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


class SimpleShapeAsTensorModel(torch.nn.Module):
    def __init__(self):
        super(SimpleShapeAsTensorModel, self).__init__()

    def forward(self, tensor):
        result = torch._shape_as_tensor(tensor)
        return result + result


class TestShapeAsTensor(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("single dimension", SimpleShapeAsTensorModel(), torch.randn(6)),
            lambda: (
                "multiple dimensions",
                SimpleShapeAsTensorModel(),
                torch.randn(3, 2, 4),
            ),
        ]
    )
    def test_shape_as_tensor(self, _, module, tensor):
        """Test of the PyTorch ShapeAsTensor Node on Glow."""
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops={"aten::_shape_as_tensor"},
        )
