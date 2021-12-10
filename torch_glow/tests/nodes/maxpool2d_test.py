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
import torch.nn.functional as F
from tests import utils


class SimpleMaxPool2dTest(torch.nn.Module):
    def __init__(self, kernel_size, padding=0, ceil_mode=False):
        super(SimpleMaxPool2dTest, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.ceil_mode = ceil_mode

    def forward(self, inputs):
        return F.max_pool2d(
            inputs,
            kernel_size=self.kernel_size,
            padding=self.padding,
            ceil_mode=self.ceil_mode,
        )


class TestMaxPool2d(utils.TorchGlowTestCase):
    def test_max_pool2d_basic(self):
        """Basic test of the PyTorch max_pool2d Node on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(3),
            torch.randn(1, 4, 5, 5),
            fusible_ops={"aten::max_pool2d"},
        )

    def test_max_pool2d_with_args(self):
        """Test of the PyTorch max_pool2d Node with arguments on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(7, 3),
            torch.randn(1, 4, 10, 10),
            fusible_ops={"aten::max_pool2d"},
        )

    def test_max_pool2d_ceil_mode(self):
        """Test of the PyTorch max_pool2d Node with ceil_mode on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(7, 1, ceil_mode=True),
            torch.randn(1, 4, 16, 16),
            fusible_ops={"aten::max_pool2d"},
        )

    def test_max_pool2d_ceil_mode_strong_1(self):
        """Stronger test of the PyTorch max_pool2d Node with ceil_mode on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(5, 1, ceil_mode=True),
            torch.randn(1, 5, 33, 33),
            fusible_ops={"aten::max_pool2d"},
        )

    def test_max_pool2d_ceil_mode_strong_2(self):
        """Stronger test of the PyTorch max_pool2d Node with ceil_mode on Glow."""

        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(8, 2, ceil_mode=True),
            torch.randn(1, 3, 41, 41),
            fusible_ops={"aten::max_pool2d"},
        )
