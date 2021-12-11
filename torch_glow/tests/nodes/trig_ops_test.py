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

import numpy as np
import torch
from tests import utils


class SimpleCosModule(torch.nn.Module):
    def __init__(self):
        super(SimpleCosModule, self).__init__()

    def forward(self, a):
        return torch.cos(a + a)


class SimpleSinModule(torch.nn.Module):
    def __init__(self):
        super(SimpleSinModule, self).__init__()

    def forward(self, a):
        return torch.sin(a + a)


class SimpleACosModule(torch.nn.Module):
    def __init__(self):
        super(SimpleACosModule, self).__init__()

    def forward(self, a):
        return torch.acos(a + a)


class SimpleASinModule(torch.nn.Module):
    def __init__(self):
        super(SimpleASinModule, self).__init__()

    def forward(self, a):
        return torch.asin(a + a)


class SimpleATanModule(torch.nn.Module):
    def __init__(self):
        super(SimpleATanModule, self).__init__()

    def forward(self, a):
        return torch.atan(a + a)


class TestCos(utils.TorchGlowTestCase):
    def test_cos(self, skip_to_glow=False):
        # Ensures range is in [-2*pi, 2*pi]
        x = 4 * np.pi * (torch.rand(2, 3, 4) - 0.5)
        utils.compare_tracing_methods(
            SimpleCosModule(), x, fusible_ops={"aten::cos"}, skip_to_glow=skip_to_glow
        )


class TestSin(utils.TorchGlowTestCase):
    def test_sin(self, skip_to_glow=False):
        # Ensures range is in [-2*pi, 2*pi]
        x = 4 * np.pi * (torch.rand(2, 3, 4) - 0.5)
        utils.compare_tracing_methods(
            SimpleSinModule(), x, fusible_ops={"aten::sin"}, skip_to_glow=skip_to_glow
        )


class TestACos(utils.TorchGlowTestCase):
    def test_acos(self, skip_to_glow=False):
        x = torch.rand(2, 3, 4) - 0.5  # Ensures range is in [-1,1]
        utils.compare_tracing_methods(
            SimpleACosModule(), x, fusible_ops={"aten::acos"}, skip_to_glow=skip_to_glow
        )


class TestASin(utils.TorchGlowTestCase):
    def test_asin(self, skip_to_glow=False):
        x = torch.rand(2, 3, 4) - 0.5  # Ensures range is in [-1,1]
        utils.compare_tracing_methods(
            SimpleASinModule(), x, fusible_ops={"aten::asin"}, skip_to_glow=skip_to_glow
        )


class TestATan(utils.TorchGlowTestCase):
    def test_atan(self, skip_to_glow=False):
        x = torch.randn(2, 3, 4)
        utils.compare_tracing_methods(
            SimpleATanModule(), x, fusible_ops={"aten::atan"}, skip_to_glow=skip_to_glow
        )
