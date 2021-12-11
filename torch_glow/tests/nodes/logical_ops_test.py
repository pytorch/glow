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


class SimpleXorModule(torch.nn.Module):
    def __init__(self):
        super(SimpleXorModule, self).__init__()

    def forward(self, a, b):
        c = torch.logical_xor(a, b)
        return torch.logical_xor(c, c)


class SimpleOrModule(torch.nn.Module):
    def __init__(self):
        super(SimpleOrModule, self).__init__()

    def forward(self, a, b):
        c = torch.logical_or(a, b)
        return torch.logical_or(c, c)


class SimpleAndModule(torch.nn.Module):
    def __init__(self):
        super(SimpleAndModule, self).__init__()

    def forward(self, a, b):
        c = torch.logical_and(a, b)
        return torch.logical_and(c, c)


class SimpleNotModule(torch.nn.Module):
    def __init__(self):
        super(SimpleNotModule, self).__init__()

    def forward(self, a):
        b = torch.logical_not(a)
        return torch.logical_not(b)


class TestXor(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            lambda: (
                "broadcast",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((4, 5), dtype=torch.bool),
            ),
        ]
    )
    def test_xor(self, _, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            SimpleXorModule(),
            a,
            b,
            fusible_ops={"aten::logical_xor"},
            skip_to_glow=skip_to_glow,
        )


class TestOr(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            lambda: (
                "broadcast",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((4, 5), dtype=torch.bool),
            ),
        ]
    )
    def test_or(self, _, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            SimpleOrModule(),
            a,
            b,
            fusible_ops={"aten::logical_or"},
            skip_to_glow=skip_to_glow,
        )


class TestAnd(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            lambda: (
                "broadcast",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((4, 5), dtype=torch.bool),
            ),
        ]
    )
    def test_and(self, _, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            SimpleAndModule(),
            a,
            b,
            fusible_ops={"aten::logical_and"},
            skip_to_glow=skip_to_glow,
        )


class TestNot(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [lambda: ("basic", torch.zeros((3, 4, 5), dtype=torch.bool))]
    )
    def test_not(self, _, a, skip_to_glow=False):
        utils.compare_tracing_methods(
            SimpleNotModule(),
            a,
            fusible_ops={"aten::logical_not"},
            skip_to_glow=skip_to_glow,
        )
