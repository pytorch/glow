from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
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


class TestXor(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            (
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


class TestOr(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            (
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


class TestAnd(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                torch.zeros((3, 4, 5), dtype=torch.bool),
                torch.ones((3, 4, 5), dtype=torch.bool),
            ),
            (
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


class TestNot(unittest.TestCase):
    @parameterized.expand([("basic", torch.zeros((3, 4, 5), dtype=torch.bool))])
    def test_not(self, _, a, skip_to_glow=False):
        utils.compare_tracing_methods(
            SimpleNotModule(),
            a,
            fusible_ops={"aten::logical_not"},
            skip_to_glow=skip_to_glow,
        )
