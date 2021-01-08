from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


def forward_func(a, b, opType):
    b = b + b
    if opType == "equal":
        return a == b
    elif opType == "notEqual":
        return a != b
    elif opType == "lessThan":
        return a < b
    elif opType == "lessEqual":
        return a <= b
    elif opType == "greaterThan":
        return a > b
    elif opType == "greaterEqual":
        return a >= b


class SimpleCompareOpsModule(torch.nn.Module):
    def __init__(self, opType):
        super(SimpleCompareOpsModule, self).__init__()
        self.opType = opType

    def forward(self, lhs, rhs):
        return forward_func(lhs, rhs, self.opType)


class SimpleCompareOpsScalarModule(torch.nn.Module):
    def __init__(self, opType, rhs):
        super(SimpleCompareOpsScalarModule, self).__init__()
        self.opType = opType
        self.rhs = rhs

    def forward(self, lhs):
        return forward_func(lhs, self.rhs, self.opType)


class TestCmp(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "equal_basic",
                SimpleCompareOpsModule("equal"),
                torch.randn(3, 4, 5),
                torch.randn(3, 4, 5),
                "aten::eq",
            ),
            (
                "equal_broadcast",
                SimpleCompareOpsModule("equal"),
                torch.randn(3, 4, 5),
                torch.randn(4, 5),
                "aten::eq",
            ),
            (
                "not_equal_basic",
                SimpleCompareOpsModule("notEqual"),
                torch.randn(3, 4, 5),
                torch.randn(3, 4, 5),
                "aten::ne",
            ),
            (
                "not_equal_broadcast",
                SimpleCompareOpsModule("notEqual"),
                torch.randn(3, 4, 5),
                torch.randn(4, 5),
                "aten::ne",
            ),
            (
                "less_than_basic",
                SimpleCompareOpsModule("lessThan"),
                torch.randn(3, 4, 5),
                torch.randn(3, 4, 5),
                "aten::lt",
            ),
            (
                "less_than_broadcast",
                SimpleCompareOpsModule("lessThan"),
                torch.randn(3, 4, 5),
                torch.randn(4, 5),
                "aten::lt",
            ),
            (
                "less_equal_basic",
                SimpleCompareOpsModule("lessEqual"),
                torch.randn(3, 4, 5),
                torch.randn(3, 4, 5),
                "aten::le",
            ),
            (
                "less_equal_broadcast",
                SimpleCompareOpsModule("lessEqual"),
                torch.randn(3, 4, 5),
                torch.randn(4, 5),
                "aten::le",
            ),
            (
                "greater_than_basic",
                SimpleCompareOpsModule("greaterThan"),
                torch.randn(3, 4, 5),
                torch.randn(3, 4, 5),
                "aten::gt",
            ),
            (
                "greater_than_broadcast",
                SimpleCompareOpsModule("greaterThan"),
                torch.randn(3, 4, 5),
                torch.randn(4, 5),
                "aten::gt",
            ),
            (
                "greater_equal_basic",
                SimpleCompareOpsModule("greaterEqual"),
                torch.randn(3, 4, 5),
                torch.randn(3, 4, 5),
                "aten::ge",
            ),
            (
                "greater_equal_broadcast",
                SimpleCompareOpsModule("greaterEqual"),
                torch.randn(3, 4, 5),
                torch.randn(4, 5),
                "aten::ge",
            ),
        ]
    )
    def test_compare_ops(self, _, module, lhs, rhs, fusible_ops):
        utils.compare_tracing_methods(module, lhs, rhs, fusible_ops={fusible_ops})


class TestCmpScalar(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "equal_scalar",
                SimpleCompareOpsScalarModule(opType="equal", rhs=0.5),
                torch.ones(3, 4, 5),
                "aten::eq",
            ),
            (
                "not_equal_scalar",
                SimpleCompareOpsScalarModule(opType="notEqual", rhs=0.5),
                torch.ones(3, 4, 5),
                "aten::ne",
            ),
            (
                "less_than_scalar",
                SimpleCompareOpsScalarModule(opType="lessThan", rhs=0.5),
                torch.ones(3, 4, 5),
                "aten::lt",
            ),
            (
                "less_equal_scalar",
                SimpleCompareOpsScalarModule(opType="lessEqual", rhs=0.5),
                torch.ones(3, 4, 5),
                "aten::le",
            ),
            (
                "greater_than_scalar",
                SimpleCompareOpsScalarModule(opType="greaterThan", rhs=0.5),
                torch.ones(3, 4, 5),
                "aten::gt",
            ),
            (
                "greater_equal_scalar",
                SimpleCompareOpsScalarModule(opType="greaterEqual", rhs=0.5),
                torch.ones(3, 4, 5),
                "aten::ge",
            ),
        ]
    )
    def test_compare_ops_scalar(self, _, module, lhs, fusible_ops):
        utils.compare_tracing_methods(module, lhs, fusible_ops={fusible_ops})
