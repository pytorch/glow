from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
from typing import Union

import torch
from parameterized import parameterized
from tests import utils


class SimpleCompareOpsModule(torch.nn.Module):
    def __init__(self, opType):
        super(SimpleCompareOpsModule, self).__init__()
        self.opType = opType

    def forward(self, a, b):
        if self.opType == "equal":
            return torch.eq(a, b + 0.1)
        elif self.opType == "notEqual":
            return torch.ne(a, b + 0.1)
        elif self.opType == "lessThan":
            return torch.lt(a, b + 0.1)
        elif self.opType == "lessEqual":
            return torch.le(a, b + 0.1)
        elif self.opType == "greaterThan":
            return torch.gt(a, b + 0.1)
        elif self.opType == "greaterEqual":
            return torch.ge(a, b + 0.1)


class SimpleScalarVectorCmpModule(torch.nn.Module):
    def __init__(self, opType: str, scalar: Union[float, int]):
        super().__init__()
        self.opType = opType
        self.scalar = scalar

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if self.opType == "equal":
            return a == self.scalar
        if self.opType == "greaterEqual":
            return a >= self.scalar
        if self.opType == "greaterThan":
            return a > self.scalar
        if self.opType == "lessEqual":
            return a <= self.scalar
        if self.opType == "lessThan":
            return a < self.scalar
        if self.opType == "notEqual":
            return a != self.scalar


class TestCmp(unittest.TestCase):
    def test_equal_basic(self):
        """Basic test of the PyTorch Equal Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("equal"),
            torch.randn(3, 4, 5),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::eq"},
        )

    def test_equal_bcast(self):
        """Basic test of the PyTorch Equal Node (broadcast) on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("equal"),
            torch.randn(3, 4, 5),
            torch.randn(4, 5),
            fusible_ops={"aten::eq"},
        )

    def test_not_equal(self):
        """Basic test of the PyTorch Not equal Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("notEqual"),
            torch.randn(3, 4, 5),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::ne"},
        )

    def test_not_equal_bcast(self):
        """Basic test of the PyTorch Not equal (broadcast) Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("notEqual"),
            torch.randn(3, 4, 5),
            torch.randn(4, 5),
            fusible_ops={"aten::ne"},
        )

    def test_less_than(self):
        """Basic test of the PyTorch Less than Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("lessThan"),
            torch.randn(3, 4, 5),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::lt"},
        )

    def test_less_than_bcast(self):
        """Basic test of the PyTorch Less than (broadcast) Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("lessThan"),
            torch.randn(3, 4, 5),
            torch.randn(4, 5),
            fusible_ops={"aten::lt"},
        )

    def test_less_equal(self):
        """Basic test of the PyTorch less equal Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("lessEqual"),
            torch.randn(3, 4, 5),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::le"},
        )

    def test_less_equal_bcast(self):
        """Basic test of the PyTorch less equal (Broadcast) Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("lessEqual"),
            torch.randn(3, 4, 5),
            torch.randn(4, 5),
            fusible_ops={"aten::le"},
        )

    def test_greater_than(self):
        """Basic test of the PyTorch Greater than Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("greaterThan"),
            torch.randn(3, 4, 5),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::gt"},
        )

    def test_greater_than_bcast(self):
        """Basic test of the PyTorch Greater than (Broadcast) Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("greaterThan"),
            torch.randn(3, 4, 5),
            torch.randn(4, 5),
            fusible_ops={"aten::gt"},
        )

    def test_greater_equal(self):
        """Basic test of the PyTorch Greater Equal Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("greaterEqual"),
            torch.randn(3, 4, 5),
            torch.randn(3, 4, 5),
            fusible_ops={"aten::ge"},
        )

    def test_greater_equal_bcast(self):
        """Basic test of the PyTorch Greater Equal (broadcast) Node on Glow."""
        utils.compare_tracing_methods(
            SimpleCompareOpsModule("greaterEqual"),
            torch.randn(3, 4, 5),
            torch.randn(4, 5),
            fusible_ops={"aten::ge"},
        )

    @parameterized.expand(
        [
            ("eq_tensor_scalar", "equal", "aten::eq"),
            ("gt_tensor_scalar", "greaterThan", "aten::gt"),
            ("ge_tensor_scalar", "greaterEqual", "aten::ge"),
            ("le_tensor_scalar", "lessEqual", "aten::le"),
            ("lt_tensor_scalar", "lessThan", "aten::lt"),
            ("ne_tensor_scalar", "notEqual", "aten::ne"),
        ]
    )
    def test_scalar_vector_cmp(self, _, opType, op):
        """Testing comparisons between tensors and scalars."""
        utils.compare_tracing_methods(
            SimpleScalarVectorCmpModule(opType, 0.5),
            torch.randn(3, 4, 5),
            fusible_ops={op},
        )
