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

from typing import Union

import torch
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
    def __init__(self, opType: str, rhsScalar: Union[float, int]):
        super().__init__()
        self.opType = opType
        self.rhsScalar = rhsScalar

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        if self.opType == "equal":
            return a == self.rhsScalar
        if self.opType == "greaterEqual":
            return a >= self.rhsScalar
        if self.opType == "greaterThan":
            return a > self.rhsScalar
        if self.opType == "lessEqual":
            return a <= self.rhsScalar
        if self.opType == "lessThan":
            return a < self.rhsScalar
        if self.opType == "notEqual":
            return a != self.rhsScalar


class TestCmp(utils.TorchGlowTestCase):
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

    @utils.deterministic_expand(
        [
            lambda: (
                "eq_tensor_scalar",
                "equal",
                "aten::eq",
                torch.randn(3, 4, 5),
                0.5,
            ),
            lambda: (
                "gt_tensor_scalar",
                "greaterThan",
                "aten::gt",
                torch.randn(3, 4, 5),
                0.5,
            ),
            lambda: (
                "ge_tensor_scalar",
                "greaterEqual",
                "aten::ge",
                torch.randn(3, 4, 5),
                0.5,
            ),
            lambda: (
                "le_tensor_scalar",
                "lessEqual",
                "aten::le",
                torch.randn(3, 4, 5),
                0.5,
            ),
            lambda: (
                "lt_tensor_scalar",
                "lessThan",
                "aten::lt",
                torch.randn(3, 4, 5),
                0.5,
            ),
            lambda: (
                "ne_tensor_scalar",
                "notEqual",
                "aten::ne",
                torch.randn(3, 4, 5),
                0.5,
            ),
            lambda: (
                "eq_tensor_scalar_int64",
                "equal",
                "aten::eq",
                torch.randn(3, 4, 5).to(torch.int64),
                5,
            ),
            lambda: (
                "gt_tensor_scalar_int64",
                "greaterThan",
                "aten::gt",
                torch.randn(3, 4, 5).to(torch.int64),
                5,
            ),
            lambda: (
                "ge_tensor_scalar_int64",
                "greaterEqual",
                "aten::ge",
                torch.randn(3, 4, 5).to(torch.int64),
                5,
            ),
            lambda: (
                "le_tensor_scalar_int64",
                "lessEqual",
                "aten::le",
                torch.randn(3, 4, 5).to(torch.int64),
                5,
            ),
            lambda: (
                "lt_tensor_scalar_int64",
                "lessThan",
                "aten::lt",
                torch.randn(3, 4, 5).to(torch.int64),
                5,
            ),
            lambda: (
                "ne_tensor_scalar_int64",
                "notEqual",
                "aten::ne",
                torch.randn(3, 4, 5).to(torch.int64),
                5,
            ),
            lambda: (
                "eq_tensor_scalar_int32",
                "equal",
                "aten::eq",
                torch.randn(3, 4, 5).to(torch.int32),
                5,
            ),
            lambda: (
                "gt_tensor_scalar_int32",
                "greaterThan",
                "aten::gt",
                torch.randn(3, 4, 5).to(torch.int32),
                5,
            ),
            lambda: (
                "lt_tensor_scalar_int32",
                "lessThan",
                "aten::lt",
                torch.randn(3, 4, 5).to(torch.int32),
                5,
            ),
            lambda: (
                "eq_tensor_scalar_float_int",
                "equal",
                "aten::eq",
                torch.randn(3, 4, 5),
                5,
            ),
            lambda: (
                "gt_tensor_scalar_float_int",
                "greaterThan",
                "aten::gt",
                torch.randn(3, 4, 5),
                5,
            ),
            lambda: (
                "lt_tensor_scalar_float_int",
                "lessThan",
                "aten::lt",
                torch.randn(3, 4, 5),
                5,
            ),
        ]
    )
    def test_scalar_vector_cmp(self, _, opType, op, lhsTensor, rhsScalar):
        """Testing comparisons between tensors and scalars."""
        utils.compare_tracing_methods(
            SimpleScalarVectorCmpModule(opType, rhsScalar),
            lhsTensor,
            fusible_ops={op},
        )
