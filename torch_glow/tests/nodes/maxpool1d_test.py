from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized
from tests import utils


class SimpleMaxPool1dTest(torch.nn.Module):
    def __init__(self, kernel_size, padding=0, stride=None):
        super(SimpleMaxPool1dTest, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

    def forward(self, inputs):
        return F.max_pool1d(
            inputs,
            kernel_size=self.kernel_size,
            padding=self.padding,
            stride=self.stride,
        )


class TestMaxPool1d(unittest.TestCase):
    @parameterized.expand(
        [
            (
                "basic",
                SimpleMaxPool1dTest(3),
                torch.randn(1, 4, 5),
            ),
            (
                "padding",
                SimpleMaxPool1dTest(2, padding=1),
                torch.randn(2, 5, 3),
            ),
            (
                "stride",
                SimpleMaxPool1dTest(3, stride=2),
                torch.randn(3, 4, 4),
            ),
            (
                "args",
                SimpleMaxPool1dTest(7, padding=3, stride=2),
                torch.randn(3, 8, 13),
            ),
        ]
    )
    def test_max_pool1d(self, _, module, tensor):
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops={"aten::max_pool1d"},
        )
