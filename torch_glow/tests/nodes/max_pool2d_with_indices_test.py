from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleMaxPool2dTest(torch.nn.Module):
    def __init__(self, kernel_size, padding=0):
        super(SimpleMaxPool2dTest, self).__init__()
        self.kernel_size = kernel_size
        self.padding = padding

    def forward(self, inputs):
        return F.max_pool2d_with_indices(
            inputs, kernel_size=self.kernel_size, padding=self.padding
        )


class TestMaxPool2dWithIndices(unittest.TestCase):
    def test_max_pool2d_with_indices_basic(self):
        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(3),
            torch.randn(20, 16, 50, 32),
            fusible_ops={"aten::max_pool2d_with_indices"},
        )

    def test_max_pool2d_with_with_indices_pad(self):
        utils.compare_tracing_methods(
            SimpleMaxPool2dTest(7, 3),
            torch.randn(1, 4, 10, 10),
            fusible_ops={"aten::max_pool2d_with_indices"},
        )
