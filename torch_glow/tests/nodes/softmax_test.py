from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from parameterized import parameterized
from tests import utils


class SimpleSoftmaxModel(torch.nn.Module):
    def __init__(self, dimension):
        super(SimpleSoftmaxModel, self).__init__()
        self.dimension = dimension

    def forward(self, tensor):
        return F.softmax(tensor, self.dimension)


class TestSoftmax(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleSoftmaxModel(1), torch.randn(2, 3)),
            ("neg_dim", SimpleSoftmaxModel(-1), torch.randn(2, 3)),
        ]
    )
    def test_softmax(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor)

    def test_softmax_oob_neg_dim(self):
        """Test out of bounds negative dimension index for the PyTorch SoftMax Node on Glow."""

        with self.assertRaises(IndexError):
            utils.compare_tracing_methods(SimpleSoftmaxModel(-3), torch.randn(2, 3))
