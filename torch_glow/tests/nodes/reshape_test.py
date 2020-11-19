from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleReshapeModel(torch.nn.Module):
    def __init__(self, shape):
        super(SimpleReshapeModel, self).__init__()
        self.shape = shape

    def forward(self, tensor):
        combined = tensor + tensor
        return combined.reshape(self.shape)


class TestReshape(unittest.TestCase):
    def test_reshape(self):
        """Test of the PyTorch reshape Node on Glow."""

        utils.compare_tracing_methods(
            SimpleReshapeModel([2, -1]),
            torch.rand(2, 3, 4),
            fusible_ops={"aten::reshape"},
        )
