from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from parameterized import parameterized
from tests import utils


class SimpleSqueezeModel(torch.nn.Module):
    def __init__(self, dimension=None, inplace=False):
        super(SimpleSqueezeModel, self).__init__()
        self.dimension = dimension
        self.inplace = inplace

    def forward(self, tensor):
        if self.inplace:
            tensor = tensor + tensor
            if self.dimension:
                return tensor.squeeze_(self.dimension)
            else:
                return tensor.squeeze_()
        else:
            if self.dimension:
                return torch.squeeze(tensor + tensor, self.dimension)
            else:
                return torch.squeeze(tensor + tensor)


class TestSqueeze(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleSqueezeModel(), torch.randn(1, 3, 1, 2, 5, 1)),
            ("with_dim", SimpleSqueezeModel(2), torch.randn(1, 3, 1, 2, 5, 1)),
            ("with_neg_dim", SimpleSqueezeModel(-1), torch.randn(1, 3, 1, 2, 5, 1)),
            (
                "inplace",
                SimpleSqueezeModel(inplace=True),
                torch.randn(1, 3, 1, 2, 5, 1),
            ),
        ]
    )
    def test_squeeze(self, _, module, tensor):
        utils.compare_tracing_methods(module, tensor)
