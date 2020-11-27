from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleSliceModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSliceModel, self).__init__()

    def forward(self, tensor):
        other = (tensor + tensor)[1:]
        return other[0][1:]


class TestSlice(unittest.TestCase):
    def test_slice_basic(self):
        """Test of the PyTorch slice Node on Glow."""

        utils.compare_tracing_methods(
            SimpleSliceModel(), torch.rand((2, 3)), skip_to_glow=True
        )
