from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleTanhModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleTanhModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        tensor = tensor + tensor
        return tensor.tanh_() if self.inplace else tensor.tanh()


class TestTanh(unittest.TestCase):
    def test_tanh(self):
        """Basic test of the PyTorch aten::tanh Node on Glow."""

        utils.compare_tracing_methods(
            SimpleTanhModel(), torch.randn(4), fusible_ops={"aten::tanh"}
        )

    def test_tanh_inplace(self):
        """Basic test of the PyTorch aten::tanh_ Node on Glow."""

        utils.compare_tracing_methods(
            SimpleTanhModel(inplace=True), torch.randn(4), fusible_ops={"aten::tanh"}
        )
