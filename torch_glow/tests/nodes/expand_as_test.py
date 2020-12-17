from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class ExpandAsModel(torch.nn.Module):
    def __init__(self, shape):
        super(ExpandAsModel, self).__init__()
        self.other = torch.randn(shape)

    def forward(self, a):
        return a.expand_as(self.other)


class TestClamp(unittest.TestCase):
    def test_clamp_min(self):
        """Test of the PyTorch expand_as Node on Glow."""

        utils.compare_tracing_methods(
            ExpandAsModel([2, 2, 4]), torch.randn(1, 4), fusible_ops={"aten::expand_as"}
        )
