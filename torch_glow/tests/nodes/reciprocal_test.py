from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleReciprocalModel(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleReciprocalModel, self).__init__()
        self.inplace = inplace

    def forward(self, tensor):
        other = tensor + tensor
        return other.reciprocal_() if self.inplace else torch.reciprocal(other)


class TestReciprocal(unittest.TestCase):
    def test_reciprocal(self):
        """Test of the PyTorch reciprocal Node on Glow."""

        utils.compare_tracing_methods(
            SimpleReciprocalModel(), torch.randn(4), fusible_ops={"aten::reciprocal"}
        )

    def test_inplace_reciprocal(self):
        """Test of the PyTorch inplace reciprocal Node on Glow."""

        # Expect fuser to out-of-place the operator
        utils.compare_tracing_methods(
            SimpleReciprocalModel(inplace=True),
            torch.randn(4),
            fusible_ops={"aten::reciprocal"},
        )
