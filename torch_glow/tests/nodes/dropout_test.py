from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
import torch.nn.functional as F
from tests import utils


class SimpleDropoutModule(torch.nn.Module):
    def __init__(self, p=0.5, training=True, inplace=False):
        super(SimpleDropoutModule, self).__init__()
        self.p = p
        self.training = training
        self.inplace = inplace

    def forward(self, input):
        return F.dropout(
            input + input, p=self.p, training=self.training, inplace=self.inplace
        )


class TestDropout(unittest.TestCase):
    def test_dropout(self):
        """Basic test of the PyTorch aten::dropout Node on Glow."""

        utils.compare_tracing_methods(
            SimpleDropoutModule(training=False),
            torch.randn(6, 4, 10),
            fusible_ops={"aten::dropout"},
        )

    def test_dropout_inplace(self):
        """Basic test of the PyTorch aten::dropout_ Node on Glow."""
        # Expect fuser to out-of-place the operator
        utils.compare_tracing_methods(
            SimpleDropoutModule(training=False, inplace=True),
            torch.randn(6, 4, 10),
            fusible_ops={"aten::dropout"},
        )
