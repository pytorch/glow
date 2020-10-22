from __future__ import absolute_import, division, print_function, unicode_literals

import unittest

import torch
from tests import utils


class SimpleTopkModel(torch.nn.Module):
    def __init__(self, count):
        super(SimpleTopkModel, self).__init__()
        self.count = count

    def forward(self, tensor):
        tensor = tensor + tensor
        return torch.topk(tensor, self.count)


class TestTopk(unittest.TestCase):
    def test_topk_basic(self):
        """Test of the PyTorch TopK Node on Glow."""
        utils.compare_tracing_methods(
            SimpleTopkModel(3), torch.arange(1.0, 6.0), fusible_ops={"aten::topk"}
        )
