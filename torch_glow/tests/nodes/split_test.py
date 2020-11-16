from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils
import unittest
from parameterized import parameterized


class SimpleSplitModule(torch.nn.Module):
    def __init__(self, split, axis):
        super(SimpleSplitModule, self).__init__()
        self.split = split
        self.axis = axis

    def forward(self, tensor):
        return torch.split(tensor + tensor, self.split, self.axis)


class TestSplit(unittest.TestCase):
    @parameterized.expand(
        [
            ("1d_axis_0", SimpleSplitModule(2, 0), torch.arange(8)),
            ("2d_axis_0", SimpleSplitModule(2, 0), torch.arange(8).reshape(4, 2)),
            ("2d_axis_1", SimpleSplitModule(3, 1), torch.arange(14).reshape(2, 7)),
            ("3d_axis_1", SimpleSplitModule(2, 1), torch.arange(30).reshape(2, 5, 3)),
        ]
    )
    def test_split_basic(self, _, module, tensor):
        """Test of the PyTorch split Node on Glow."""
        utils.compare_tracing_methods(
            module,
            tensor,
            fusible_ops={"aten::split", "prim::ListUnpack"},
            scripted=False,
        )
