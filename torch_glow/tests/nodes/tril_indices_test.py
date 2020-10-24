from __future__ import absolute_import, division, print_function, unicode_literals

import torch

from tests import utils
from parameterized import parameterized
import unittest


class SimpleTrilIndicesModule(torch.nn.Module):
    def __init__(self, row, col, offset):
        super(SimpleTrilIndicesModule, self).__init__()
        self.row = row
        self.col = col
        self.offset = offset

    def forward(
        self,
    ):
        b = torch.tril_indices(self.row, self.col, self.offset)
        return b + b


class TestTrilIndices(unittest.TestCase):
    @parameterized.expand(
        [
            ("basic", SimpleTrilIndicesModule(2, 4, 0)),
            ("negative_offset", SimpleTrilIndicesModule(2, 4, -1)),
            ("positive_offset", SimpleTrilIndicesModule(2, 4, 1)),
            ("col_gt_row", SimpleTrilIndicesModule(4, 2, 1)),
            ("large_offset", SimpleTrilIndicesModule(4, 6, 1324)),
        ]
    )
    def test_tril_indices(self, _, module):
        utils.compare_tracing_methods(
            module, fusible_ops={"aten::tril_indices"}, scripted=True
        )
