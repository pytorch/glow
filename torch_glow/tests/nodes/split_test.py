from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleSplitModel(torch.nn.Module):
    def __init__(self, split_size_or_sections, dimension):
        super(SimpleSplitModel, self).__init__()
        self.split_size_or_sections = split_size_or_sections
        self.dimension = dimension

    def forward(self, x):
        return torch.split(x, self.split_size_or_sections, self.dimension)


class TestSplit(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (torch.randn(8), 4, 0),
            lambda: (torch.randn(10), [1, 2, 3, 4], 0),
            lambda: (torch.randn(10, 10, 10), 3, 2),
            lambda: (torch.randn(100, 100), [25, 50, 25], 1),
        ]
    )
    def test_split(self, tensor, split_size_or_sections, dimension):
        utils.compare_tracing_methods(
            SimpleSplitModel(split_size_or_sections, dimension), tensor
        )
