from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleArgSortModule(torch.nn.Module):
    def __init__(self, descending=True):
        super(SimpleArgSortModule, self).__init__()
        self.descending = descending

    def forward(self, inputs):
        # Only last dim is currently supported
        return torch.argsort(inputs, dim=-1, descending=self.descending)


class TestArgSort(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "desc",
                SimpleArgSortModule(),
                torch.randn(4),
            ),
            lambda: (
                "asc",
                SimpleArgSortModule(descending=False),
                torch.randn(4),
            ),
            lambda: (
                "2d_desc",
                SimpleArgSortModule(),
                torch.randn(4, 3),
            ),
            lambda: (
                "3d_asc",
                SimpleArgSortModule(descending=False),
                torch.randn(6, 4, 5),
            ),
            lambda: (
                "4d_desc",
                SimpleArgSortModule(),
                torch.randn(4, 7, 7, 3),
            ),
        ]
    )
    def test_argsort(self, _, module, a):
        utils.compare_tracing_methods(module, a, fusible_ops={"aten::argsort"})
