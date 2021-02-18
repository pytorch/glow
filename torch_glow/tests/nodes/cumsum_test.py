from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleCumSumModule(torch.nn.Module):
    def __init__(self):
        super(SimpleCumSumModule, self).__init__()

    def forward(self, tensor):
        # TODO remove default of 0 when axis/dimension to sum is supported
        return torch.cumsum(tensor, dim=0)


class TestCumSum(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("1", False, torch.randn(1)),
            lambda: ("2", False, torch.randn(2)),
            lambda: ("20", False, torch.randn(20)),
            # TODO add these tests when multi-dimension is supported
            lambda: ("3x3", True, torch.randn(3, 3)),
            lambda: ("5x4", True, torch.randn(5, 4)),
            lambda: ("3x3x3", True, torch.randn(3, 4, 5)),
            lambda: ("3x4x5", True, torch.randn(3, 4, 5)),
            lambda: ("4x4x4x4", True, torch.randn(6, 5, 4, 3)),
            lambda: ("6x5x4x3", True, torch.randn(6, 5, 4, 3)),
        ]
    )
    def test_cumsum(self, _, skip, tensor):
        if skip:
            self.skipTest("multi-dimension is supported yet support")
        utils.compare_tracing_methods(
            SimpleCumSumModule(), tensor, fusible_ops={"aten::cumsum"}
        )
