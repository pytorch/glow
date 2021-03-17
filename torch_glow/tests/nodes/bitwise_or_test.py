from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleBitwiseOrModule(torch.nn.Module):
    def __init__(self, dtype=None):
        super(SimpleBitwiseOrModule, self).__init__()
        self.dtype = dtype

    def forward(self, a, b):
        c = torch.bitwise_or(a, b)
        d = torch.bitwise_or(a, c)
        return d


class TestBitwiseOr(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: (
                "basic",
                torch.tensor([0x01, 0x03, 0xFFFFFFF0, 0x5], dtype=torch.int32),
                torch.tensor([0x02, 0x03, 0x2, 0x1F], dtype=torch.int32),
            ),
            lambda: (
                "basic_64",
                torch.tensor([0x01, 0x03, 0xFFFFFFFFFF00, 0x5], dtype=torch.int64),
                torch.tensor([0x02, 0x03, 0x7, 0x1F], dtype=torch.int64),
            ),
            lambda: (
                "basic_bool",
                torch.tensor([True, True, False, False], dtype=torch.bool),
                torch.tensor([True, False, True, False], dtype=torch.bool),
            ),
            lambda: (
                "basic_3d",
                torch.zeros((0x1, 0x04, 0x1), dtype=torch.int32),
                torch.ones((0x2, 0x1, 0x4), dtype=torch.int32),
            ),
            lambda: (
                "broadcast_3d",
                torch.zeros((3, 4, 5), dtype=torch.int32),
                torch.ones((4, 5), dtype=torch.int32),
            ),
        ]
    )
    def test_bitwise_or(self, _, a, b, skip_to_glow=False):
        """Tests of the PyTorch Bitwise Or Node on Glow."""
        utils.compare_tracing_methods(
            SimpleBitwiseOrModule(),
            a,
            b,
            fusible_ops={"aten::bitwise_or"},
        )
