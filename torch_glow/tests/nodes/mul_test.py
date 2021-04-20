from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleMulModule(torch.nn.Module):
    def __init__(self):
        super(SimpleMulModule, self).__init__()

    def forward(self, left, right):
        other = left.mul(right.item() if right.size() == torch.Size([]) else right)
        return other.mul(other)


class TestMul(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", torch.randn(4), torch.randn(4)),
            lambda: ("broadcast", torch.randn(8, 3, 4, 2), torch.randn(4, 2)),
            lambda: ("broadcast", torch.randn(8, 3, 4, 2), torch.randn(1, 2)),
            lambda: ("broadcast", torch.randn(4, 2), torch.randn(8, 3, 4, 2)),
            lambda: ("float", torch.randn(4, 2), torch.tensor(3.2)),
            lambda: ("int", torch.randn(4, 2), torch.tensor(22)),
            lambda: (
                "int64",
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
            ),
        ]
    )
    def test_mul(self, _, left, right, skip_to_glow=False):
        """Basic test of the PyTorch mul Node on Glow."""

        utils.run_comparison_tests(
            SimpleMulModule(),
            (left, right),
            fusible_ops={"aten::mul"},
            skip_to_glow=skip_to_glow,
        )
