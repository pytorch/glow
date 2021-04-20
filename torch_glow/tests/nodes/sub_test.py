from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleSubtractModel(torch.nn.Module):
    def __init__(self):
        super(SimpleSubtractModel, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).sub(b.item())
        else:
            c = a.sub(b)
            return c.sub(c)


class TestSub(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleSubtractModel(), torch.randn(4), torch.randn(4)),
            lambda: (
                "broadcast_1",
                SimpleSubtractModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast_2",
                SimpleSubtractModel(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "broadcast_3",
                SimpleSubtractModel(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            lambda: ("float", SimpleSubtractModel(), torch.randn(4), torch.tensor(3.9)),
            lambda: (
                "int",
                SimpleSubtractModel(),
                torch.randn(4),
                torch.tensor(20),
                True,
            ),
            lambda: (
                "int64",
                SimpleSubtractModel(),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
            ),
        ]
    )
    def test_subtract(self, _, module, tensor, other, skip_to_glow=False):
        utils.compare_tracing_methods(module, tensor, other, skip_to_glow=skip_to_glow)
