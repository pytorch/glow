from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleDivModule(torch.nn.Module):
    def __init__(self):
        super(SimpleDivModule, self).__init__()

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).div(b.item())
        else:
            c = a.div(b)
            return c.div(c)


class TestDiv(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleDivModule(), torch.randn(4), torch.randn(4)),
            lambda: (
                "broadcast",
                SimpleDivModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast",
                SimpleDivModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "broadcast",
                SimpleDivModule(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            lambda: (
                "float_tensor",
                SimpleDivModule(),
                torch.randn(4),
                torch.tensor(3.9),
            ),
            lambda: (
                "int_tensor",
                SimpleDivModule(),
                torch.tensor([4]),
                torch.tensor([10]),
            ),
            # This one will go through (a * a) / b.item() and b.item() is an integer.
            lambda: (
                "int_number",
                SimpleDivModule(),
                torch.tensor([4]),
                torch.tensor(10),
            ),
        ]
    )
    def test_div(self, _, module, a, b):
        utils.compare_tracing_methods(module, a, b, fusible_ops={"aten::div"})
