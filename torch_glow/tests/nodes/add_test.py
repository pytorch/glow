from __future__ import absolute_import, division, print_function, unicode_literals

import torch
from tests import utils


class SimpleAddModule(torch.nn.Module):
    def __init__(self, inplace=False):
        super(SimpleAddModule, self).__init__()
        self.inplace = inplace

    def forward(self, a, b):
        if b.size() == torch.Size([]):
            return (a * a).add(b.item())
        if self.inplace:
            c = a.add_(b)
            return c.add_(c)
        else:
            c = a.add(b)
            return c.add(c)


class TestAdd(utils.TorchGlowTestCase):
    @utils.deterministic_expand(
        [
            lambda: ("basic", SimpleAddModule(), torch.randn(4), torch.randn(4)),
            lambda: ("inplace", SimpleAddModule(True), torch.randn(4), torch.randn(4)),
            lambda: (
                "broadcast",
                SimpleAddModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(4, 2),
            ),
            lambda: (
                "broadcast",
                SimpleAddModule(),
                torch.randn(8, 3, 4, 2),
                torch.randn(1, 2),
            ),
            lambda: (
                "broadcast",
                SimpleAddModule(),
                torch.randn(4, 2),
                torch.randn(8, 3, 4, 2),
            ),
            lambda: ("float", SimpleAddModule(), torch.randn(4), torch.tensor(1.2345)),
            lambda: ("int", SimpleAddModule(), torch.randn(4), torch.tensor(42), True),
        ]
    )
    def test_add(self, _, module, a, b, skip_to_glow=False):
        utils.compare_tracing_methods(
            module,
            a,
            b,
            skip_to_glow=skip_to_glow,
            fusible_ops={"aten::add_"} if module.inplace else {"aten::add"},
        )
