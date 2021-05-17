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
            lambda: (
                "float_and_int",
                SimpleAddModule(),
                torch.randn(4),
                torch.tensor(42),
                True,
            ),
            lambda: (
                "int32",
                SimpleAddModule(),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int32),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int32),
            ),
            lambda: (
                "int64",
                SimpleAddModule(),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
                torch.torch.randint(-10, 10, (2, 4), dtype=torch.int64),
            ),
        ]
    )
    def test_add(self, _, module, a, b, skip_to_glow=False):
        utils.run_comparison_tests(
            module,
            (a, b),
            fusible_ops={"aten::add_"} if module.inplace else {"aten::add"},
        )
